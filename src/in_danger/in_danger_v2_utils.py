from pathlib import Path
from time import time
from typing import Any
import re
import math
import rasterio

from shapely.vectorized import contains as vectorized_contains
from shapely.geometry import Polygon, box
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rasterio_mask

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from rasterio.crs import CRS

# from skimage import transform as skt


RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (255, 0, 0)
PURPLE = (128, 0, 128)
CLASS_COLOR = [BLUE, PURPLE]

shades_of_red = [
    (255, 0, 0),  # Bright Red
    (139, 0, 0),  # Dark Red
    (255, 102, 102),  # Light Red
    (220, 20, 60),  # Crimson Red
    (205, 92, 92)  # Indian Red
]

shades_of_yellow = [
    (255, 255, 0),  # Bright Yellow (similar to Bright Red)
    (204, 204, 0),  # Dark Yellow (similar to Dark Red)
    (255, 255, 102),  # Light Yellow (similar to Light Red)
    (255, 215, 0),  # Golden Yellow (similar to Crimson Red)
    (255, 160, 0)  # Amber Yellow (similar to Indian Red)
]


def plot_2d_array(array, png_path, title="2D Array Plot", cmap="viridis", colorbar=True):
    """
    Plots a 2D NumPy array.

    Parameters:
    - array (numpy.ndarray): The 2D array to plot.
    - title (str): Title of the plot.
    - cmap (str): Colormap for the plot.
    - colorbar (bool): Whether to include a colorbar.
    """
    if array.ndim != 2:
        raise ValueError("The input array must be 2-dimensional.")

    plt.figure(figsize=(8, 6))
    plt.imshow(array, cmap=cmap, origin='upper')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    if colorbar:
        plt.colorbar(label="Value")
    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"PNG saved to: {png_path}")


def plot_histogram(data, png_path, bins=100, title="Histogram", xlabel="Value", ylabel="Frequency", color='blue'):
    """
    Plot a histogram from a NumPy array.

    Parameters:
        data (numpy array): The input data for the histogram.
        bins (int): Number of bins for the histogram. Default is 30.
        title (str): Title of the histogram. Default is "Histogram".
        xlabel (str): Label for the x-axis. Default is "Value".
        ylabel (str): Label for the y-axis. Default is "Frequency".
        color (str): Color of the histogram bars. Default is 'blue'.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data.ravel(), bins=bins, color=color, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

    print(f"Histogram saved to: {png_path}")


def get_dem(dem_path, output_dir, plot):
    # Open the DEM
    dem_tif = rasterio.open(Path(dem_path))
    dem_np = dem_tif.read(1)  # Read the first band (1-index)

    if plot:
        plot_2d_array(dem_np, output_dir/"dem.png", title="DEM")
        plot_histogram(dem_np, output_dir/"hist_dem.png", title="DEM histogram")

    return dem_tif, dem_np


def get_dem_mask(dem_mask_path, fallback_mask_shape, output_dir, plot=False):
    # Open or create the DEM mask
    if dem_mask_path is not None:
        dem_mask_tif = rasterio.open(Path(dem_mask_path))
        nodata_dem_mask = dem_mask_tif.read(1)  # Read the first band (1-index)
        nodata_dem_mask = nodata_dem_mask.astype(dtype=np.uint8)  # convert to int8
        terminate_if_no_valid_pixels(nodata_dem_mask)  # if all the dem is invalid, terminate
    else:
        # if no mask is provided, assume all pixels are valid
        dem_mask_tif = None
        nodata_dem_mask = np.zeros(fallback_mask_shape, dtype=np.uint8)

    if plot:
        plot_2d_array(nodata_dem_mask, output_dir / "dem_mask.png", title="nodata DEM mask")
        plot_histogram(nodata_dem_mask, output_dir / "hist_dem_mask.png", title="nodata DEM histogram")

    return dem_mask_tif, nodata_dem_mask


def get_geofencing_mask(dem_tif, geofencing_vertexes, fallback_mask_shape, output_dir, plot=False):
    # Create a mask highlighting the areas outside the geofenced boundaries
    if geofencing_vertexes is not None:
        geofencing_mask = create_polygon_mask(dem_tif, geofencing_vertexes, crop=False)
    else:
        # if no geofencing vertextes are provided, assume all pixels are valid
        geofencing_mask = np.zeros(fallback_mask_shape, dtype=np.uint8)

    if plot:
        plot_2d_array(geofencing_mask, output_dir / "geofencing_mask.png", title="geofenced DEM mask")
        plot_histogram(geofencing_mask, output_dir / "hist_geofencing_mask.png", title="geofenced DEM histogram")

    return geofencing_mask


def create_dangerous_slope_mask(tif, array, angle_threshold, output_dir, plot=False):

    # project the DEM array into EPSG:32633 to get info on ground resolution of the DEM
    reprojected_array, reprojected_transform = reproject_array(
        src_array=array,
        src_transform=tif.transform,
        src_crs=tif.crs,
        src_width=tif.width,
        src_height=tif.height,
        src_bounds=tif.bounds,
        src_dtype=tif.meta['dtype'],
        dst_crs="EPSG:32633",
    )

    # create slope map using meter-converted dem
    dangerous_slope_mask_epsg32633 = create_slope_mask_epsg32633(reprojected_array, reprojected_transform, angle_threshold)

    # convert DEM array back into WSG84 reference system
    dangerous_slope_mask, _ = reproject_array(
        src_array=dangerous_slope_mask_epsg32633,
        src_transform=reprojected_transform,
        src_crs="EPSG:32633",
        src_width=dangerous_slope_mask_epsg32633.shape[1],
        src_height=dangerous_slope_mask_epsg32633.shape[0],
        src_bounds=tif.bounds,
        src_dtype=dangerous_slope_mask_epsg32633.dtype,
        dst_crs="WGS84",
        dst_transform=tif.transform,
        dst_width=tif.width,
        dst_height=tif.height,
    )

    dangerous_slope_mask = dangerous_slope_mask.astype(np.uint8)

    if plot:
        plot_2d_array(dangerous_slope_mask, output_dir/"slope_mask.png", title="slope DEM mask")
        plot_histogram(dangerous_slope_mask, output_dir/"hist_slope_mask.png", title="slope DEM histogram")

    return dangerous_slope_mask


def reproject_array(
    src_array,
    src_transform,
    src_crs,
    src_width,
    src_height,
    src_bounds,
    src_dtype,
    dst_crs,
    dst_transform=None,
    dst_width=None,
    dst_height=None,
    resampling=Resampling.nearest
):
    """
    Reproject a 2D NumPy array to a new CRS.

    Parameters:
        src_array (numpy array): Source data array.
        src_transform (Affine): Affine transform of the source array.
        src_crs (str): Source CRS (e.g., "EPSG:4326").
        src_width (int): Width of the source array.
        src_height (int): Height of the source array.
        src_bounds (tuple): Bounds of the source array (left, bottom, right, top).
        src_dtype (numpy dtype): Data type of the source array.
        dst_crs (str): Destination CRS (e.g., "EPSG:32633").
        dst_transform (Affine, optional): Affine transform of the destination array.
        dst_width (int, optional): Width of the destination array.
        dst_height (int, optional): Height of the destination array.
        resampling (rasterio.warp.Resampling, optional): Resampling method. Default is Resampling.nearest.

    Returns:
        numpy array: Reprojected data array.
        Affine: Affine transform of the reprojected data.
    """

    assert (dst_transform is None and dst_width is None and dst_height is None) or \
           (dst_transform is not None and dst_width is not None and dst_height is not None), \
        "Target transform and dimensions must either all be provided or all omitted."

    src_crs = src_crs.to_string() if isinstance(src_crs, CRS) else src_crs
    dst_crs = dst_crs.to_string() if isinstance(dst_crs, CRS) else dst_crs

    if dst_transform is None and dst_width is None and dst_height is None:
        # Calculate transform and new dimensions
        left, bottom, right, top = src_bounds
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs=src_crs,
            dst_crs=dst_crs,
            width=src_width,
            height=src_height,
            left=left,
            bottom=bottom,
            right=right,
            top=top
        )

    # Prepare the destination array
    reprojected_data = np.empty((dst_height, dst_width), dtype=src_dtype)

    # Perform the reprojection
    reproject(
        source=src_array,  # Reprojecting the first band
        destination=reprojected_data,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )

    return reprojected_data, dst_transform


def create_slope_mask_wsg84(dem_data, bounds, transform, angle_threshold):
    """
    Create a mask where the terrain inclination exceeds a certain slope angle.

    Parameters:
        dem_data (numpy array): 3D array (1,H,W) of elevation values.
        transform (Affine): Affine transformation of the DEM.
        angle_threshold (float): Slope angle threshold in degrees.

    Returns:
        numpy array: Mask with 1 where the slope exceeds the angle, 0 otherwise.
    """
    # Calculate pixel size (resolution) in meters
    pixel_size_x_deg = transform[0]  # Cell width (in degrees)
    pixel_size_y_deg = -transform[4]  # Cell height (in degrees, negative because of affine conventions)

    # Get the center latitude of the raster
    center_lat = (bounds[1] + bounds[3]) / 2  # Average of top and bottom latitude

    # Convert pixel size from degrees to meters
    pixel_size_x_m = pixel_size_x_deg * 111320 * math.cos(math.radians(center_lat))
    pixel_size_y_m = pixel_size_y_deg * 111320

    print(f"DEM pixel size resolution on x: {pixel_size_x_m}m")
    print(f"DEM pixel size resolution on y: {pixel_size_y_m}m")

    # Compute gradients (dz/dx and dz/dy)
    dz_dy, dz_dx = np.gradient(dem_data, pixel_size_y_m, pixel_size_x_m)

    # Compute slope in radians
    slope_radians = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # Convert slope to degrees
    slope_degrees = np.degrees(slope_radians)

    # Create mask where slope exceeds threshold
    mask = (slope_degrees > angle_threshold).astype(np.uint8)

    return mask


def create_slope_mask_epsg32633(dem_data, transform, angle_threshold):
    """
    Create a mask where the terrain inclination exceeds a certain slope angle.

    Parameters:
        dem_data (numpy array): 3D array (1,H,W) of elevation values.
        transform (Affine): Affine transformation of the DEM.
        angle_threshold (float): Slope angle threshold in degrees.

    Returns:
        numpy array: Mask with 1 where the slope exceeds the angle, 0 otherwise.
    """
    # Calculate pixel size (resolution) in meters
    pixel_size_x_m = transform[0]  # Cell width (in meters)
    pixel_size_y_m = -transform[4]  # Cell height (in meters, negative because of affine conventions)

    print(f"DEM pixel resolution (X): {pixel_size_x_m:.2f} meters")
    print(f"DEM pixel resolution (Y): {pixel_size_y_m:.2f} meters")

    # Compute gradients (dz/dx and dz/dy)
    dz_dy, dz_dx = np.gradient(dem_data, pixel_size_y_m, pixel_size_x_m)

    # Compute slope in radians
    slope_radians = np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2))

    # Convert slope to degrees
    slope_degrees = np.degrees(slope_radians)

    # Create mask where slope exceeds threshold
    mask = (slope_degrees > angle_threshold).astype(np.uint8)

    return mask


# TODO check and bugfixing
def create_polygon_mask(tif, lon_lat_tuples_list, crop):
    """
    Create a mask for a GeoTIFF where everything outside a given polygon is set to 1.

    Parameters:
        tif (rasterio.io.DatasetReader): Opened GeoTIFF file.
        lon_lat_tuples_list (list of tuple): List of (longitude, latitude) tuples defining the polygon vertexes.
        crop (bool): whether to return a smaller array describing only the area around the area inside the polygon

    Returns:
        numpy.ndarray: A mask with 1 outside the polygon and 0 inside.
    """
    # Convert the list of lat-lon tuples into a Shapely Polygon
    polygon = Polygon(lon_lat_tuples_list)

    try:
        # from the tif, all pixels outside the polygon are set to np.nan
        masked_array, masked_tif_transform = rasterio_mask(tif, [polygon], nodata=np.nan, crop=crop)
        # create a mask by setting the value to 1 where the tif array is nan, leaving zeros inside the polygon
        mask = np.isnan(masked_array[0]).astype(np.uint8)
    except rasterio.errors.WindowError:
        # if the polygon is completely outside of the tif area, mask all the area
        print(f"Polygon is outside the tif. Bounds: {tif.bounds}, Polygon: {polygon}")
        mask = np.ones((tif.height, tif.width), dtype=np.uint8)

    return mask

"""
def extract_image_from_tif(tif, lon_lat_tuples_list, crop):
    # Convert the list of lat-lon tuples into a Shapely Polygon
    polygon = Polygon(lon_lat_tuples_list)

    try:
        # from the tif, all pixels outside the polygon are set to np.nan
        masked_array, masked_tif_transform = rasterio_mask(tif, [polygon], nodata=np.nan, crop=crop)
        # create a mask by setting the value to 1 where the tif array is nan, leaving zeros inside the polygon
        mask = np.isnan(masked_array[0]).astype(np.uint8)
"""


def terminate_if_no_valid_pixels(array):
    if np.all(array == 1):
        print("No valid/safe pixels found in the DEM: TERMINATING")
        exit()


def perform_detection(detector, frame, detection_args):
    # Detect animals in frame
    detection_results = detector.predict(source=frame, **detection_args)

    # Parse detection results to get bounding boxes
    classes = detection_results[0].boxes.cls.cpu().numpy().astype(int)
    xywh_boxes = detection_results[0].boxes.xywh.cpu().numpy().astype(int)
    xyxy_boxes = detection_results[0].boxes.xyxy.cpu().numpy().astype(int)

    # Create additional variables to store useful info from the detections
    boxes_centers = xywh_boxes[:, :2]
    boxes_wh = xywh_boxes[:, 2:]
    boxes_corner1 = xyxy_boxes[:, :2]
    boxes_corner2 = xyxy_boxes[:, 2:]

    return classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2


def perform_segmentation(segmenter, frame, segmentation_args):

    # Highlight dangerous objects
    segment_results = segmenter.predict(source=frame, **segmentation_args)

    # frame size (H, W, 3)
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    if segment_results[0].masks is not None:  # danger found in the frame
        masks = segment_results[0].masks.data.int().cpu().numpy()
        segment_danger_mask = np.any(masks, axis=0).astype(np.uint8)
        segment_danger_mask = cv2.resize(segment_danger_mask, dsize=(frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

    else:  # mask not found in frame
        segment_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    return segment_danger_mask


def merge_3d_mask(mask_3d):
    return np.logical_or.reduce(mask_3d, axis=0).astype(np.uint8)


def parse_line_to_dict(line):
    # Regex to capture key-value pairs
    pattern = r'(\w+)\s*:\s*([\w\.\-/]+)'

    # Use re.findall to extract all key-value pairs
    matches = re.findall(pattern, line)

    # Convert the matches to a dictionary
    result = {key: _convert_value(value) for key, value in matches}

    return result


def _convert_value(value):
    """Convert the value to the appropriate type: int, float, or keep as string."""
    try:
        # Attempt to convert to int
        return int(value)
    except ValueError:
        try:
            # Attempt to convert to float
            return float(value)
        except ValueError:
            # Keep as string if it can't be converted
            return value

"""
def parse_drone_frame(file, frame_id):
    lines = file.readlines()

    current_frame = None

    for i, line in enumerate(lines):
        line = line.strip()
        # Check if the line is a frame number
        if line.isdigit():
            current_frame = int(line)

        # When the current frame matches the desired frame_id
        if current_frame == frame_id:
            # The useful line is 4 lines after the frame number
            useful_line = lines[i + 4].strip()
            return parse_line_to_dict(useful_line)

    # If no matching frame or data found
    raise ValueError(f"Frame ID {frame_id} or flight data not found.")
"""


def parse_drone_frame(file, frame_id):
    # Read all lines from the file
    lines = file.readlines()

    # Calculate the starting line for the desired frame_id
    frame_start_line = (frame_id - 1) * 6  # Each frame has 6 lines: 1 frame number + 4 data lines + 1 space before next frame
    # Check if the required lines are within the file's length
    if frame_start_line + 5 > len(lines):
        raise ValueError(f"Frame {frame_id} not found or in the file (frames info are supposed to be sequential).")

    # Extract the 5 lines for the desired frame
    frame_lines = lines[frame_start_line:frame_start_line + 5]

    # The useful line is the 5th line in the block (index 4)
    useful_line = frame_lines[4].strip()

    # rewind to file beginning to avoid error on next readlines()
    file.seek(0)

    return parse_line_to_dict(useful_line)


# TODO check formula, fix downsampling cator depeding on which sensor dimension is the largest
def get_meters_per_pixel(
        rel_altitude_m: float,
        focal_length_mm: float,
        sensor_width_mm: float,
        sensor_height_mm: float,
        sensor_width_pixels: int,
        sensor_height_pixels: int,
        image_width_pixels: int,
        image_height_pixels: int,
):
    """
    Converts pixels to meters using the pinhole camera model for a drone camera pointing at the ground.
    This function uses the full camera sensor resolution to compute the ground resolution.
    Afterward, it scales the ground resolution to match the output image resolution.

    Parameters:
    - rel_altitude_m: The altitude of the drone in meters.
    - focal_length_mm: The focal length of the camera in millimeters.
    - sensor_width_mm: The width of the camera sensor in millimeters.
    - sensor_height_mm: The height of the camera sensor in millimeters.
    - sensor_width_pixels: The sensor width in pixels (original sensor resolution).
    - sensor_height_pixels: The sensor height in pixels (original sensor resolution).
    - image_width_pixels: The output image width in pixels (final image resolution).
    - image_height_pixels: The output image height in pixels (final image resolution).

    Returns:
    - ground_resolution_x: Ground resolution in pixels per meter along the x-axis (horizontal).
    - ground_resolution_y: Ground resolution in pixels per meter along the y-axis (vertical).
    """

    # Calculate ground resolution (in meters/pixel) for both axes using the full sensor resolution
    #    meters         millimeters
    # -------------- *  ------------
    #  millimeters        pixels

    # TODO ASSERT sensor_width_mm/sensor_height_mm == sensor_width_pixels/sensor_height_pixels

    ground_resolution = (rel_altitude_m / focal_length_mm) * (sensor_width_mm / sensor_width_pixels)

    # 4:3 is more square than 16:9
    # while the mapping from 5184 to 1920 still covers the whole sensor,
    # applying the same scaling factor to the height results in a picture longer than 1080 vertically (1440).
    # The excess pixels are cut off, but the pixels to meters relationship for height remains that of 1440
    # therefore the scaling factor is the same as for the width

    downsampling_factor_x = sensor_width_pixels / image_width_pixels
    downsampling_factor_y = sensor_height_pixels / image_height_pixels
    downsampling_factor = min(downsampling_factor_x, downsampling_factor_y)

    ground_resolution = ground_resolution * downsampling_factor

    return ground_resolution


# TODO check and bugfixing
def get_objects_coordinates(
        objects_coords,
        center_lat,
        center_lon,
        frame_width_pixels,
        frame_height_pixels,
        meters_per_pixel,
        angle_wrt_north
):
    """
    Calculate the coordinates of the four corners of a rectangle.

    Parameters:
        ...

    Returns:
        list of tuple: Coordinates of the four corners in (lon, lat).
    """

    # objects_coords must be a (N,2) numpy array
    assert isinstance(objects_coords, np.ndarray) \
           and len(objects_coords.shape) == 2 \
           and objects_coords.shape[1] == 2

    # get the (x,y) position of the center of the frame
    center_point_pixel_x = frame_width_pixels / 2
    center_point_pixel_y = frame_height_pixels / 2

    # Convert center point lat/long to radians
    center_lat_rad = math.radians(center_lat)
    center_lon_rad = math.radians(center_lon)
    theta_rad = math.radians(angle_wrt_north)

    # Earth radius in meters
    earth_radius_m = 6378137

    # Precompute corner angles relative to the center
    distances_x_m = (objects_coords[:, 0] - center_point_pixel_x) * meters_per_pixel  # Shape (N,)
    distances_y_m = (objects_coords[:, 1] - center_point_pixel_y) * meters_per_pixel  # Shape (N,)

    # Compute total Euclidean distances
    distances_m = np.sqrt(distances_x_m ** 2 + distances_y_m ** 2)  # Shape (N,)

    # Compute angles in radians using atan2
    angles = np.atan2(distances_y_m, distances_x_m) + theta_rad  # Shape (N,)

    # Latitude change in radians
    dlats = (distances_m / earth_radius_m) * np.cos(angles)
    # Longitude change in radians
    dlons = (distances_m / (earth_radius_m * math.cos(center_lat_rad))) * np.sin(angles)

    # Convert back to degrees
    corners_lats = np.degrees(dlats + center_lat_rad)
    corners_longs = np.degrees(dlons + center_lon_rad)

    corners = np.stack((corners_longs, corners_lats), axis=-1)

    return corners


def create_runtime_geofencing_mask(min_long, min_lat, max_long, max_lat, polygon, resolution_x, resolution_y):
    # TODO what to do with rotation (min/max is upper or lower?)
    """
    Create a geofencing mask for a polygon within given geographic bounds.

    Parameters:
    - min_long: Minimum longitude.
    - min_lat: Minimum latitude.
    - max_long: Maximum longitude.
    - max_lat: Maximum latitude.
    - polygon: Shapely Polygon object.
    - resolution_x: Number of grid points along the horizontal (longitude) direction
    - resolution_y: Number of grid points along the horizontal (vertical) direction

    Returns:
    - mask: 2D boolean numpy array of shape (resolution, resolution).
    - longitudes: 2D array of longitude values.
    - latitudes: 2D array of latitude values.
    """

    # Create a grid of longitude and latitude values
    # min_long/max_lat ------- max_long/max_lat
    #       |                           |
    #       |                           |
    #       |                           |
    #       |                           |
    # min_long/min_lat ------- max_long/min_lat
    longitudes = np.linspace(min_long, max_long, resolution_x)
    latitudes = np.linspace(max_lat, min_lat, resolution_y)
    long_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    # Create the mask using shapely's vectorized contains
    mask = vectorized_contains(polygon, long_grid, lat_grid)
    # Invert the mask: True -> 0 (inside the polygon), False -> 1 (outside the polygon)
    mask = (~mask).astype(np.uint8)
    return mask


def is_polygon_within_bounds(bounds, polygon):
    """
    Check if the given shapely polygon is completely contained within the bounds of a GeoTIFF image.

    Parameters:
    - polygon (shapely.geometry.Polygon): The polygon to check.
    - bounds (tuple): The bounds of the GeoTIFF in the form (minx, miny, maxx, maxy).

    Returns:
    - bool: True if the polygon is completely within the bounds, False otherwise.
    """

    # Create a box (polygon) from the bounds
    raster_bounds = box(*bounds)  # Create a polygon from the bounds tuple
    # Check if the polygon is completely within the raster bounds
    is_inside = polygon.within(raster_bounds)
    return is_inside


def clip_array_into_rectangle_no_nodata(array, nodata):
    """Clip the rotated array into the bounding box of the valid (non-NaN) values."""
    # Find indices of non-NaN values
    valid_rows = np.any(array != nodata, axis=1)
    valid_cols = np.any(array != nodata, axis=0)

    # Find the bounding box
    min_row, max_row = np.where(valid_rows)[0][[0, -1]]
    min_col, max_col = np.where(valid_cols)[0][[0, -1]]

    # Crop to the bounding box
    return array[min_row:max_row+1, min_col:max_col+1]


def upscale_array_to_image_size(array, target_height, target_width):
    array_hwc = np.transpose(array, (1, 2, 0))
    resized_array_hcw = cv2.resize(array_hwc, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    resized_chw = np.transpose(resized_array_hcw, (2, 0, 1))
    return resized_chw


def send_alert(alerts_file, frame_id: int, danger_type:str="Generic"):
    # Write alert to file
    alerts_file.write(f"Alert: True Frame {frame_id} - Animal(s) near or in dangerous area.  Danger type: {danger_type}.\n")


def create_safety_mask(frame_height, frame_width, boxes_centers, safety_radius):
    # Initialize the mask with zeros
    safety_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Draw circles on the mask
    for box_center in boxes_centers:
        cv2.circle(safety_mask, box_center, safety_radius, 1, cv2.FILLED)

    return safety_mask


def draw_safety_areas(
        annotated_frame,
        boxes_centers,
        safety_radius,
):
    # drawing safety circles & detection boxes
    for box_center in boxes_centers:
        # Draw safety circle on annotated frame (green)
        cv2.circle(annotated_frame, box_center, safety_radius, GREEN, 2)


def draw_dangerous_area(
        annotated_frame,
        dangerous_mask,
        intersection
):
    red_overlay = np.zeros_like(annotated_frame)

    red_overlay[dangerous_mask == 1] = RED  # Red color channel only
    # red_overlay[dangerous_mask.astype(bool)] = RED  # Red color channel only

    annotated_frame = cv2.addWeighted(red_overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)

    # Dangerous intersection areas in YELLOW on mask frame
    annotated_frame[np.where((dangerous_mask - intersection) > 0)] = YELLOW
    # annotated_frame[intersection.astype(bool)] = YELLOW


def draw_detections(
        annotated_frame,
        classes,
        boxes_corner1,
        boxes_corner2,
):
    # drawing safety circles & detection boxes
    for obj_class, box_corner1, box_corner2 in zip(classes, boxes_corner1, boxes_corner2):
        # Draw bounding box on annotated frame (blue sheep, purple goat), on top of safety circles
        cv2.rectangle(annotated_frame, box_corner1, box_corner2, CLASS_COLOR[obj_class], 2)


def draw_count(
        classes,
        annotated_frame,
        frame_height,
):
    # Overlay the count text on the annotated frame
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = BLACK
    fill_color = WHITE
    thickness = 1
    line_type = cv2.LINE_AA
    org = (10, frame_height - 10)  # text position in image

    num_classes = classes.max() + 1
    class_counts = np.zeros(num_classes, dtype=np.int32)
    class_counts[: len(np.bincount(classes))] = np.bincount(classes)

    text = ""
    for idx, count in enumerate(class_counts):
        text = f"N. Detected class {idx}: {count}\n"

    (text_width, text_height), _ = cv2.getTextSize(
        text=text,
        fontFace=font_face,
        fontScale=font_scale,
        thickness=thickness
    )
    textbox_coord_ul = (org[0] - 5, org[1] - text_height - 5)
    textbox_coord_br = (org[0] + text_width + 5, org[1] + 5)

    # Draw white rectangle as background
    cv2.rectangle(annotated_frame, textbox_coord_ul, textbox_coord_br, fill_color, cv2.FILLED)

    cv2.putText(
        img=annotated_frame,
        text=text,
        org=org,
        fontFace=font_face,
        fontScale=font_scale,
        color=text_color,
        thickness=thickness,
        lineType=line_type,
    )
