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
from rasterio.transform import rowcol
from rasterio.transform import Affine, from_origin
from rasterio.windows import bounds

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from rasterio.crs import CRS

import geopy
from geopy.distance import geodesic


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)

CLASS_COLOR = [BLUE, PURPLE]


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
        :param data (numpy array): The input data for the histogram.
        bins (int): Number of bins for the histogram. Default is 30.
        title (str): Title of the histogram. Default is "Histogram".
        xlabel (str): Label for the x-axis. Default is "Value".
        ylabel (str): Label for the y-axis. Default is "Frequency".
        color (str): Color of the histogram bars. Default is 'blue'.
        :param png_path:
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
    dangerous_slope_mask_epsg32633, pixel_size_y_m, pixel_size_x_m = create_slope_mask_epsg32633(
        dem_data=reprojected_array,
        transform=reprojected_transform,
        angle_threshold=angle_threshold
    )

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

    return dangerous_slope_mask, pixel_size_y_m, pixel_size_x_m


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
    with rasterio.Env():
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

    return mask, pixel_size_y_m, pixel_size_x_m


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
    - focal_length_mm: The true focal length of the camera in millimeters.
    - sensor_width_mm: The width of the camera sensor in millimeters.
    - sensor_height_mm: The height of the camera sensor in millimeters.
    - sensor_width_pixels: The sensor width in pixels (original sensor resolution).
    - sensor_height_pixels: The sensor height in pixels (original sensor resolution).
    - image_width_pixels: The output image width in pixels (final image resolution).
    - image_height_pixels: The output image height in pixels (final image resolution).

    Returns:
    - ground_resolution: Ground resolution in meters per pixels.
    """

    # Calculate ground resolution (in meters/pixel) for both axes using the full sensor resolution
    #    meters         millimeters
    # -------------- *  ------------
    #  millimeters        pixels

    ground_resolution = (rel_altitude_m / focal_length_mm) * (sensor_width_mm / sensor_width_pixels)

    # 4:3 is more square than 16:9
    # while the mapping from 5280 to 1920 still covers the whole sensor,
    # applying the same scaling factor to the height results in a picture longer than 1080 vertically (1440).
    # The excess pixels are cut off, but the pixels to meters relationship for height remains that of 1440
    # therefore the scaling factor is the same as for the width

    downsampling_factor_x = sensor_width_pixels / image_width_pixels
    downsampling_factor_y = sensor_height_pixels / image_height_pixels
    downsampling_factor = min(downsampling_factor_x, downsampling_factor_y)

    ground_resolution = ground_resolution * downsampling_factor

    return ground_resolution


def get_objects_coordinates(
        objects_coords,
        center_lat,
        center_lon,
        frame_width_pixels,
        frame_height_pixels,
        meters_per_pixel,
        angle_wrt_north
):

    # objects_coords must be a (N,2) numpy array
    assert isinstance(objects_coords, np.ndarray) \
           and len(objects_coords.shape) == 2 \
           and objects_coords.shape[1] == 2

    # get the (x,y) position of the center of the frame
    center_point_pixel_x = (frame_width_pixels - 1) / 2
    center_point_pixel_y = ((frame_height_pixels - 1) / 2) * (-1)

    print("center_point: ", (center_point_pixel_x, center_point_pixel_y))
    print("others: ", objects_coords)

    center_point_coords = geopy.Point(latitude=center_lat, longitude=center_lon)
    print("geo Center: ", center_point_coords)

    # Precompute distances of points from the center of the frame
    distances_y_m = (((-1) * objects_coords[:, 1]) - center_point_pixel_y) * meters_per_pixel  # Shape (N,)
    distances_x_m = (objects_coords[:, 0] - center_point_pixel_x) * meters_per_pixel  # Shape (N,)
    distances_m = np.sqrt(distances_x_m ** 2 + distances_y_m ** 2)  # Shape (N,)
    print("distances: ", distances_m)

    base_angles = np.degrees(np.atan2(distances_y_m, distances_x_m))
    print("base_angles: ", base_angles)
    print(angle_wrt_north)
    angles = (base_angles - angle_wrt_north)
    print("angles: ", angles)
    bearings = np.mod(90 - angles, 360)
    print("bearings: ", bearings)

    # compute target point coordinates by creating a circle of a certain radius around the start point
    # and identify the target point based on the bearing angle
    print("corners coordinates")
    final_coords = []
    for distance_m, bearing in zip(distances_m, bearings):
        destination = geopy.distance.geodesic(kilometers=distance_m/1000).destination(center_point_coords, bearing)
        final_coords.append([destination.longitude, destination.latitude])
        print(destination.latitude, destination.longitude)

    return np.array(final_coords)


def extract_masks_window(raster_dataset, center_lonlat, rectangle_lonlat):
    """
    Extracts a square window from a raster that fully encompasses a rotated rectangle.

    Args:
        raster_dataset (rasterio.DatasetReader): Opened raster dataset.
        center_lonlat (tuple): (longitude, latitude) of the center point.
        rectangle_lonlat (numpy.ndarray): (4,2) array of (longitude, latitude) rectangle corners.

    Returns:
        window_array (numpy.ndarray): Extracted raster window as a NumPy array.
        window_transform (Affine): Georeferencing transform of the extracted window.
    """
    # --- Step 1: Convert center point to pixel coordinates ---
    transform = raster_dataset.transform
    center_y, center_x = rowcol(transform=transform, xs=center_lonlat[0], ys=center_lonlat[1])

    # --- Step 2: Convert rectangle corners to pixel coordinates ---
    pixel_coords_yx = np.array([rowcol(transform=transform, xs=lon, ys=lat) for lon, lat in rectangle_lonlat])

    # Compute the maximum pixel distance from the center
    pixel_dists = np.linalg.norm(pixel_coords_yx - np.array([center_y, center_x]), axis=1)
    max_dist = int(np.max(pixel_dists))  # Maximum pixel distance

    # --- Step 3: Compute square window size (odd number with buffer) ---
    buffer = int(np.ceil(max_dist * 0.5))  # Extra space for rotation
    half_size = max_dist + buffer
    window_size = 2 * half_size + 1  # Ensure window is odd

    # --- Step 4: Define the window in pixel coordinates ---
    window_row_start = center_y - half_size
    window_col_start = center_x - half_size

    # --- Step 5: Extract the window from the raster ---
    window = rasterio.windows.Window(col_off=window_col_start, row_off=window_row_start, width=window_size, height=window_size)
    window_transform = raster_dataset.window_transform(window)

    # Read the window from the raster
    window_array = raster_dataset.read(window=window)

    # get the bounds of the window
    window_bounds = bounds(window, raster_dataset.transform)    # TODO original or window transform

    return window_array, window_transform, window_bounds, window_size


def get_window_size_m(reference_lat, window_bounds):
    (min_lon, min_lat, max_lon, max_lat) = window_bounds
    assert min_lat < reference_lat < max_lat

    # points in form (lat,long)
    point1 = (reference_lat, min_lon)
    point2 = (reference_lat, max_lon)
    distance_m = geodesic(point1, point2).meters

    return distance_m


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


def map_window_onto_drone_frame(
    window,
    window_transform,
    window_crs,
    center_coords,
    corners_coords,
    angle_wrt_north,
    frame_width,
    frame_height,
    window_size_pixels,
    window_size_m,
    frame_pixel_size_m,
):

    center_lon = center_coords[0]
    center_lat = center_coords[1]
    new_upper_left_lon = corners_coords[0, 0]
    new_upper_left_lat = corners_coords[0, 1]

    # divide the lenght of the window by the meters/pixels in the frame to get how many frame pixels the window would be
    # then divide the number frame-equivalent window pixels by the actual number of pixels in the window
    # to get the upscaling factor
    scaling_ratio = (window_size_m / frame_pixel_size_m) / window_size_pixels

    # --- 1. SCALE: Adjust pixel size to match target frame ---
    scaling = Affine.scale(scaling_ratio)

    # --- 2. TRANSLATE: Move rotation center point (the drone position) to (0,0) = upper left corner ---
    to_origin = Affine.translation(-center_lon, -center_lat)

    # --- 3. ROTATE: Apply rotation around the upper left corner ---
    rotation = Affine.rotation(angle_wrt_north)

    # rotation_angle = math.radians(angle_wrt_north)
    # cos_theta = math.cos(rotation_angle)
    # sin_theta = math.sin(rotation_angle)
    # rotation = Affine(
    #    cos_theta, -sin_theta, 0,
    #    sin_theta, cos_theta, 0
    # )

    # --- 4. TRANSLATE so that the upper left corner of the frame is the new (0,0) ---
    to_upper_left = Affine.translation(new_upper_left_lon, new_upper_left_lat)

    # Final combined transform: 4 <- 3 <- 2 <- 1
    rotated_rescaled_transform = to_upper_left * rotation * to_origin * scaling

    reprojected_mask = np.zeros((window.shape[0], frame_height, frame_width), dtype=np.uint8)
    with rasterio.Env():
        reproject(
            source=window,
            destination=reprojected_mask,
            src_transform=window_transform,
            src_crs=window_crs,
            dst_transform=rotated_rescaled_transform,
            dst_crs=window_crs,  # Assume the drone uses the same CRS as the binary mask (?)
            resampling=Resampling.nearest
        )

    return reprojected_mask


"""
def clip_array_into_rectangle_no_nodata(array, nodata):
    # Clip the rotated array into the bounding box of the valid (non-NaN) values.
    
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
"""


def send_alert(alerts_file, frame_id: int, danger_type: str = "Generic"):
    # Write alert to file
    alerts_file.write(f"Alert: Frame {frame_id} - Animal(s) near or in dangerous area.  Danger type: {danger_type}.\n")


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
        dangerous_mask_no_intersersection,
        intersection
):
    red_overlay = np.zeros_like(annotated_frame)
    yellow_overlay = np.zeros_like(annotated_frame)

    #red_overlay[dangerous_mask_no_intersersection == 1] = RED  # Red color channel only
    #yellow_overlay[intersection == 1] = YELLOW  # Red color channel only

    red_overlay[dangerous_mask_no_intersersection.astype(bool)] = RED  # Red color channel only
    yellow_overlay[intersection.astype(bool)] = YELLOW  # YELLOW color channel only

    cv2.addWeighted(red_overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
    cv2.addWeighted(yellow_overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)


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
):

    frame_height = annotated_frame.shape[0]

    # Dynamically scale font size and thickness based on frame height
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    base_font_scale = 0.001 * frame_height  # Scale with frame height
    base_thickness = max(1, int(0.002 * frame_height))  # Ensure thickness is at least 1
    text_color = (0, 0, 0)  # BLACK
    fill_color = (255, 255, 255)  # WHITE
    line_type = cv2.LINE_AA
    org = (10, frame_height - 10)  # Initial position of the bottom-left corner of the text

    # Count classes
    num_classes = classes.max() + 1
    class_counts = np.zeros(num_classes, dtype=np.int32)
    class_counts[: len(np.bincount(classes))] = np.bincount(classes)

    # Generate text lines
    lines = [f"N. Detected class {idx}: {count}" for idx, count in enumerate(class_counts)]

    # Measure text dimensions for all lines
    max_line_width = 0
    total_height = 0
    line_height = 0
    for line in lines:
        (line_width, line_height), _ = cv2.getTextSize(
            text=line,
            fontFace=font_face,
            fontScale=base_font_scale,
            thickness=base_thickness,
        )
        max_line_width = max(max_line_width, line_width)
        total_height += line_height + 5  # Add a little spacing between lines

    # Adjust text box coordinates (expand upward for all lines)
    textbox_coord_ul = (org[0] - 5, org[1] - total_height - 5)  # Expand upward
    textbox_coord_br = (org[0] + max_line_width + 5, org[1] + 5)

    # Draw white rectangle as background
    cv2.rectangle(annotated_frame, textbox_coord_ul, textbox_coord_br, fill_color, cv2.FILLED)

    # Draw each line of text inside the box
    y_offset = org[1] - total_height + line_height  # Start at the top line
    for line in lines:
        cv2.putText(
            img=annotated_frame,
            text=line,
            org=(org[0], y_offset),
            fontFace=font_face,
            fontScale=base_font_scale,
            color=text_color,
            thickness=base_thickness,
            lineType=line_type,
        )
        y_offset += line_height + 5  # Move down to the next line

    return annotated_frame

