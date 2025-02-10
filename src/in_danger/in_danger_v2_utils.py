from pathlib import Path
from time import time
import re
from shapely.geometry import box
from rasterio.warp import reproject, Resampling
from rasterio.transform import rowcol
from rasterio.windows import bounds
from affine import Affine
import rasterio
from rasterio.features import rasterize

import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def get_dem(dem_path):
    # Open the DEM
    dem_tif = rasterio.open(Path(dem_path))
    return dem_tif


def get_dem_mask(dem_mask_path):
    # Open the DEM mask if provided, otherwise assume all pixels are valid are return None
    if dem_mask_path is not None:
        dem_mask_tif = rasterio.open(Path(dem_mask_path))
    else:
        dem_mask_tif = None

    return dem_mask_tif


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
    boxes_corner1 = xyxy_boxes[:, :2]
    boxes_corner2 = xyxy_boxes[:, 2:]

    return classes, boxes_centers, boxes_corner1, boxes_corner2


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
    distances_x_m = (objects_coords[:, 0] - center_point_pixel_x) * meters_per_pixel  # distances on X, Shape (N,)
    distances_y_m = (((-1) * objects_coords[:, 1]) - center_point_pixel_y) * meters_per_pixel  # distances on Y, Shape (N,)
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


def extract_dem_window(dem_tif, dem_mask_tif, center_lonlat, rectangle_lonlat):
    """
    Extracts a square window from a raster that fully encompasses a rotated rectangle.

    Args:
        dem_tif (rasterio.DatasetReader): Opened raster dataset.
        dem_mask_tif (rasterio.DatasetReader): Opened raster mask dataset.
        center_lonlat (tuple): (longitude, latitude) of the center point.
        rectangle_lonlat (numpy.ndarray): (4,2) array of (longitude, latitude) rectangle corners.

    Returns:
        window_array (numpy.ndarray): Extracted raster window as a NumPy array.
        window_transform (Affine): Georeferencing transform of the extracted window.
    """
    # --- Step 1: Convert center point to pixel coordinates ---
    transform = dem_tif.transform
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

    window_row_end = center_y + half_size
    window_col_end = center_x + half_size

    # center in indexes (row=9, col=6), half size =3
    # => window_row_start = 6 ... |6|7|8| X |10|11|12
    # => window_col_start = 3 ... |3|4|5| X |7 |8 |9

    # --- Step 5: Make sure the window is inside the tif ---
    if (
            window_col_start < 0 or
            window_row_start < 0 or
            window_col_end > (dem_tif.width - 1) or
            window_row_end > (dem_tif.height - 1)
    ):
        print(f"ERROR: Cannot monitor the safety of animals when the drones is leaving the DEM area")
        print(f"DEM rows: {dem_tif.height}")
        print(f"DEM window rows: [{window_row_start}, {window_row_start + window_size}]")
        print(f"DEM columns: {dem_tif.width}")
        print(f"DEM window columns: [{window_col_start}, {window_col_start + window_size}]")
        exit()

    # --- Step 6: Extract the window from the raster ---
    window = rasterio.windows.Window(col_off=window_col_start, row_off=window_row_start, width=window_size, height=window_size)
    window_transform = dem_tif.window_transform(window)

    # Read the dem window from the raster
    dem_window_array = dem_tif.read(window=window)

    # Read the dem window from the raster (if None, assume mask alla values are valid)
    if dem_mask_tif is not None:
        dem_mask_window_array = dem_mask_tif.read(window=window)
    else:
        dem_mask_window_array = np.zeros((window_size, window_size), dtype=np.uint8)

    # get the bounds of the window
    window_bounds = bounds(window, dem_tif.transform)    # TODO original or window transform

    return dem_window_array, dem_mask_window_array, window_transform, window_bounds, window_size


def get_window_size_m(reference_lat, window_bounds):
    (min_lon, min_lat, max_lon, max_lat) = window_bounds
    assert min_lat < reference_lat < max_lat

    # points for geopy must be in form (lat,long)
    point1 = (reference_lat, min_lon)
    point2 = (reference_lat, max_lon)
    distance_m = geodesic(point1, point2).meters

    return distance_m


def compute_slope_mask_runtime(elev_array, pixel_size, slope_threshold_deg):
    """
    Compute a mask indicating where the terrain slope is steeper than a given threshold.

    Parameters
    ----------
    elev_array : np.ndarray
        A square 2D array of elevation values.
    pixel_size : float
        The size of each pixel in meters.
    slope_threshold_deg : float
        The slope threshold in degrees. Cells with a slope greater than this threshold
        will be marked with a 1 in the output mask.

    Returns
    -------
    np.ndarray
        A 2D array (of the same shape as elev_array) containing 1 where the slope is
        greater than slope_threshold_deg and 0 elsewhere.
    """
    # Compute the gradient along the y (row) and x (column) directions.
    # Note: np.gradient returns gradients in the order (axis0, axis1) which correspond
    # to (dy, dx) given the array dimensions.
    gy, gx = np.gradient(elev_array, pixel_size)

    # Compute the magnitude of the slope (rise over run)
    # The slope in radians is given by arctan(sqrt((dz/dx)^2 + (dz/dy)^2)).
    slope_radians = np.arctan(np.sqrt(gx ** 2 + gy ** 2))

    # Convert the slope to degrees
    slope_degrees = np.degrees(slope_radians)

    # Create a mask where a cell is 1 if the slope exceeds the threshold, 0 otherwise.
    mask = (slope_degrees > slope_threshold_deg).astype(np.uint8)

    return mask


def create_geofencing_mask_runtime(frame_width, frame_height, corners_coordinates, polygon):

    # Example: suppose your frame is a numpy array with shape (H, W)
    # (Here H is the number of rows, W is the number of columns)

    # Your 4 corner coordinates (longitude, latitude)
    # In order: upper-left, upper-right, bottom-right, bottom-left
    ul = corners_coordinates[0]
    ur = corners_coordinates[1]
    br = corners_coordinates[2]
    bl = corners_coordinates[3]

    # If we assume the mapping is affine we can use three of the points.
    # Here we assume that the pixel at (0,0) corresponds to ul,
    # the pixel at (W-1, 0) corresponds to ur, and
    # the pixel at (0, H-1) corresponds to bl.
    #
    # Then the affine transformation parameters are:
    #    x_geo = c + a * col + b * row
    #    y_geo = f + d * col + e * row
    #
    # So we set:
    a = (ur[0] - ul[0]) / (frame_width - 1)  # pixel width in x-direction
    d = (ur[1] - ul[1]) / (frame_width - 1)  # change in y per column
    b = (bl[0] - ul[0]) / (frame_height - 1)  # change in x per row
    e = (bl[1] - ul[1]) / (frame_height - 1)  # pixel height (often negative if latitude decreases downward)
    c = ul[0]
    f = ul[1]

    # Create the affine transform: it maps from (col, row) to (x, y)
    transform = Affine(a, b, c, d, e, f)

    # Now, suppose your Shapely polygon is defined as:
    # (You may have constructed it already in your code.)
    # polygon = Polygon([...])

    # Use rasterio.features.rasterize to create an array of shape (H, W)
    # The polygon will be burned with a value of 1 and all other pixels will be 0.
    mask_inside = rasterize(
        [(polygon, 1)],  # list of (geometry, value)
        out_shape=(frame_height, frame_width),
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    # At this point, mask_inside has value 1 for pixels inside the polygon and 0 outside.
    # Since you want the mask to be 0 inside and 1 outside, simply invert the mask:
    mask = 1 - mask_inside

    # 'mask' is now a numpy array of shape (H, W) with 0 inside the polygon and 1 outside.

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


def create_dangerous_intersections_masks(
    frame_height,
    frame_width,
    boxes_centers,
    safety_radius_pixels,
    segment_danger_mask,
    dem_nodata_danger_mask,
    geofencing_danger_mask,
    slope_danger_mask,

):
    crono_start = time()

    # create the safety mask
    inter = time()
    safety_mask = create_safety_mask(frame_height, frame_width, boxes_centers, safety_radius_pixels)
    st_time = time() - inter

    # create the intersection mask between safety areas and dangerous areas masks
    inter = time()
    intersection_segment = np.logical_and(safety_mask, segment_danger_mask)
    intersection_nodata = np.logical_and(safety_mask, dem_nodata_danger_mask)
    intersection_geofencing = np.logical_and(safety_mask, geofencing_danger_mask)
    intersection_slope = np.logical_and(safety_mask, slope_danger_mask)
    ands_time = time() - inter

    danger_types = []
    if np.any(intersection_segment > 0):
        danger_types.append("Vehicles Danger")
    if np.any(intersection_nodata > 0):
        danger_types.append("Missing DEM data Danger")
    if np.any(intersection_geofencing > 0):
        danger_types.append("Out of Geofenced area Danger DEM data")
    if np.any(intersection_slope > 0):
        danger_types.append("Steep slope Danger")

    inter = time()
    # compute combined danger mask
    combined_danger_mask = merge_3d_mask(np.stack([
        segment_danger_mask,
        dem_nodata_danger_mask,
        geofencing_danger_mask,
        slope_danger_mask,
    ]))
    dm_time = time() - inter

    # compute combined intersection mask
    inter = time()
    combined_intersections = merge_3d_mask(np.stack([
        intersection_segment,
        intersection_nodata,
        intersection_geofencing,
        intersection_slope
    ]))
    inter_time = time() - inter

    combined_danger_mask_no_intersections = combined_danger_mask - combined_intersections
    assert np.min(combined_danger_mask_no_intersections) >= 0 and np.max(combined_danger_mask_no_intersections) <= 1

    print(f"Danger analysis and reporting completed in {(time() - crono_start) * 1000:.1f} ms")
    print(f"\tCompute safety mask mask in {st_time * 1000:.1f} ms")
    print(f"\tCompute single intersections masks in {ands_time * 1000:.1f} ms")
    print(f"\tCompute combined danger mask in {dm_time * 1000:.1f} ms")
    print(f"\tCompute combined intersection mask in {inter_time * 1000:.1f} ms")

    return combined_danger_mask_no_intersections, combined_intersections, danger_types


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


def annotate_and_save_frame(
        annotated_writer,
        output_dir,
        frame,
        frame_id,
        cooldown_has_passed,
        danger_exists,
        classes,
        boxes_centers,
        boxes_corner1,
        boxes_corner2,
        safety_radius_pixels,
        danger_mask,
        intersection_mask,
):
    """ Additional annotations if videos are to be saved, or for frames where danger exist (74 ms)
    Optimization Opportunities:
    1. Batching Disk Writes:
    Disk I/O is one of the slowest parts of the process. Writing files frame by frame can be inefficient, especially if you’re saving many images.
    Solution: Buffer the frames (e.g., accumulate them in memory) and write them to disk periodically, or use a background thread/process for I/O.
    2. Avoid Repeated Path Creation:
    The Path object creation is relatively lightweight, but it can add up in tight loops.
    Solution: Pre-compute constant paths or reusable parts of the path.
    3. Optimize cv2.imwrite:
    cv2.imwrite is slower because it compresses images before saving.
    Solution: Use less compression or switch to a faster image format like .bmp if file size isn’t critical.
    4. Parallelize Save Operations:
    Writing frames and images can be offloaded to a background thread or separate process to avoid blocking the main execution.
    """

    crono_start = time()

    inter = time()
    annotated_frame = frame.copy()  # copy of the original frame on which to draw
    print(f"\tFrame copy generated in {(time() - inter) * 1000:.1f} ms")

    # draw safety circles
    inter = time()
    draw_safety_areas(annotated_frame, boxes_centers, safety_radius_pixels)
    print(f"\tsafety areas generated in {(time() - inter) * 1000:.1f} ms")

    # Overlay dangerous areas (in red) and intersections (in yellow) on the annotated frame
    inter = time()
    draw_dangerous_area(annotated_frame, danger_mask, intersection_mask)
    print(f"\tDangerous areas AND danger INTERSECTION drawn in {(time() - inter) * 1000:.1f} ms")

    # draw detection boxes
    inter = time()
    draw_detections(annotated_frame, classes, boxes_corner1, boxes_corner2)
    print(f"\tDetections drawn in {(time() - inter) * 1000:.1f} ms")

    # draw animal count
    inter = time()
    draw_count(classes, annotated_frame)
    print(f"\tAnimal Count drawn in {(time() - inter) * 1000:.1f} ms")

    inter = time()
    # save the annotated frame
    annotated_writer.write(annotated_frame)
    if cooldown_has_passed and danger_exists:  # save also an image for better identify the exact frame
        annotated_img_path = Path(output_dir, f"danger_frame_{frame_id}_annotated.png")
        cv2.imwrite(annotated_img_path, annotated_frame)
    print(f"\tFrame saving completed in {(time() - inter) * 1000:.1f} ms")

    print(f"Video annotations completed in {(time() - crono_start) * 1000:.1f} ms")

