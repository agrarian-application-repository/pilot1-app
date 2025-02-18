from pathlib import Path
from time import time
import re
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

__all__ = [
    "RED",
    "GREEN",
    "BLUE",
    "YELLOW",
    "WHITE",
    "BLACK",
    "PURPLE",
    "get_dem",
    "get_dem_mask",
    "perform_detection",
    "perform_segmentation",
    "parse_drone_frame",
    "get_meters_per_pixel",
    "get_objects_coordinates",
    "extract_dem_window",
    "get_window_size_m",
    "compute_slope_mask_runtime",
    "create_geofencing_mask_runtime",
    "get_frame_transform",
    "map_window_onto_drone_frame",
    "create_dangerous_intersections_masks",
    "send_alert",
    "create_safety_mask",
    "annotate_and_save_frame",
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


def _parse_line_to_dict(line):
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

    return _parse_line_to_dict(useful_line)


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
            window_col_end >= dem_tif.width or
            window_row_end >= dem_tif.height
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
        dem_mask_window_array = np.zeros((1, window_size, window_size), dtype=np.uint8)

    # get the bounds of the window
    window_bounds = bounds(window, dem_tif.transform)

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
        A square 3D array of elevation values in meters, with first dimension has shape 1.
    pixel_size : float
        The size of each pixel in meters.
    slope_threshold_deg : float
        The slope threshold in degrees. Cells with a slope greater than this threshold
        will be marked with a 1 in the output mask.

    Returns
    -------
    np.ndarray
        A 3D array (of the same shape as elev_array) containing 1 where the slope is
        greater than slope_threshold_deg and 0 elsewhere.
    """

    # If elev_array has a singleton first dimension, remove it to work with a 2D array.
    assert elev_array.ndim == 3 and elev_array.shape[0] == 1
    elev_array = elev_array[0]

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

    # Expand the dimensions to ensure the output is (1, W, H).
    mask = mask[np.newaxis, :, :]

    return mask


def create_geofencing_mask_runtime(frame_width, frame_height, transform, polygon):

    # Use rasterio.features.rasterize to create an array of shape (H, W)
    # The inside of the polygon will be burned with a value of 0
    # all external pixels will be 1
    mask = rasterize(
        [(polygon, 0)],  # list of (geometry, value)
        out_shape=(frame_height, frame_width),
        transform=transform,
        fill=1,
        dtype=np.uint8
    )

    return mask


def get_frame_transform(
        height,
        width,
        drone_ul,  # (lon, lat) for upper-left
        drone_ur,  # (lon, lat) for upper-right
        drone_bl,  # (lon, lat) for bottom-left
):

    # Build dst_transform using the known drone frame corners.
    # Here, we assume:
    #  (0, 0)             --> drone_ul
    #  (width-1, 0)        --> drone_ur
    #  (0, height-1)       --> drone_bl
    #

    # Build the affine transform.
    # In Rasterio, the transform maps (col, row) to (x, y) as:
    #    x = a * col + b * row + c
    #    y = d * col + e * row + f

    c, f = drone_ul  # The translation (c, f) is just the UL coordinate
    a = (drone_ur[0] - drone_ul[0]) / (width - 1)  # change in x per column
    d = (drone_ur[1] - drone_ul[1]) / (width - 1)  # change in y per column
    b = (drone_bl[0] - drone_ul[0]) / (height - 1)  # change in x per row
    e = (drone_bl[1] - drone_ul[1]) / (height - 1)  # change in y per row

    dst_transform = Affine(a, b, c, d, e, f)
    return dst_transform


def map_window_onto_drone_frame(
        window,
        window_transform,
        dst_transform,
        output_shape=(2, 1080, 1920),
        crs='EPSG:4326'
):
    """
    Map the DEM window to the drone frame using the output transform dst_transform,
    which is built from the provided drone frame corner coordinates.
    """
    # Reproject DEM into the output frame using dst_transform directly.
    out_array = np.empty(output_shape, dtype=window.dtype)
    reproject(
        source=window,
        destination=out_array,
        src_transform=window_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.nearest
    )

    return out_array


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
        num_classes,
        classes_names,
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
    class_counts = np.zeros(num_classes, dtype=np.int32)
    class_counts[: len(np.bincount(classes))] = np.bincount(classes)

    # Generate text lines
    lines = [f"N. {classes_names[idx]}: {count}" for idx, count in enumerate(class_counts)]

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
        num_classes,
        classes_names,
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
    draw_count(classes, num_classes, classes_names, annotated_frame)
    print(f"\tAnimal Count drawn in {(time() - inter) * 1000:.1f} ms")

    inter = time()
    # save the annotated frame
    annotated_writer.write(annotated_frame)
    if cooldown_has_passed and danger_exists:  # save also an image for better identify the exact frame
        annotated_img_path = Path(output_dir, f"danger_frame_{frame_id}_annotated.jpg")
        cv2.imwrite(annotated_img_path, annotated_frame)
    print(f"\tFrame saving completed in {(time() - inter) * 1000:.1f} ms")

    print(f"Video annotations completed in {(time() - crono_start) * 1000:.1f} ms")

