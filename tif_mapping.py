from rasterio.transform import from_origin
import cv2
from src.in_danger.in_danger_v2_utils import get_window_size_m, extract_dem_window
import matplotlib.pyplot as plt
import numpy as np
from affine import Affine
import rasterio
from rasterio.warp import reproject, Resampling


def fake_tif():
    array0 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    array1 = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ])

    array2 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    array3 = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    ])

    array4 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    array5 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    array6 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    array7 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    return [array1, array2, array3, array4, array5, array6, array7]


def create_centered_tif_with_snapping(original_tif_path, output_tif_path, center_point, binary_array):
    """
    Creates a 15x15 TIFF centered at a specific longitude/latitude point, snapping the center point to the nearest
    pixel in the original raster's grid to ensure proper alignment.

    Parameters:
        original_tif_path (str): Path to the original TIFF file.
        output_tif_path (str): Path to save the smaller output TIFF file.
        center_point (tuple): The (longitude, latitude) of the center point.
        binary_array (numpy.ndarray): The 15x15 binary array of values to write to the new TIFF.

    Returns:
        None
    """
    with rasterio.open(original_tif_path) as src:
        # Get the resolution of the original raster
        pixel_width, pixel_height = src.res

        # Extract the center longitude and latitude
        center_lon, center_lat = center_point

        # Snap the center point to the nearest pixel in the original raster's grid
        center_row, center_col = rasterio.transform.rowcol(src.transform, center_lon, center_lat)
        snapped_lon, snapped_lat = rasterio.transform.xy(src.transform, center_row, center_col)

        print(f"Original center point: {center_point}")
        print(f"Snapped center point: ({snapped_lon}, {snapped_lat})")

        # Calculate the bounds of the new 15x15 raster
        half_width = (binary_array.shape[1] / 2) * pixel_width
        half_height = (binary_array.shape[0] / 2) * pixel_height

        left = snapped_lon - half_width
        right = snapped_lon + half_width
        bottom = snapped_lat - half_height
        top = snapped_lat + half_height

        # Create the new affine transform for the smaller raster
        new_transform = from_origin(left, top, pixel_width, pixel_height)

        # Create the new 15x15 raster
        with rasterio.open(
                output_tif_path,
                "w",
                driver="GTiff",
                height=binary_array.shape[0],
                width=binary_array.shape[1],
                count=1,  # Single band
                dtype=np.uint8,  # Assume binary array is uint8
                crs=src.crs,
                transform=new_transform,
        ) as dst:
            # Write the binary array to the new TIFF
            dst.write(binary_array, 1)

        print(f"New 15x15 TIFF saved to {output_tif_path}")
        print(f"Bounding box of new TIFF: {left}, {bottom}, {right}, {top}")
        print(f"Center of the bounding box: {snapped_lon}, {snapped_lat}")


def plot_tif(tif_path, output_path, band=1, cmap="viridis"):
    """
    Plots a single-band or multi-band TIFF file.

    Parameters:
        tif_path (str): Path to the TIFF file.
        band (int): Band number to plot (default is 1).
        cmap (str): Colormap to use for single-band rasters (default is "viridis").

    Returns:
        None
    """
    # Open the TIFF file
    with rasterio.open(tif_path) as src:
        # Get raster metadata
        count = src.count
        bounds = src.bounds
        crs = src.crs
        resolution = src.res

        print(f"Raster Info:")
        print(f"- Number of bands: {count}")
        print(f"- CRS: {crs}")
        print(f"- Resolution: {resolution}")
        print(f"- Bounds: {bounds}\n")

        # Read the selected band
        data = src.read(band)

        # Plot the raster
        plt.figure(figsize=(10, 10))

        if count == 1:  # Single-band raster
            plt.title(f"Band {band} (Single-Band Raster)")
            plt.imshow(data, cmap=cmap, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))
            plt.colorbar(label="Value")

        elif count > 1:  # Multi-band raster
            plt.title(f"Band {band} (Multi-Band Raster)")
            plt.imshow(data, extent=(bounds.left, bounds.right, bounds.bottom, bounds.top))

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.grid(color='white', linestyle='--', linewidth=0.5)

        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


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
    # then divide the number frame-equivalent window pixels by the actual number of pixels in the window to get the upscaling factor

    # --- 1. TRANSLATE: Move so that the frame center is set to (0,0) ---
    to_center = Affine.translation(center_lon, center_lat)
    my_transform = to_center
    print("moving center to (0,0)")
    print(my_transform, "\n")

    # --- 2. SCALE: Adjust pixel size to match target frame ---
    scaling_ratio = (window_size_m / frame_pixel_size_m) / window_size_pixels
    print("scaling_ratio: ", scaling_ratio)
    scaling = Affine.scale(scaling_ratio)

    my_transform = scaling * my_transform
    print("rescaling transform")
    print(my_transform, "\n")

    # --- 3. ROTATE: Apply rotation around the center point of the frame ---
    rotation = Affine.rotation(angle_wrt_north)
    my_transform = rotation * my_transform
    print("base rotation transform")
    print(my_transform, "\n")

    # --- 4. TRANSLATE so that the upper left corner of the frame is the new (0,0) ---
    (new_upper_left_lon, new_upper_left_lat) = my_transform * (new_upper_left_lon, new_upper_left_lat)
    to_upper_left = Affine.translation(-new_upper_left_lon, -new_upper_left_lat)

    my_transform = to_upper_left * my_transform
    print("base to_upper_left transform")
    print(my_transform, "\n")

    reprojected_mask = np.zeros((window.shape[0], frame_height, frame_width), dtype=np.uint8)
    with rasterio.Env():
        reproject(
            source=window,
            destination=reprojected_mask,
            src_transform=window_transform,
            src_crs=window_crs,
            dst_transform=my_transform,
            dst_crs=window_crs,  # Assume the drone uses the same CRS as the binary mask (?)
            resampling=Resampling.nearest
        )

    top_left_coords = my_transform * (0, 0)
    top_right_coords = my_transform * (frame_width, 0)
    bottom_right_coords = my_transform * (frame_width, frame_height)
    bottom_left_coords = my_transform * (0, frame_height)
    print(f"Frame geographic bounds. top-left: {top_left_coords}")
    print(f"Frame geographic bounds. top right: {top_right_coords}")
    print(f"Frame geographic bounds. bottom right: {bottom_right_coords}")
    print(f"Frame geographic bounds. bottom left: {bottom_left_coords}")

    return reprojected_mask


def map_dem_to_drone_frame(
    dem_array,
    dem_transform,
    drone_center,  # (lon, lat) tuple for center
    drone_ul,      # (lon, lat) for upper-left
    drone_ur,      # (lon, lat) for upper-right
    drone_br,      # (lon, lat) for bottom-right (not used below)
    drone_bl,      # (lon, lat) for bottom-left
    yaw_angle,     # in degrees; positive means counterclockwise rotation
    scale_factor,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
    output_shape=(1080, 1920),
    crs='EPSG:4326'
):
    """
    Given:
      - dem_array: a NumPy array holding the DEM window (e.g., at 10 m resolution)
      - dem_transform: its affine transform (mapping DEM-window pixel indices to geospatial coords)
      - drone_center: geospatial coordinate (lon,lat) for the drone frame center
      - drone_ul, drone_ur, drone_br, drone_bl: geospatial coordinates of the drone frame corners
        (ordered as upper-left, upper-right, bottom-right, bottom-left)
      - yaw_angle: rotation (in degrees) of the drone frame relative to north. (Note that
        Affine.rotation rotates counterclockwise about the origin.)
      - scale_factor: a multiplicative scale factor to apply.
      - output_shape: (height, width) of the output drone frame (default 1080×1920)
      - crs: coordinate reference system for all coordinates (source and destination)

    The DEM is “warped” so that it is rotated (around drone_center) and scaled,
    and only the portion corresponding to the drone frame (with its known corners)
    is extracted into an array of size output_shape.
    """
    height, width = output_shape

    # --- STEP 1. Build the output transform (T_out) from the drone frame's corners.
    # We “pin”:
    #   (0,0)   --> drone_ul
    #   (width, 0)   --> drone_ur
    #   (0, height)  --> drone_bl
    #
    # The standard affine (in rasterio) maps (col, row) to (x, y) as:
    #    x = a * col + b * row + c
    #    y = d * col + e * row + f
    #
    # We then set:
    c, f = drone_ul  # upper-left coordinate
    # The x-offset per pixel in the column direction comes from the vector from UL to UR:
    a = (drone_ur[0] - drone_ul[0]) / width
    d = (drone_ur[1] - drone_ul[1]) / width
    # Similarly, the x-offset per pixel in the row direction comes from the vector from UL to BL:
    b = (drone_bl[0] - drone_ul[0]) / height
    e = (drone_bl[1] - drone_ul[1]) / height

    T_out = Affine(a, b, c, d, e, f)
    # Now, T_out maps from drone frame pixel coordinates (col, row) to geospatial coordinates.

    # --- STEP 2. Build the rotation+scaling transform (T_rot) about the drone center.
    # This transform (operating in geospatial coordinates) rotates and scales points about drone_center.
    T_rot = (
        Affine.translation(drone_center[0], drone_center[1])
        * (Affine.rotation(yaw_angle) * Affine.scale(scale_factor, scale_factor))
        * Affine.translation(-drone_center[0], -drone_center[1])
    )
    # When applied to any geospatial coordinate G, T_rot(G) gives the rotated and scaled location.

    # --- STEP 3. Compute the destination (output) transform.
    #
    # We want the output pixel (i,j) (with geospatial coordinate T_out(i,j)) to “sample”
    # the DEM at the coordinate that, after applying T_rot, would land at that location.
    # That is, we want to pull values from the DEM at:
    #
    #     G_src = T_rot⁻¹ ( T_out(i,j) )
    #
    # In rasterio's reproject, the destination transform (dst_transform) maps output
    # pixel coordinates to the source’s geospatial coordinates. Therefore, we define:
    dst_transform = ~T_rot * T_out
    # (~T_rot is the inverse of T_rot)

    # --- STEP 4. Reproject (warp) the DEM window into the output drone frame.
    out_array = np.empty(output_shape, dtype=dem_array.dtype)
    reproject(
        source=dem_array,
        destination=out_array,
        src_transform=dem_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.bilinear
    )

    # DEBUG

    print("true center coordinates")
    print(center_point)
    print("true corners coordinates")
    print(corners_coordinates)

    print("\n")

    print("EVALUATING T_rot")
    sample_center = (output_shape[1] / 2, output_shape[0] / 2)
    sample_ul = (0, 0)
    sample_ur = (output_shape[1]-1, 0)
    sample_br = (output_shape[1]-1, output_shape[0]-1)
    sample_bl = (0, output_shape[0]-1)
    print("Sample center coordinate:", (~T_rot) * T_out * sample_center)
    print("Sample UL coordinate:", (~T_rot) * T_out * sample_ul)
    print("Sample UR coordinate:", (~T_rot) * T_out * sample_ur)
    print("Sample BR coordinate:", (~T_rot) * T_out * sample_br)
    print("Sample BL coordinate:", (~T_rot) * T_out * sample_bl)

    print("\n")

    print("EVALUATING T_out")
    sample_ul = (0, 0)
    sample_ur = (output_shape[1] - 1, 0)
    sample_br = (output_shape[1] - 1, output_shape[0] - 1)
    sample_bl = (0, output_shape[0] - 1)
    print("Sample UL coordinate:", T_out * sample_ul)
    print("Sample UR coordinate:", T_out * sample_ur)
    print("Sample BR coordinate:", T_out * sample_br)
    print("Sample BL coordinate:", T_out * sample_bl)

    print("\n")

    return out_array


if __name__ == "__main__":

    """ PREPROCCESSING """

    # Inputs
    binary_mask_path = 'xxx2_output_plots/a6_centered_15x15.tif'  # Binary mask file (TIFF)
    drone_image_path = 'experiments/test_in_danger_v2/danger_frame_1_annotated.png'  # Drone image file (JPEG)
    output_overlay_path = 'xxx2_output_plots/overlayed_image.jpg'   # Output overlay path

    # Load the drone image
    drone_image = cv2.imread(drone_image_path)
    frame_height, frame_width = drone_image.shape[:2]

    mask_src = rasterio.open(binary_mask_path)

    yaw = -35.4
    frame_pixel_size_m = 0.0199
    center_point = (24.174019, 35.427075)
    corners_coordinates = np.array([
        [24.173774291947453, 35.42705376362871],
        [24.17412397856997, 35.427257168725454],
        [24.174263708181012, 35.42709623587535],
        [24.17391402190279, 35.426892831177796],
    ])

    window, dem_mask_window, window_transform, masks_window_bounds, masks_window_size = extract_dem_window(
        mask_src,
        None,
        center_point,
        corners_coordinates
    )

    print("window info")
    print(window)
    print(masks_window_bounds)
    print(masks_window_size)
    print(window_transform)
    print(window_transform.c)
    print(window_transform.f)

    # find the distance between two points on opposite side of the window at the drone latitude
    window_size_m = get_window_size_m(center_point[1], masks_window_bounds)
    print("\n", window_size_m, "\n")

    scale_factor = (window_size_m / frame_pixel_size_m) / masks_window_size
    print("scale factor: ", scale_factor)

    from time import time
    start = time()
    reprojected_mask = map_dem_to_drone_frame(
        dem_array=window,
        dem_transform=window_transform,
        drone_center=center_point,  # (lon, lat) tuple for center
        drone_ul=tuple(corners_coordinates[0]),      # (lon, lat) for upper-left
        drone_ur=tuple(corners_coordinates[1]),      # (lon, lat) for upper-right
        drone_br=tuple(corners_coordinates[2]),      # (lon, lat) for bottom-right (not used below)
        drone_bl=tuple(corners_coordinates[3]),      # (lon, lat) for bottom-left
        yaw_angle=yaw,     # in degrees; positive means counterclockwise rotation
        scale_factor=(window_size_m / 0.0199) / masks_window_size,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
        output_shape=(1080, 1920),
        crs=mask_src.crs
    )

    print(f"run in {(time()-start)*1000:.2f} ms")

    """
    reprojected_mask = map_window_onto_drone_frame(
        window=window,
        window_transform=window_transform,
        window_crs=mask_src.crs,
        center_coords=center_point,
        corners_coords=corners_coordinates,
        angle_wrt_north=yaw,
        frame_width=frame_width,
        frame_height=frame_height,
        window_size_pixels=masks_window_size,
        window_size_m=window_size_m,
        frame_pixel_size_m=frame_pixel_size_m,
    )
    """

    print("min:", np.min(reprojected_mask))
    print("max:", np.max(reprojected_mask))
    cv2.imwrite(f"xxx2_output_plots/reprojected_mask{yaw}.jpg", (reprojected_mask[0]*255).astype(np.uint8))

    print(f"Overlayed image saved to {output_overlay_path}")

    mask_src.close()


    """

    # Path to the original large TIFF file
    original_tif_path = "data/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif"
    # Center point in (longitude, latitude)
    center_point = (24.174019, 35.427075)

    # fake maps
    a_list = fake_tif()

    for i, a in enumerate(a_list, 1):
        # Path to save the new smaller 15x15 TIFF
        output_tif_path = f"xxx2_output_plots/a{i}_centered_15x15.tif"
        # Create the 15x15 TIFF
        create_centered_tif_with_snapping(original_tif_path, output_tif_path, center_point, a)
        plot_tif(output_tif_path, f"xxx2_output_plots/a{i}_centered_15x15.png")
    """









