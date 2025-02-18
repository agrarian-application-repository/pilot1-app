from rasterio.transform import from_origin
import cv2
from src.in_danger.in_danger_utils import get_window_size_m, extract_dem_window
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


def simplified_map_dem_to_drone_frame(
        dem_array,
        dem_transform,
        drone_ul,  # (lon, lat) for upper-left
        drone_ur,  # (lon, lat) for upper-right
        drone_bl,  # (lon, lat) for bottom-left
        output_shape=(1080, 1920),
        crs='EPSG:4326'
):
    """
    Map the DEM window to the drone frame using only the output transform T_out,
    which is built from the provided drone frame corner coordinates.

    Note: This does not apply any additional rotation/scale correction via T_rot.
    """
    height, width = output_shape

    # Build T_out using the known drone frame corners.
    # Here, we assume:
    #  (0, 0)             --> drone_ul
    #  (width-1, 0)        --> drone_ur
    #  (0, height-1)       --> drone_bl
    #
    # The affine transform T_out is defined as:
    #     x = a * col + b * row + c
    #     y = d * col + e * row + f
    c, f = drone_ul
    a = (drone_ur[0] - drone_ul[0]) / (width - 1)
    d = (drone_ur[1] - drone_ul[1]) / (width - 1)
    b = (drone_bl[0] - drone_ul[0]) / (height - 1)
    e = (drone_bl[1] - drone_ul[1]) / (height - 1)

    T_out = Affine(a, b, c, d, e, f)

    # For debugging: print T_out mappings
    sample_center = (width / 2, height / 2)
    sample_ul = (0, 0)
    sample_ur = (width - 1, 0)
    sample_br = (width - 1, height - 1)
    sample_bl = (0, height - 1)
    print("Sample center coordinate:", T_out * sample_center)
    print("Sample UL coordinate:", T_out * sample_ul)
    print("Sample UR coordinate:", T_out * sample_ur)
    print("Sample BR coordinate:", T_out * sample_br)
    print("Sample BL coordinate:", T_out * sample_bl)

    # Reproject DEM into the output frame using T_out directly.
    out_array = np.empty(output_shape, dtype=dem_array.dtype)
    reproject(
        source=dem_array,
        destination=out_array,
        src_transform=dem_transform,
        src_crs=crs,
        dst_transform=T_out,
        dst_crs=crs,
        resampling=Resampling.nearest
    )

    return out_array


def simplified_map_dem_to_drone_frame_fast(
        dem_array,
        dem_transform,
        drone_ul,  # (lon, lat) for upper-left
        drone_ur,  # (lon, lat) for upper-right
        drone_bl,  # (lon, lat) for bottom-left
        output_shape=(1080, 1920),
        crs='EPSG:4326',
        debug=False
):
    """
    Optimized version of simplified DEM to drone frame mapping.
    """
    height, width = output_shape

    # Pre-compute width and height differences (avoid repeated subtraction)
    width_minus_1 = width - 1
    height_minus_1 = height - 1

    # Pre-compute coordinate differences (avoid repeated subtraction)
    dx_ur = drone_ur[0] - drone_ul[0]
    dy_ur = drone_ur[1] - drone_ul[1]
    dx_bl = drone_bl[0] - drone_ul[0]
    dy_bl = drone_bl[1] - drone_ul[1]

    # Calculate transform coefficients (single division operations)
    a = dx_ur / width_minus_1
    d = dy_ur / width_minus_1
    b = dx_bl / height_minus_1
    e = dy_bl / height_minus_1

    # Create transform (stored in memory for reuse)
    T_out = Affine(a, b, drone_ul[0], d, e, drone_ul[1])

    # Debug output (only if needed)
    if debug:
        sample_points = {
            "Center": (width / 2, height / 2),
            "UL": (0, 0),
            "UR": (width_minus_1, 0),
            "BR": (width_minus_1, height_minus_1),
            "BL": (0, height_minus_1)
        }
        for name, point in sample_points.items():
            print(f"{name} coordinate: {T_out * point}")

    # Pre-allocate output array with correct data type
    out_array = np.empty(output_shape, dtype=dem_array.dtype)

    # Perform reprojection with optimized parameters
    reproject(
        source=dem_array,
        destination=out_array,
        src_transform=dem_transform,
        src_crs=crs,
        dst_transform=T_out,
        dst_crs=crs,
        resampling=Resampling.nearest,
        init_dest_nodata=False,  # Skip nodata initialization
        warp_mem_limit=256  # Reduced memory limit but still efficient
    )

    return out_array


def map_dem_to_drone_frame_v2(
        dem_array,
        dem_transform,
        drone_center,   # (lon, lat) for center of the drone frame (around which to rotate)
        drone_ul,  # (lon, lat) of the final upper-left corner of drone frame (post rotation and scaling)
        yaw_angle,  # in degrees; positive means counterclockwise rotation
        scale_factor,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
        output_shape=(1080, 1920),
        crs='EPSG:4326'
):
    """
    This version builds a transform that rotates and scales the DEM using the drone
    frame's upper-left corner as the pivot (i.e. that point remains fixed).

    The destination transform is defined solely by T_rot, whose inverse maps
    output pixel coordinates to the DEM's geographic coordinates.

    Parameters:
      dem_array: numpy array of the DEM window.
      dem_transform: affine transform for the DEM window.
      drone_center: geographic coordinate (lon, lat) of the drone frame's center, around which the tif should be rotated.
      drone_ul: geographic coordinate (lon, lat) of the drone frame's upper-left corner -> to be mapped onto new array (0,0).
      yaw_angle: rotation (in degrees) of the drone frame.
      scale_factor: multiplicative scaling factor.
      output_shape: (height, width) of output drone frame.
      crs: coordinate reference system.

    Returns:
      Reprojected DEM portion as an array with shape output_shape.
    """
    height, width = output_shape
    print(scale_factor)

    # Build a transform that rotates and scales about the UL.
    T_rot = (
            Affine.translation(drone_center[0], drone_center[1])  # Move to center
            * Affine.rotation(yaw_angle)  # Rotate
            * Affine.scale(scale_factor, scale_factor)  # Scale
            * Affine.translation(-drone_center[0], -drone_center[1])  # Move back
            * Affine.translation(drone_ul[0] - drone_center[0], drone_ul[1] - drone_center[1])  # Shift UL to match
    )

    dst_transform = ~T_rot  # inverse of T_rot

    # Optionally, you can print out what geographic coordinates various output pixels map to:
    # For debugging: print T_out mappings
    sample_center = (width / 2, height / 2)
    sample_ul = (0, 0)
    sample_ur = (width - 1, 0)
    sample_br = (width - 1, height - 1)
    sample_bl = (0, height - 1)
    print("Sample center coordinate:", dst_transform * sample_center)
    print("Sample UL coordinate:", dst_transform * sample_ul)
    print("Sample UR coordinate:", dst_transform * sample_ur)
    print("Sample BR coordinate:", dst_transform * sample_br)
    print("Sample BL coordinate:", dst_transform * sample_bl)

    # Reproject the DEM into the output grid using dst_transform.
    out_array = np.empty(output_shape, dtype=dem_array.dtype)
    reproject(
        source=dem_array,
        destination=out_array,
        src_transform=dem_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.nearest
    )

    return out_array


def map_dem_to_drone_frame_v3(
        dem_array,
        dem_transform,
        drone_center,  # (lon, lat) for center of the drone frame (around which to rotate)
        drone_ul,  # (lon, lat) of the final upper-left corner of drone frame (post rotation and scaling)
        yaw_angle,  # in degrees; positive means counterclockwise rotation
        scale_factor,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
        output_shape=(1080, 1920),
        crs='EPSG:4326'
):
    """
    Maps DEM data onto a drone frame using center coordinates, rotation, and scaling.

    Parameters:
        dem_array: numpy array of the DEM window.
        dem_transform: affine transform for the DEM window.
        drone_center: geographic coordinate (lon, lat) of the drone frame's center.
        drone_ul: geographic coordinate (lon, lat) of the drone frame's upper-left corner.
        yaw_angle: rotation (in degrees) of the drone frame (positive is counterclockwise).
        scale_factor: multiplicative scaling factor (>1 enlarges, <1 shrinks).
        output_shape: (height, width) of output drone frame.
        crs: coordinate reference system.

    Returns:
        numpy array: Reprojected DEM portion matching the drone frame dimensions.
    """
    height, width = output_shape

    # Calculate pixel size based on scale factor and original DEM resolution
    original_pixel_size = dem_transform.a  # Assuming square pixels
    target_pixel_size = original_pixel_size / scale_factor

    # Convert yaw angle to radians
    angle_rad = np.radians(yaw_angle)

    # Build the destination transform in steps:
    # 1. Start with a basic transform that maps pixel coordinates to geographic coordinates
    base_transform = Affine.translation(drone_ul[0], drone_ul[1]) * \
                     Affine.scale(target_pixel_size, -target_pixel_size)

    # 2. Calculate the offset to the rotation center (drone_center)
    offset_x = drone_center[0] - drone_ul[0]
    offset_y = drone_ul[1] - drone_center[1]  # Note: y is inverted in geographic coordinates

    # 3. Build the complete transform
    dst_transform = (
        # Move to drone_ul
            Affine.translation(drone_ul[0], drone_ul[1]) *
            # Move to rotation center
            Affine.translation(offset_x, offset_y) *
            # Apply rotation
            Affine.rotation(yaw_angle) *
            # Move back from rotation center
            Affine.translation(-offset_x, -offset_y) *
            # Apply scaling
            Affine.scale(target_pixel_size, -target_pixel_size)
    )

    # Initialize output array
    out_array = np.empty(output_shape, dtype=dem_array.dtype)

    # For debugging: print transformed coordinates
    if True:  # Can be turned off in production
        sample_points = {
            "Center": (width / 2, height / 2),
            "Upper Left": (0, 0),
            "Upper Right": (width - 1, 0),
            "Bottom Right": (width - 1, height - 1),
            "Bottom Left": (0, height - 1)
        }
        print("\nDebug: Transformed Coordinates:")
        for name, point in sample_points.items():
            transformed_point = dst_transform * point
            print(f"{name}: {transformed_point}")

    # Perform the reprojection
    reproject(
        source=dem_array,
        destination=out_array,
        src_transform=dem_transform,
        src_crs=crs,
        dst_transform=dst_transform,
        dst_crs=crs,
        resampling=Resampling.nearest
    )

    return out_array


if __name__ == "__main__":

    """ PREPROCCESSING """
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    # Inputs
    binary_mask_path = 'xxx2_output_plots/a2_centered_15x15.tif'  # Binary mask file (TIFF)
    drone_image_path = 'experiments/test_in_danger/danger_frame_1_annotated.png'  # Drone image file (JPEG)

    # Load the drone image
    drone_image = cv2.imread(drone_image_path)
    frame_height, frame_width = drone_image.shape[:2]

    yaw = -35.4
    frame_pixel_size_m = 0.0199
    center_point = (24.174019, 35.427075)
    corners_coordinates = np.array([
        [24.173774291947453, 35.42705376362871],
        [24.17412397856997, 35.427257168725454],
        [24.174263708181012, 35.42709623587535],
        [24.17391402190279, 35.426892831177796],
    ])

    mask_src = rasterio.open(binary_mask_path)

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

    print("\nTARGETS:")
    print("center_point:")
    print(center_point)
    print("corners:")
    print(corners_coordinates)
    print("\n")

    # find the distance between two points on opposite side of the window at the drone latitude
    window_size_m = get_window_size_m(center_point[1], masks_window_bounds)
    print("\n", window_size_m, "\n")

    scale_factor = (window_size_m / frame_pixel_size_m) / masks_window_size
    print("scale factor: ", scale_factor)

    from time import time

    for _ in range(2):
        start = time()
        reprojected_mask = simplified_map_dem_to_drone_frame(
            dem_array=window,
            dem_transform=window_transform,
            drone_ul=tuple(corners_coordinates[0]),      # (lon, lat) for upper-left
            drone_ur=tuple(corners_coordinates[1]),      # (lon, lat) for upper-right
            drone_bl=tuple(corners_coordinates[3]),      # (lon, lat) for bottom-left
            output_shape=(1080, 1920),
            crs=mask_src.crs
        )
        print(f"run in {(time()-start)*1000:.2f} ms")
        print("min:", np.min(reprojected_mask))
        print("max:", np.max(reprojected_mask))
        print(reprojected_mask.shape)
        cv2.imwrite(f"xxx2_output_plots/simple_reprojected_mask.jpg", (reprojected_mask*255).astype(np.uint8))

        print(">>>>>>>>>>")

        start = time()
        reprojected_mask = simplified_map_dem_to_drone_frame(
            dem_array=window,
            dem_transform=window_transform,
            drone_ul=tuple(corners_coordinates[0]),      # (lon, lat) for upper-left
            drone_ur=tuple(corners_coordinates[1]),      # (lon, lat) for upper-right
            drone_bl=tuple(corners_coordinates[3]),      # (lon, lat) for bottom-left
            output_shape=(1080, 1920),
            crs=mask_src.crs
        )
        print(f"run in {(time()-start)*1000:.2f} ms")
        print("min:", np.min(reprojected_mask))
        print("max:", np.max(reprojected_mask))
        print(reprojected_mask.shape)
        cv2.imwrite(f"xxx2_output_plots/fast_simple_reprojected_mask.jpg", (reprojected_mask*255).astype(np.uint8))

        print(">>>>>>>>>>")

        start = time()
        reprojected_mask = map_dem_to_drone_frame_v2(
            dem_array=window,
            dem_transform=window_transform,
            drone_center=center_point,
            drone_ul=tuple(corners_coordinates[0]),  # (lon, lat) for upper-left
            yaw_angle=yaw,  # in degrees; positive means counterclockwise rotation
            scale_factor=scale_factor,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
            output_shape=(1080, 1920),
            crs=mask_src.crs
        )
        print(f"run in {(time() - start) * 1000:.2f} ms")
        print("min:", np.min(reprojected_mask))
        print("max:", np.max(reprojected_mask))
        print(reprojected_mask.shape)
        cv2.imwrite(f"xxx2_output_plots/map_dem_to_drone_frame_v2.jpg", (reprojected_mask * 255).astype(np.uint8))

        print(">>>>>>>>>>")

        start = time()
        reprojected_mask = map_dem_to_drone_frame_v3(
            dem_array=window,
            dem_transform=window_transform,
            drone_center=center_point,
            drone_ul=tuple(corners_coordinates[0]),  # (lon, lat) for upper-left
            yaw_angle=yaw,  # in degrees; positive means counterclockwise rotation
            scale_factor=scale_factor,  # e.g., 1.0 means no scaling; >1 enlarges, <1 shrinks
            output_shape=(1080, 1920),
            crs=mask_src.crs
        )
        print(f"run in {(time() - start) * 1000:.2f} ms")
        print("min:", np.min(reprojected_mask))
        print("max:", np.max(reprojected_mask))
        print(reprojected_mask.shape)
        cv2.imwrite(f"xxx2_output_plots/map_dem_to_drone_frame_v3.jpg", (reprojected_mask * 255).astype(np.uint8))

        print(">>>>>>>>>>")

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









