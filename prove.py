
def geofencing():

    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Polygon

    test_dem = "data/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif"

    crop = True

    out_path_tif = f"data/DEM/cut_area_{crop}.tif"
    out_path_mask = f"data/DEM/cut_area_mask_{crop}.tif"

    png_path = f"data/DEM/cut_area_{crop}.png"
    png_path_mask = f"data/DEM/cut_area_mask_{crop}.png"

    fill_value = 0

    coords = [
        (23.9385, 35.4769),
        (24.7611, 35.4081),
        (24.0772, 35.2226),
    ]

    coords = [
        (24.210, 35.250),
        (24.220, 35.260),
        (24.250, 35.230),
        (24.240, 35.220),
    ]

    coords = [
        (24.210, 35.250),
        (24.220, 35.260),
        (24.215, 35.230),
        (24.225, 35.220),
    ]
    # coords = [
    #    (21.210, 31.250),
    #    (21.220, 31.260),
    #    (21.215, 31.230),
    #    (21.225, 31.220),
    #]


    # Create a Shapely polygon
    polygon = [Polygon(coords)]

    # Use rasterio.mask.mask to extract the DEM data for the polygon
    # out_image is a NumPy array containing the extracted DEM values.
    # out_transform gives the affine transform for the extracted data.

    with rasterio.open(test_dem) as src:
        print(src.bounds)
        out_array, out_transform = mask(src, polygon, nodata=np.nan, crop=crop)

    print(out_array.shape)

    # dem_data will contain NoData values for areas outside the polygon
    mask = np.isnan(out_array[0]).astype(np.uint8)

    # Replace nodata values in merged_data with 0
    out_array[0][mask == 1] = fill_value

    # Convert mask to 3D array (C,H,W)
    mask = np.expand_dims(mask, axis=0)

    # Get metadata from the first TIF
    out_meta = src.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        'count': 1,
        "height": out_array.shape[1],
        "width": out_array.shape[2],
        "transform": out_transform,
        "nodata": None
    })

    out_meta_mask = src.meta.copy()
    out_meta_mask.update({
        "driver": "GTiff",
        'count': 1,
        "height": out_array.shape[1],
        "width": out_array.shape[2],
        "transform": out_transform,
        "nodata": None,
        'dtype': 'uint8',
    })

    # Write the merged DEM to the output file
    with rasterio.open(out_path_tif, "w", **out_meta) as dest:
        dest.write(out_array)

    # Write the merged DEM mask to the output file
    with rasterio.open(out_path_mask, "w", **out_meta_mask) as dest:
        dest.write(mask)

    print(f"Merged DEM saved to: {out_path_tif}")
    print(f"Merged DEM mask saved to: {out_path_mask}")

    tif_to_png(out_path_tif, png_path)
    tif_to_png(out_path_mask, png_path_mask)

    from scipy.ndimage import rotate

    # Rotate by 45 degrees
    rotated_arbitrary = rotate(out_array[0], angle=45, reshape=True, order=3)
    rotated_arbitrary_mask = rotate(mask[0], angle=45, reshape=True, order=3, cval=1.0)
    plot_2d_array(rotated_arbitrary, f"data/DEM/rotated_{crop}.png")
    plot_2d_array(rotated_arbitrary_mask, f"data/DEM/rotated_mask_{crop}.png")

def rotate():

    import rasterio
    from rasterio.mask import mask
    from shapely.geometry import Polygon
    from scipy.ndimage import rotate

    test_dem = "data/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif"

    crop = True

    out_path_tif = f"data/DEM/cut_area_{crop}.tif"
    out_path_mask = f"data/DEM/cut_area_mask_{crop}.tif"

    png_path = f"data/DEM/cut_area_{crop}.png"
    png_path_mask = f"data/DEM/cut_area_mask_{crop}.png"

    fill_value = 0

    coords = [
        (24.210, 35.250),
        (24.220, 35.260),
        (24.215, 35.230),
        (24.225, 35.220),
    ]

    # Create a Shapely polygon
    polygon = [Polygon(coords)]

    # Use rasterio.mask.mask to extract the DEM data for the polygon
    # out_image is a NumPy array containing the extracted DEM values.
    # out_transform gives the affine transform for the extracted data.

    with rasterio.open(test_dem) as src:
        print(src.bounds)
        out_array, out_transform = mask(src, polygon, nodata=np.nan, crop=crop)

    print(out_array.shape)

    # dem_data will contain NoData values for areas outside the polygon
    mask = np.isnan(out_array[0]).astype(np.uint8)

    # Replace nodata values in merged_data with 0
    out_array[0][mask == 1] = fill_value


    # Rotate by 45 degrees
    rotated_arbitrary = rotate(out_array[0], angle=45, reshape=True, order=3)
    rotated_arbitrary_mask = rotate(mask[0], angle=45, reshape=True, order=3, cval=1.0)
    plot_2d_array(rotated_arbitrary, f"data/DEM/rotated_{crop}.png")
    plot_2d_array(rotated_arbitrary_mask, f"data/DEM/rotated_mask_{crop}.png")

import numpy as np
from shapely.geometry import Polygon
from shapely.vectorized import contains
from time import time

def geofencing_mask(min_long, min_lat, max_long, max_lat, polygon, resolution_x=100, resolution_y=100):
    """
    Create a geofencing mask for a polygon within given geographic bounds.

    Parameters:
    - min_long: Minimum longitude.
    - min_lat: Minimum latitude.
    - max_long: Maximum longitude.
    - max_lat: Maximum latitude.
    - polygon: Shapely Polygon object.
    - resolution: Number of grid points along each axis (default: 100).

    Returns:
    - mask: 2D boolean numpy array of shape (resolution, resolution).
    - longitudes: 2D array of longitude values.
    - latitudes: 2D array of latitude values.
    """

    # Create a grid of longitude and latitude values
    longitudes = np.linspace(min_long, max_long, resolution_x)
    latitudes = np.linspace(max_lat, min_lat, resolution_y)
    long_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    print(latitudes)
    print(longitudes)

    # Create the mask using shapely's vectorized contains
    mask = shapely.vectorized.contains(polygon, long_grid, lat_grid)
    print(mask)

    if mask.ndim != 2:
        raise ValueError("The input array must be 2-dimensional.")

    plt.figure(figsize=(8, 6))
    plt.imshow(mask, extent=[np.min(longitudes), np.max(longitudes), np.min(latitudes), np.max(latitudes)],
               origin='upper', cmap='viridis', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Geofencing Mask')
    plt.colorbar(label='Inside Polygon (True/False)')
    plt.savefig("./mask_test.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    for i, row in enumerate(mask):
        true_count = np.sum(row)
        print(f"Row {i}: {true_count} elements equal to True")

    for j in range(mask.shape[1]):
        true_count = np.sum(mask[:, j])
        print(f"Column {j}: {true_count} elements equal to True")

    return mask, long_grid, lat_grid



import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import from_bounds
import os
from shapely import Polygon
import numpy as np
from rasterio.mask import mask as rasterio_mask


def extract_and_plot_area(tif_file, lat_min, lat_max, lon_min, lon_max, polygon, output_dir):
    """
    Extracts an area from a GeoTIFF file based on latitude and longitude bounds,
    and saves a plot of each channel in the extracted area.

    Parameters:
        tif_file (str): Path to the input GeoTIFF file.
        lat_min (float): Minimum latitude of the area to extract.
        lat_max (float): Maximum latitude of the area to extract.
        lon_min (float): Minimum longitude of the area to extract.
        lon_max (float): Maximum longitude of the area to extract.
        output_dir (str): Directory to save the output plots.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the GeoTIFF file
    src = rasterio.open(tif_file)
    # Create a window for the area to extract
    window = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
    # Get the updated transform for the window
    window_transform = src.window_transform(window)
    # Read the data within the window
    data = src.read(window=window)
    print(data.shape)

    with rasterio.open(
            "xxx_output_plots/reduced_tif.tif",
            'w',
            driver='GTiff',
            height=data.shape[1],
            width=data.shape[2],
            count=src.count,
            dtype=data.dtype,
            crs=src.crs,
            transform=window_transform
    ) as dst:
        dst.write(data)

    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    src.close()

    src = rasterio.open("xxx_output_plots/reduced_tif.tif", "r")
    data = src.read()
    print(data.shape)

    masked, _ = rasterio_mask(src, shapes=[polygon], all_touched=False, nodata=2, crop=False)
    print(masked.shape)
    src.close()

    # plotting ===============================================
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    mpl_polygon = MplPolygon(
        list(polygon.exterior.coords),
        closed=True,
        edgecolor='red',
        fill=False,
        linewidth=2
    )

    for i, channel in enumerate(data, start=1):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(channel, cmap='gray', extent=extent)  # Display the channel with its spatial extent
        ax.add_patch(mpl_polygon)  # Add the polygon to the plot

        # Set plot titles and labels
        ax.set_title(f"Shapely Polygon on TIFF Channel {i}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Save the plot to a file (e.g., PNG format)
        output_filename = f"tiff_channel_{i}_with_polygon.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Plot for channel {i} saved as {output_filename}")

        # Close the figure to avoid displaying it
        plt.close(fig)

    for i, channel in enumerate(masked, start=1):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(channel, cmap='gray', extent=extent)  # Display the channel with its spatial extent
        ax.add_patch(mpl_polygon)  # Add the polygon to the plot

        # Set plot titles and labels
        ax.set_title(f"Shapely Polygon on TIFF Channel {i}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Save the plot to a file (e.g., PNG format)
        output_filename = f"tiff_channel_masked_{i}_with_polygon.png"
        plt.savefig(output_filename, dpi=300)
        print(f"Plot for channel {i} saved as {output_filename}")

        # Close the figure to avoid displaying it
        plt.close(fig)


if __name__ == "__main__":
    # Example usage
    tif_file = "experiments/test_in_danger_v2/combined_dem_masks.tif"  # Path to your GeoTIFF file
    output_dir = "xxx_output_plots"

    # lat_min = 35.426892831177796
    # lat_max = 35.427257168725454
    # lon_min = 24.173774291947453
    # lon_max = 24.174263708181012

    lat_min = 35.426635
    lat_max = 35.427601
    lon_min = 24.172913
    lon_max = 24.174834

    lat_min = 35.426
    lat_max = 35.428
    lon_min = 24.173
    lon_max = 24.175


    frame_polygon = Polygon(np.array([
        [24.173774291947453, 35.42705376362871],
        [24.17412397856997, 35.427257168725454],
        [24.174263708181012, 35.42709623587535],
        [24.17391402190279, 35.426892831177796],

    ]))

    frame_polygon = Polygon(np.array([
        [24.1738, 35.4271],
        [24.1741, 35.4276],
        [24.1743, 35.4271],
        [24.1739, 35.4269],
    ]))

    extract_and_plot_area(tif_file, lat_min, lat_max, lon_min, lon_max, frame_polygon, output_dir)

