from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling
import numpy as np
import math
import rasterio
import matplotlib.pyplot as plt


def rotate_tif_around_point(input_tif, output_tif, center_lat, center_lon, rotation_angle_deg):
    """
    Rotates a GeoTIFF around a specified latitude/longitude point and saves the result.

    Parameters:
        input_tif (str): Path to the input GeoTIFF file.
        output_tif (str): Path to save the rotated GeoTIFF.
        center_lat (float): Latitude of the rotation center.
        center_lon (float): Longitude of the rotation center.
        rotation_angle_deg (float): Rotation angle in degrees (clockwise).

    Returns:
        None
    """
    # Open the original TIFF
    with rasterio.open(input_tif) as src:
        data = src.read(1)  # Read single-band data (update if multi-band is needed)
        mask_crs = src.crs
        original_transform = src.transform
        height, width = src.shape

        # Compute pixel size
        pixel_width = original_transform.a  # Scale in x (longitude)
        pixel_height = -original_transform.e  # Scale in y (latitude)

        # Step 1: Translate center to origin (center_lon, center_lat → (0,0))
        # to_origin = Affine.translation(-center_lon, -center_lat)

        # Step 2: Rotate around the origin
        rotation = Affine.rotation(
            angle=rotation_angle_deg,
            pivot=(center_lon, center_lat)
        )

        # Step 3: Translate back to original center
        from_origin = Affine.translation(center_lon, center_lat)

        # Step 4: Apply scaling to maintain pixel size
        scaling = Affine.scale(pixel_width, -pixel_height)

        # Combine all transforms
        # rotated_transform = from_origin * scaling * rotation * to_origin
        # rotated_transform = from_origin * rotation * to_origin

        rotated_transform = original_transform * rotation
        rotated_transform = rotation * original_transform

        # Create an empty array for the rotated image
        rotated_data = np.zeros((3*height, 3*width), dtype=data.dtype)

        # Reproject the data using the new transform
        with rasterio.Env():
            reproject(
                source=data,
                destination=rotated_data,
                src_transform=original_transform,
                src_crs=mask_crs,
                dst_transform=rotated_transform,
                dst_crs=mask_crs,
                resampling=Resampling.nearest  # Change to cubic/sinc if smooth interpolation is needed
            )

        # Save the rotated TIFF
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,  # Single-band output
            dtype=data.dtype,
            crs=mask_crs,
            transform=rotated_transform
        ) as dst:
            dst.write(rotated_data, 1)

    print(f"Rotated GeoTIFF saved to {output_tif}")


def save_tif_plot_with_center(tif_path, output_image_path, title, center_lat, center_lon):
    """
    Saves a plot of a GeoTIFF file as an image (PNG, JPG, etc.), with a red dot marking the center point.

    Parameters:
        tif_path (str): Path to the TIFF file to be plotted.
        output_image_path (str): Path to save the plotted image.
        center_latlon (tuple): (longitude, latitude) of the center point to be marked.
        title (str): Title of the plot.

    Returns:
        None
    """
    with rasterio.open(tif_path) as src:
        data = src.read(1)  # Read the first band (assumes single-band TIFF)
        transform = src.transform  # Get the georeferencing transform

        # Get geographic extent
        extent = [
            transform.c,  # Left (min longitude)
            transform.c + transform.a * src.width,  # Right (max longitude)
            transform.f + transform.e * src.height,  # Bottom (min latitude)
            transform.f  # Top (max latitude)
        ]

        # Plot the image
        plt.figure(figsize=(8, 6))
        plt.imshow(data, cmap="viridis", extent=extent, origin="upper")
        plt.colorbar(label="Pixel Value")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(title)
        plt.grid(visible=True, linestyle="--", alpha=0.5)

        # Plot the center point in red
        plt.scatter(center_lon, center_lat, color="red", marker="o", s=50, label="Center Point")
        plt.legend()

        # Save the figure
        plt.savefig(output_image_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory

    print(f"Plot saved to {output_image_path}")


if __name__ == "__main__":

    angle = 45
    center_lat = 35.427075
    center_lon = 24.174019

    rotate_tif_around_point(
        input_tif="data/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif",
        output_tif=f"xxx2/output_rotated{angle}.tif",
        center_lat=center_lat,
        center_lon=center_lon,
        rotation_angle_deg=angle  # Rotate 30 degrees clockwise
    )

    save_tif_plot_with_center(
        f"xxx2/output_rotated{angle}.tif",
        output_image_path=f"xxx2/output_rotated{angle}.png",
        title="Rotated GeoTIFF",
        center_lat=center_lat,
        center_lon=center_lon,
    )
