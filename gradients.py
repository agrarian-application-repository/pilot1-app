import numpy as np
import rasterio
import matplotlib.pyplot as plt
from src.in_danger.in_danger_utils import compute_slope_mask_runtime, compute_slope_mask_horn


def load_dem_and_plot_slope_mask(dem_path, slope_threshold_deg):
    # Load DEM using rasterio
    with rasterio.open(dem_path) as dataset:
        dem = dataset.read(1)  # Read the first band (assumes a single-band DEM)
        pixel_size = 10

    dem = dem[np.newaxis, :, :]

    # Compute the slope mask standard
    slope_mask_base = compute_slope_mask_runtime(dem, pixel_size, slope_threshold_deg)[0]

    # Compute the slope mask standard
    slope_mask_horn = compute_slope_mask_horn(dem, pixel_size, slope_threshold_deg)[0]

    # Plot the slope mask
    plt.figure(figsize=(10, 6))
    plt.imshow(slope_mask_base, cmap='gray', interpolation='nearest')
    plt.title(f"Base Slope Mask (Threshold: {slope_threshold_deg}°)")
    plt.colorbar(label="Mask Value")
    plt.savefig(f"scripts_and_testing/xxx2_output_plots/base_gradients_{slope_threshold_deg}.png", dpi=300)
    plt.close()

    # Plot the slope mask
    plt.figure(figsize=(10, 6))
    plt.imshow(slope_mask_horn, cmap='gray', interpolation='nearest')
    plt.title(f"Horn Slope Mask (Threshold: {slope_threshold_deg}°)")
    plt.colorbar(label="Mask Value")
    plt.savefig(f"scripts_and_testing/xxx2_output_plots/horn_gradients_{slope_threshold_deg}.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    load_dem_and_plot_slope_mask("data/DEM/Copernicus_DSM_04_N35_00_E024_00_DEM.tif", 35)
