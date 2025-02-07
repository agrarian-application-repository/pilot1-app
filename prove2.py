import numpy as np

"""
center_x = 5
center_y = 7
pixel_coords = np.array([
    [6, 4],
    [8, 6],
    [2, 7],
    [4, 10],
])
"""

center_y = 3
center_x = 4
pixel_coords = np.array([
    [1, 2],
    [1, 6],
    [5, 6],
    [5, 2],
])

pixel_dists = np.linalg.norm(pixel_coords - np.array([center_y, center_x]), axis=1)
print(pixel_dists)
max_dist = int(np.max(pixel_dists))  # Maximum pixel distance

print(max_dist)

# --- Step 3: Compute square window size (odd number) ---
buffer = int(np.ceil(max_dist * 0.5))  # Extra space for rotation

half_size = max_dist + buffer
window_size = 2 * half_size + 1  # Ensure window is odd

print(window_size)

# --- Step 4: Define the window in pixel coordinates ---
window_row_start = center_y - half_size
window_row_end = center_y + half_size + 1
window_col_start = center_x - half_size
window_col_end = center_x + half_size + 1

print(window_row_start)
print(window_row_end)
print(window_col_start)
print(window_col_end)