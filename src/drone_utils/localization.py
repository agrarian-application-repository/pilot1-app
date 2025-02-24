import geopy
import numpy as np


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

    # get the (x,y) position of the center of the frame (UL corner is 0,0)
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
