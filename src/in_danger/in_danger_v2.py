from pathlib import Path
from time import time
from typing import Any
import rasterio

import cv2
import numpy as np
from ultralytics import YOLO
from scipy.ndimage import rotate as rotate_array
from concurrent.futures import ThreadPoolExecutor


from src.in_danger.in_danger_v2_utils import *


def perform_in_danger_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        detection_args: dict[str:Any],
        segmentation_args: dict[str:Any],
) -> None:

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============== LOAD AND PREPROCESS DEM ===================================
    plot_dem_preprocessing = False

    crono_start = time()

    # Open the DEM
    dem_tif, dem_np = get_dem(
        dem_path=input_args["dem"],
        output_dir=output_dir,
        plot=plot_dem_preprocessing
    )

    # Open or create the DEM mask
    dem_mask_tif, nodata_dem_mask = get_dem_mask(
        dem_mask_path=input_args["dem_mask"],
        fallback_mask_shape=(dem_tif.height, dem_tif.width),
        output_dir=output_dir,
        plot=plot_dem_preprocessing,
    )

    # Create a mask highlighting the areas outside the geofenced boundaries
    geofencing_mask = get_geofencing_mask(
        dem_tif=dem_tif,
        geofencing_vertexes=input_args["geofencing_vertexes"],
        fallback_mask_shape=(dem_tif.height, dem_tif.width),
        output_dir=output_dir,
        plot=plot_dem_preprocessing,
    )

    # if the combination of invalid dem pixels and geofencing leaves no valid pixels, terminate
    aggregated_dem_masks = np.stack([nodata_dem_mask, geofencing_mask])
    terminate_if_no_valid_pixels(merge_3d_mask(aggregated_dem_masks))

    # Create a mask highlighting the areas where the slope angle is above a given threshold
    dangerous_slope_mask, dem_pixel_size_y_m, dem_pixel_size_x_m = create_dangerous_slope_mask(
        tif=dem_tif,
        array=dem_np,
        angle_threshold=input_args["slope_angle_threshold"],
        output_dir=output_dir,
        plot=plot_dem_preprocessing,
    )

    # if the combination of the valid geofenced area and slope leaves no valid pixels, terminate
    aggregated_dem_masks = np.stack([nodata_dem_mask, geofencing_mask, dangerous_slope_mask])
    terminate_if_no_valid_pixels(merge_3d_mask(aggregated_dem_masks))

    if plot_dem_preprocessing:
        plot_2d_array(merge_3d_mask(aggregated_dem_masks), output_dir/"combined_mask.png", title="combined DEM mask")
        plot_histogram(merge_3d_mask(aggregated_dem_masks), output_dir/"hist_combined_mask.png", title="combined DEM histogram")

    # Save the 3-channel mask to a GeoTIFF
    combined_dem_masks_tif_path = output_dir/"combined_dem_masks.tif"
    with rasterio.open(
            combined_dem_masks_tif_path,
            "w",
            driver="GTiff",
            height=dem_tif.height,
            width=dem_tif.width,
            count=3,  # Number of channels
            dtype=aggregated_dem_masks.dtype,
            crs=dem_tif.crs,
            transform=dem_tif.transform,
    ) as dst:
        dst.write(nodata_dem_mask, 1)  # Write DEM nodata mask to band 1
        dst.write(geofencing_mask, 2)  # Write DEM geofencing mask to band 2
        dst.write(dangerous_slope_mask, 3)  # Write DEM slope mask to band 3

    # Close previously opened tifs
    dem_tif.close()
    if dem_mask_tif is not None:
        dem_mask_tif.close()

    # Reopen the mask tif
    combined_dem_masks_tif = rasterio.open(combined_dem_masks_tif_path)

    print(f"DEM data preprocessing took {time()-crono_start:.1f} seconds")

    # ============== LOAD AI MODELS ===================================

    # Load AI models
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")

    # Load YOLO models
    detector = YOLO(detection_model_checkpoint, task="detect")  # Animal detection model
    segmenter = YOLO(segmentation_model_checkpoint, task="segment")  # Dangerous terrain segmentation model

    # ============== LOAD FLIGHT INFO ===================================

    # Open drone flight data
    flight_data_file_path = Path(input_args["flight_data"])
    flight_data_file = open(flight_data_file_path, "r")

    # ============== LOAD VIDEO INFO ===================================

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # avoids unreasonable video strides
    input_args["vid_stride"] = max(1, min(input_args["vid_stride"], total_frames))

    # set the row/col identifier of the frame corners (x,y)
    frame_corners = np.array([
        [0                  , 0               ],    # upper left (0, 0)
        [frame_width - 1    , 0               ],    # upper right (C-1, 0)
        [frame_height - 1   , frame_width - 1 ],    # lower right (R-1, C-1)
        [0                  , frame_height - 1],    # lower left ( 0, R-1)
    ])

    # ============== LOAD ALERTS FILE ===================================

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    # ============== LOAD VIDEO WRITERS ===================================

    annotated_writer = None

    if output_args["save_videos"]:
        annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
        annotated_writer = cv2.VideoWriter(
            filename=annotated_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

    # ============== BEGIN VIDEO PROCESSING ===================================

    # Frame counter
    frame_id = 0
    processed_frames_counter = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = output_args["alerts_cooldown_seconds"] * fps   # convert cooldown from seconds to frames
    last_alert_frame_id = - fps  # to avoid dealing with initial None value, at frame 0 alert is allowed

    # Time keeper
    processing_start_time = time()

    # Video processing loop
    while cap.isOpened():
        iteration_start_time = time()
        success, frame = cap.read()
        if not success or frame_id > 2:
            print("Video processing has been successfully completed.")
            break

        if frame_id % input_args["vid_stride"] != 0:
            frame_id += 1  # Update frame ID
            continue  # go to next frame directly (processes 1 frame every 'vid_stride' frames)

        processed_frames_counter += 1  # update the actual number of frames processed
        frame_id += 1  # Update frame ID
        print(f"\n------------- Processing frame {frame_id}/{total_frames}-----------")

        # load frame flight data
        flight_frame_data = parse_drone_frame(flight_data_file, frame_id)

        """ Perform detection and get bounding boxes (13 ms)"""

        crono_start = time()
        # Detect animals in frame
        classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
        print(f"detection of animals completed in {(time() - crono_start)*1000:.1f} ms")

        """ Perform segmentation and build segmentation danger mask (15 ms)"""

        crono_start = time()
        # Highlight dangerous objects
        segment_danger_mask = perform_segmentation(segmenter, frame, segmentation_args)
        print(f"Segmentation and danger mask creation completed in {(time() - crono_start)*1000:.1f} ms")

        """ extract coordinates of frame (and animals) from drone position and flight height (0.2 ms)"""

        crono_start = time()

        # Perform the pixels to meters conversion using the sensor resolution
        # TODO ASSERT sensor_width_mm/sensor_height_mm == sensor_width_pixels/sensor_height_pixels
        meters_per_pixel = get_meters_per_pixel(
            rel_altitude_m=flight_frame_data["rel_alt"],
            focal_length_mm=input_args["true_focal_len_mm"],
            sensor_width_mm=input_args["drone_sensor_width_mm"],
            sensor_height_mm=input_args["drone_sensor_height_mm"],
            sensor_width_pixels=input_args["drone_sensor_width_pixels"],
            sensor_height_pixels=input_args["drone_sensor_height_pixels"],
            image_width_pixels=frame_width,
            image_height_pixels=frame_height,
        )

        frame_width_m = frame_width * meters_per_pixel
        frame_height_m = frame_height * meters_per_pixel
        print(f"Frame dimensions: {frame_width_m:.2f}x{frame_height_m:.2f} meters")

        safety_radius_pixels = int(input_args["safety_radius_m"] / meters_per_pixel)

        # get the coordinates of the 4 corners of the frame.
        # The rectangle may be oriented in any direction wrt North
        corners_coordinates = get_objects_coordinates(
            objects_coords=frame_corners,   # (X,Y)
            center_lat=flight_frame_data["latitude"],
            center_lon=flight_frame_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=flight_frame_data["gb_yaw"],
        )

        """XXX>>>
        animals_coordinates, _ = get_objects_coordinates(
            objects_coords=boxes_centers,   # (X,Y)
            center_lat=flight_frame_data["latitude"],
            center_lon=flight_frame_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=flight_frame_data["gb_yaw"],
         )
        <<<XXX"""

        print(f"Frame location computed in {(time() - crono_start)*1000:.1f} ms. (Animals position not computed)")

        """ 
        create DEM validity/geofencing/slope masks overlapping with frame given the frame corner cooridnates (5 ms)
        """

        crono_start = time()

        frame_polygon = Polygon(corners_coordinates)  # a rectangle (may have any orientation)

        if not is_polygon_within_bounds(combined_dem_masks_tif.bounds, frame_polygon):
            print(f"ERROR: Cannot monitor the safety of animals when the drones is leaving the DEM area")
            print(f"DEM bounds: {combined_dem_masks_tif.bounds}")
            print(f"FRAME bounds: {corners_coordinates}")
            exit()

        center_coords = (flight_frame_data["longitude"], flight_frame_data["latitude"])

        masks_window, masks_window_transform, masks_window_bounds, masks_window_size = extract_masks_window(
            raster_dataset=combined_dem_masks_tif,
            center_lonlat=center_coords,
            rectangle_lonlat=corners_coordinates,
        )

        # find the distance between two points on opposite side of the window at the drone latitude
        window_size_m = get_window_size_m(flight_frame_data["latitude"], masks_window_bounds)
        dem_pixel_size_m = window_size_m / masks_window_size

        combined_dem_mask_over_frame = map_window_onto_drone_frame(
            window=masks_window,
            window_transform=masks_window_transform,
            window_crs=combined_dem_masks_tif.crs,
            center_coords=center_coords,
            corners_coords=corners_coordinates,
            angle_wrt_north=flight_frame_data["gb_yaw"],
            frame_width=frame_width,
            frame_height=frame_height,
            window_size_pixels=masks_window_size,
            window_size_m=window_size_m,
            frame_pixel_size_m=meters_per_pixel,
        )

        dem_nodata_danger_mask = combined_dem_mask_over_frame[0]
        geofencing_danger_mask = combined_dem_mask_over_frame[1]
        slope_danger_mask = combined_dem_mask_over_frame[2]

        print(f"Frame-overlapping DEM validity and slope masks computed in {(time() - crono_start)*1000:.1f} ms")

        """ Compute safety areas around each animal, and check for intersections with danger masks. If yes, send alert (12 ms)"""

        crono_start = time()

        # create the safety mask
        inter = time()
        safety_mask = create_safety_mask(frame_height, frame_width, boxes_centers, safety_radius_pixels)
        st_time = time()-inter

        inter = time()
        # create danger mask
        combined_danger_mask = merge_3d_mask(np.stack([
            segment_danger_mask,
            dem_nodata_danger_mask,
            geofencing_danger_mask,
            slope_danger_mask,
        ]))
        dm_time = time() - inter

        # create the intersection mask between safety areas and dangerous areas masks
        inter = time()
        intersection_segment = np.logical_and(safety_mask, segment_danger_mask)
        intersection_nodata = np.logical_and(safety_mask, dem_nodata_danger_mask)
        intersection_geofencing = np.logical_and(safety_mask, geofencing_danger_mask)
        intersection_slope = np.logical_and(safety_mask, slope_danger_mask)
        ands_time = time() - inter

        # compute overall intersection
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

        # if cooldown has passed, check for dangerous overlapping and report them with the appropriate string(s)
        inter = time()
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        danger_exists = False
        if cooldown_has_passed:
            danger_types = []
            if np.any(intersection_segment > 0):
                danger_types.append("Vehicles Danger")
            if np.any(intersection_nodata > 0):
                danger_types.append("Missing DEM data Danger")
            if np.any(intersection_geofencing > 0):
                danger_types.append("Out of Geofenced area Danger DEM data")
            if np.any(intersection_slope > 0):
                danger_types.append("High slope Danger")

            if len(danger_types) > 0:
                danger_exists = True
                danger_type_str = " & ".join(danger_types)
                send_alert(alerts_file, frame_id, danger_type_str)
                last_alert_frame_id = frame_id

        compint_time = time() - inter

        print(f"Danger analysis and reporting completed in {(time() - crono_start)*1000:.1f} ms")
        print(f"\tCompute safety mask mask in {st_time*1000:.1f} ms")
        print(f"\tCompute combined danger mask in {dm_time*1000:.1f} ms")
        print(f"\tCompute single intersections masks in {ands_time*1000:.1f} ms")
        print(f"\tCompute combined intersection mask in {inter_time*1000:.1f} ms")
        print(f"\tCompute and reporting danger types in {compint_time*1000:.1f} ms")

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

        # annotations can be skipped if videos are not be saved and no animal is in danger
        if output_args["save_videos"] or (danger_exists and cooldown_has_passed):

            crono_start = time()
            
            inter = time()
            annotated_frame = frame.copy()  # copy of the original frame on which to draw
            print(f"\tFrame copy generated in {(time()-inter)*1000:.1f} ms")

            # draw safety circles
            inter = time()
            draw_safety_areas(annotated_frame, boxes_centers, safety_radius_pixels)
            print(f"\tsafety areas generated in {(time()-inter)*1000:.1f} ms")

            # Overlay dangerous areas in red on the annotated frame
            inter = time()
            draw_dangerous_area(annotated_frame, combined_danger_mask_no_intersections, combined_intersections)
            print(f"\tDangerous areas AND danger INTERSECTION drawn in {(time()-inter)*1000:.1f} ms")

            # draw detection boxes
            inter = time()
            draw_detections(annotated_frame, classes, boxes_corner1, boxes_corner2)
            print(f"\tDetections drawn in {(time()-inter)*1000:.1f} ms")

            # draw animal count
            inter = time()
            draw_count(classes, annotated_frame)
            print(f"\tAnimal Count drawn in {(time()-inter)*1000:.1f} ms")

            inter = time()
            if output_args["save_videos"]:  # if the annotation code has been entered because saving the videos ...
                # save the annotated frame
                annotated_writer.write(annotated_frame)
            if danger_exists:  # if annotation code has been entered because an animal is in danger after cooldown ...
                annotated_img_path = Path(output_dir, f"danger_frame_{frame_id}_annotated.png")
                cv2.imwrite(annotated_img_path, annotated_frame)
            print(f"\tFrame saving completed in {(time()-inter)*1000:.1f} ms")

            print(f"Video annotations completed in {(time() - crono_start)*1000:.1f} ms")

        iteration_time = time() - iteration_start_time
        print(f"Iteration completed in {iteration_time*1000:.1f} ms. Equivalent fps = {1/iteration_time:.2f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    print(f"Danger Analysis for {processed_frames_counter} frames (out of {total_frames}) completed in {total_time:.1f} seconds")
    real_processing_rate = processed_frames_counter / total_time
    print(f"Real processing rate: {real_processing_rate:.1f} fps. Real time: {real_processing_rate >= fps}")
    apparent_processing_rate = total_frames / total_time
    print(f"Apparent processing rate: {apparent_processing_rate:.1f} fps. Real time: {apparent_processing_rate >= fps}")

    combined_dem_masks_tif.close()

    flight_data_file.close()
    alerts_file.close()

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")
