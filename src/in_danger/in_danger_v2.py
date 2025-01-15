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
    plot_dem_preprocessing = True

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
    dangerous_slope_mask = create_dangerous_slope_mask(
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
        dst.write(aggregated_dem_masks[0], 1)  # Write DEM nodata mask to band 1
        dst.write(aggregated_dem_masks[1], 2)  # Write DEM geofencing mask to band 2
        dst.write(aggregated_dem_masks[2], 3)  # Write DEM slope mask to band 3

    # Close previously opened tifs
    dem_tif.close()
    if dem_mask_tif is not None:
        dem_mask_tif.close()

    # Reopen the mask tif
    combined_dem_masks_tif = rasterio.open(combined_dem_masks_tif_path)

    print(f"DEM data preprocessing took {time()-crono_start:.1f} seconds")

    # ============== LOAD AI MODELS ===================================

    crono_start = time()

    # Load AI models
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")

    # Load YOLO models
    detector = YOLO(detection_model_checkpoint, task="detect")  # Animal detection model
    segmenter = YOLO(segmentation_model_checkpoint, task="segment")  # Dangerous terrain segmentation model

    print(f"AI models loaded in {time()-crono_start:.1f} seconds")

    # ============== LOAD FLIGHT INFO ===================================

    crono_start = time()

    # Open drone flight data
    flight_data_file_path = Path(input_args["flight_data"])
    flight_data_file = open(flight_data_file_path, "r")

    print(f"Flight data loaded in {time()-crono_start:.1f} seconds")

    # ============== LOAD VIDEO INFO ===================================

    crono_start = time()

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # avoids unreasonable video strides
    input_args["vid_stride"] = max(1, min(input_args["vid_stride"], total_frames))

    print(f"Video info loaded in {time()-crono_start:.1f} seconds")

    # ============== LOAD ALERTS FILE ===================================

    crono_start = time()

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    print(f"Alerts file loaded in {time()-crono_start:.1f} seconds")

    # ============== LOAD VIDEO WRITERS ===================================

    crono_start = time()

    annotated_writer = None
    mask_writer = None

    if output_args["save_videos"]:
        annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
        annotated_writer = cv2.VideoWriter(
            filename=annotated_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

        mask_video_path = (output_dir / output_args["mask_video_name"]).with_suffix(".mp4")
        mask_writer = cv2.VideoWriter(
            filename=mask_video_path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=fps,
            frameSize=(frame_width, frame_height)
        )

    print(f"Video Writers loaded in {time()-crono_start:.1f} seconds")

    # ============== BEGIN VIDEO PROCESSING ===================================

    # Frame counter
    true_frame_id = 0
    frame_id = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = output_args["alerts_cooldown_seconds"] * fps   # convert cooldown from seconds to frames
    last_alert_frame = - fps  # to avoid dealing with initial None value, at frame 0 alert is allowed

    # Time keeper
    start_time = time()

    # Video processing loop
    while cap.isOpened():
        crono_iter_start = time()
        success, frame = cap.read()
        if not success:
            print("Video processing has been successfully completed.")
            break

        if true_frame_id % input_args["vid_stride"] != 0:
            true_frame_id += 1  # Update frame ID for logging
            print("skippin' frame")
            continue  # go to next frame directly (processes 1 frame every 'vid_stride' frames)

        frame_id += 1  # update the actual number of frames processed
        true_frame_id += 1  # Update frame ID for logging
        print(f"\n------------- Processing frame {true_frame_id}/{total_frames}-----------")

        # cv2 loads image in BGR, convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # load frame flight data
        flight_frame_data = parse_drone_frame(flight_data_file, frame_id=true_frame_id)

        """ Perform detection and get bounding boxes (? ms)"""

        crono_start = time()
        # Detect animals in frame
        classes, boxes_centers, boxes_wh, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
        print(f"detection of animals completed in {time() - crono_start} seconds")

        """ Perform segmentation and build segmentation danger mask (? ms)"""

        crono_start = time()
        # Highlight dangerous objects
        segment_danger_mask = perform_segmentation(segmenter, frame, segmentation_args)
        print(f"Segmentation and danger mask creation completed in {time() - crono_start} seconds")

        """ extract coordinates of frame (and animals) from drone position and flight height (? ms)"""

        crono_start = time()

        # Perform the pixels to meters conversion using the sensor resolution
        meters_per_pixel = get_meters_per_pixel(
            rel_altitude_m=flight_frame_data["rel_alt"],
            focal_length_mm=flight_frame_data["focal_len"],
            sensor_width_mm=input_args["drone_sensor_width_mm"],
            sensor_height_mm=input_args["drone_sensor_height_mm"],
            sensor_width_pixels=input_args["drone_sensor_width_pixels"],
            sensor_height_pixels=input_args["drone_sensor_height_pixels"],
            image_width_pixels=frame_width,
            image_height_pixels=frame_height,
        )

        meters_per_pixel *= 3 # TODO remove DBG

        safety_radius_pixels = int(input_args["safety_radius_m"] / meters_per_pixel)

        frame_width_m = frame_width * meters_per_pixel
        frame_height_m = frame_height * meters_per_pixel
        print(f"Frame dimensions: {frame_width_m}x{frame_height_m} meters")

        frame_corners = np.array([
            [0, 0],
            [0, frame_width-1],
            [frame_height-1, 0],
            [frame_height-1, frame_width-1],
        ])

        print(frame_corners)

        # get the coordinates of the 4 corners of the frame.
        # The rectangle may be oriented in any direction wrt North
        corners_coordinates = get_objects_coordinates(
            objects_coords=frame_corners,
            center_lat=flight_frame_data["latitude"],
            center_lon=flight_frame_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=-flight_frame_data["gb_yaw"],
        )

        """XXX>>>
        animals_coordinates = get_objects_coordinates(
            objects_coords=boxes_centers,
            center_lat=flight_frame_data["latitude"],
            center_lon=flight_frame_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=-flight_frame_data["gb_yaw"],
         )
        <<<XXX"""

        print(f"Frame location computed in {time() - crono_start} seconds. (Animals position not computed)")

        """ create DEM validity/geofencing/slope masks overlapping with frame given the frame corner cooridnates (? ms)"""

        crono_start = time()

        frame_polygon = Polygon(corners_coordinates)  # a rectangle (may have any orientation)

        if not is_polygon_within_bounds(combined_dem_masks_tif.bounds, frame_polygon):
            print(f"ERROR: Cannot monitor the safety of animals when the drones is leaving the DEM area")
            print(f"DEM bounds: {combined_dem_masks_tif.bounds}")
            print(f"FRAME bounds: {corners_coordinates}")
            exit()

        area_masking_nodata = 255
        combined_dem_mask_over_frame, _ = rasterio_mask(combined_dem_masks_tif, [frame_polygon], nodata=area_masking_nodata, crop=True)
        print(combined_dem_mask_over_frame.shape)
        combined_dem_mask_over_frame = np.array([rotate_array(combined_dem_mask_over_frame[c], angle=-flight_frame_data["gb_yaw"], reshape=True, order=0) for c in range(combined_dem_mask_over_frame.shape[0])])
        print(combined_dem_mask_over_frame.shape)
        combined_dem_mask_over_frame = clip_array_into_rectangle_no_nodata(combined_dem_mask_over_frame, area_masking_nodata)
        print(combined_dem_mask_over_frame.shape)
        combined_dem_mask_over_frame = upscale_array_to_image_size(combined_dem_mask_over_frame, frame_height, frame_width)
        print(combined_dem_mask_over_frame.shape)

        dem_nodata_danger_mask = combined_dem_mask_over_frame[0]
        geofencing_danger_mask = combined_dem_mask_over_frame[1]
        slope_danger_mask = combined_dem_mask_over_frame[2]

        print(f"Frame-overlapping DEM validity and slope masks computed in {time() - crono_start} seconds")

        """ Compute safety areas around each animal, and check for intersections with danger masks. If yes, send alert (? ms)"""

        crono_start = time()

        # create the safety mask
        safety_mask = create_safety_mask(frame_height, frame_width, boxes_centers, safety_radius_pixels)

        # create danger mask
        combined_danger_mask = merge_3d_mask(np.stack([
            segment_danger_mask,
            dem_nodata_danger_mask,
            geofencing_danger_mask,
            slope_danger_mask,
        ]))

        # create the intersection mask between safety areas and segmentation dangerous areas
        intersection_segment = np.logical_and(safety_mask, segment_danger_mask)
        intersection_nodata = np.logical_and(safety_mask, dem_nodata_danger_mask)
        intersection_geofencing = np.logical_and(safety_mask, geofencing_danger_mask)
        intersection_slope = np.logical_and(safety_mask, slope_danger_mask)

        # compute overall intersection
        combined_intersections = merge_3d_mask(np.stack([
            intersection_segment,
            intersection_nodata,
            intersection_geofencing,
            intersection_slope
        ]))

        """XXX>>>
        # Check for intersection between safety area and dangerous areas, non-zero intersection indicates overlap
        danger_exists = np.any(combined_intersections > 0)
        cooldown_has_passed = (true_frame_id - last_alert_frame) >= alerts_frames_cooldown

        # if overlap exists and cooldown has passed, send alert and update cooldown period
        if danger_exists and cooldown_has_passed:
            send_alert(alerts_file, true_frame_id)
            last_alert_frame = true_frame_id
        <<<XXX"""

        # if cooldown has passed, check for damngerous overlapping and report them with the appropriate string(s)
        cooldown_has_passed = (true_frame_id - last_alert_frame) >= alerts_frames_cooldown
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
                send_alert(alerts_file, true_frame_id, danger_type_str)
                last_alert_frame = true_frame_id

        print(f"Danger analysis and reporting completed in {time() - crono_start} seconds")

        """ Additional annotations if videos are to be saved, or for frames where danger exist (? ms)"""


        # annotations can be skipped if videos are not be saved and no animal is in danger
        if output_args["save_videos"] or (danger_exists and cooldown_has_passed):

            crono_start = time()

            annotated_frame = frame.copy()  # copy of the original frame on which to draw
            rgb_mask_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            # draw safety circles
            draw_safety_areas(annotated_frame, rgb_mask_frame, boxes_centers, safety_radius_pixels)

            # Overlay dangerous areas in red on the annotated frame
            annotated_frame = draw_dangerous_area(annotated_frame, rgb_mask_frame, combined_danger_mask, combined_intersections)
            """
            # Overlay dangerous areas in shades of red on the annotated frame, multiask version
            draw_dangerous_area_multimask(annotated_frame, rgb_mask_frame, combined_danger_mask, combined_intersections, shades_of_red)
            """

            # Highlight intersection area in yellow
            draw_animal_in_danger_area(rgb_mask_frame, combined_intersections)
            """
            # Highlight intersection area in shades of yellow, multimask version
            draw_animal_in_danger_area_multimask(rgb_mask_frame, combined_intersections, shades_of_yellow)
            """

            # draw detection boxes
            draw_detections(annotated_frame, rgb_mask_frame, classes, boxes_corner1, boxes_corner2)

            # draw animal count
            draw_count(classes, annotated_frame, frame_height)

            if output_args["save_videos"]:  # if the annotation code has been entered because saving the videos ...
                # save the annotated rgb mask and annotated frame
                mask_writer.write(rgb_mask_frame)
                annotated_writer.write(annotated_frame)

            if danger_exists:  # if annotation code has been entered because an animal is in danger after cooldown ...
                mask_img_path = Path(output_dir, f"danger_frame_{true_frame_id}_mask.png")
                annotated_img_path = Path(output_dir, f"danger_frame_{true_frame_id}_annotated.png")
                cv2.imwrite(mask_img_path, rgb_mask_frame)
                cv2.imwrite(annotated_img_path, annotated_frame)

            print(f"Video annotations completed in {time() - crono_start} seconds")

        iteration_time = time() - crono_iter_start
        print(f"Iteration completed in {iteration_time} seconds. Equivalent fps = {1/iteration_time:.2f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - start_time
    processing_speed = total_frames / total_time
    print(f"Detection and segmentation for  {total_frames} completed in end {total_time:.1f} seconds")
    print(f"Processing rate: {processing_speed:.2f} fps")
    print(f"Input video fps: {fps}")
    print(f"Processing is real time: {processing_speed >= fps}")

    combined_dem_masks_tif.close()

    flight_data_file.close()
    alerts_file.close()

    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()
    if mask_writer is not None:
        mask_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")


