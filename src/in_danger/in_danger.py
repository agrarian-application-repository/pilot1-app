from typing import Any
from ultralytics import YOLO
from shapely import Polygon
from pathlib import Path
import cv2
from time import time
import numpy as np

from src.in_danger.in_danger_v2_utils import *


def perform_in_danger_analysis(
        input_args: dict[str:Any],
        output_args: dict[str:Any],
        detection_args: dict[str:Any],
        segmentation_args: dict[str:Any],
) -> None:

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============== LOAD THE DEM ===================================

    # Open the DEM
    dem_tif = get_dem(dem_path=input_args["dem"])

    # Open the DEM mask, if provided
    dem_mask_tif = get_dem_mask(dem_mask_path=input_args["dem_mask"])

    # ============== LOAD AI MODELS ===================================

    # Load AI models
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")

    # Load YOLO models
    detector = YOLO(detection_model_checkpoint, task="detect")  # Animal detection model
    segmenter = YOLO(segmentation_model_checkpoint, task="segment")  # Dangerous terrain segmentation model

    # prepare detection classes names and number
    classes_names = detector.names  # Dictionary of class names
    num_classes = len(detection_args["classes"]) if detection_args["classes"] is not None else len(classes_names)

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

    # set the col/row identifier of the frame corners (x,y)
    frame_corners = np.array([
        [0                  , 0               ],    # upper left  (0,   0   )
        [frame_width - 1    , 0               ],    # upper right (C-1, 0   )
        [frame_width - 1    , frame_height - 1],    # lower right (C-1, R-1 )
        [0                  , frame_height - 1],    # lower left  (0,   R-1 )
    ])

    # ============== LOAD ALERTS FILE ===================================

    alerts_file_path = (output_dir / output_args["alert_file_name"]).with_suffix(".txt")
    alerts_file = open(alerts_file_path, "w")

    # ============== LOAD VIDEO WRITERS ===================================

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
        if not success:
            print("Video processing has been successfully completed.")
            break

        if frame_id % input_args["vid_stride"] != 0:
            frame_id += 1  # Update frame ID
            continue  # go to next frame directly (processes 1 frame every 'vid_stride' frames)

        processed_frames_counter += 1  # update the actual number of frames processed
        frame_id += 1  # Update frame ID
        print(f"\n------------- Processing frame {frame_id}/{total_frames}-----------")

        """ 
        COMPONENT 1:
        Perform detection and get bounding boxes (13 ms)
        """
        crono_start = time()
        # Detect animals in frame, in the form (X,Y) = (COL, ROW) of frame
        classes, boxes_centers, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
        print(f"detection of animals completed in {(time() - crono_start)*1000:.1f} ms")
        """ 
        COMPONENT 2:
        Perform segmentation and build segmentation danger mask (15 ms)
        """
        crono_start = time()
        # Highlight dangerous objects
        segment_danger_mask = perform_segmentation(segmenter, frame, segmentation_args)
        print(f"Segmentation and danger mask creation completed in {(time() - crono_start)*1000:.1f} ms")

        """ 
        COMPONENT 3:
        extract coordinates of frame (and animals) from drone position and flight height (0.2 ms)
        create DEM validity/geofencing/slope masks overlapping with frame given the frame corner cooridnates (5 ms)
        """

        crono_start = time()

        # load frame flight data
        flight_frame_data = parse_drone_frame(flight_data_file, frame_id)

        # Perform the pixels to meters conversion using the sensor resolution
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

        """
        animals_coordinates, _ = get_objects_coordinates(
            objects_coords=boxes_centers,   # (X,Y)
            center_lat=flight_frame_data["latitude"],
            center_lon=flight_frame_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=flight_frame_data["gb_yaw"],
         )
        """

        print(f"Frame location computed in {(time() - crono_start)*1000:.1f} ms. (Animals position not computed)")

        crono_start = time()

        center_coords = (flight_frame_data["longitude"], flight_frame_data["latitude"])

        dem_window, dem_mask_window, dem_window_transform, dem_window_bounds, dem_window_size = extract_dem_window(
            dem_tif=dem_tif,
            dem_mask_tif=dem_mask_tif,
            center_lonlat=center_coords,
            rectangle_lonlat=corners_coordinates,
        )
        # dem_window and dem_mask_window are (1,dem_window_size,dem_window_size) arrays

        # find the distance in meters between two points on opposite side of the window at the drone latitude
        dem_window_size_m = get_window_size_m(flight_frame_data["latitude"], dem_window_bounds)
        # compute the resolution of each dem pixel in meters
        dem_pixel_size_m = dem_window_size_m / dem_window_size

        # compute the slope mask using the dem window and info about the resolution of each pixel
        slope_mask_window = compute_slope_mask_runtime(
            elev_array=dem_window,
            pixel_size=dem_pixel_size_m,
            slope_threshold_deg=input_args["slope_angle_threshold"]
        )

        # stack the dem_nodata and dem_slope masks to form a (2, dem_window_size, dem_window_size) mask array
        masks_window = np.concatenate((dem_mask_window, slope_mask_window), axis=0)
        assert masks_window.shape == (2, dem_window_size, dem_window_size)

        # compute the Affine transform to extract a portion of the raster corresponding to the area in the frame
        frame_transform = get_frame_transform(
            height=frame_height,
            width=frame_width,
            drone_ul=tuple(corners_coordinates[0]),  # (lon, lat) for upper-left corner
            drone_ur=tuple(corners_coordinates[1]),  # (lon, lat) for upper-right corner
            drone_bl=tuple(corners_coordinates[3]),  # (lon, lat) for bottom-left corner
        )

        # rotate and resample using the frame coordinates to obtain a (frame_height, frame_width) version of the
        # previously created mask, that matches the frame data
        combined_dem_mask_over_frame = map_window_onto_drone_frame(
            window=masks_window,
            window_transform=dem_window_transform,
            dst_transform=frame_transform,
            output_shape=(masks_window.shape[2], frame_height, frame_width),
            crs=dem_tif.crs
        )

        # separate the two masks
        dem_nodata_danger_mask = combined_dem_mask_over_frame[0]
        slope_danger_mask = combined_dem_mask_over_frame[1]

        # compute geofencing directly on the full size frame
        # independently of other two masks for flexibility (slower), as it should not require the dem data
        geofencing_danger_mask = create_geofencing_mask_runtime(
            frame_width=frame_width,
            frame_height=frame_height,
            transform=frame_transform,
            polygon=Polygon(input_args["geofencing_vertexes"])
        )

        print(f"Frame-overlapping DEM validity and slope masks computed in {(time() - crono_start)*1000:.1f} ms")

        """ 
        COMPONENT 4:
        Compute safety areas around each animal, and check for intersections with danger masks. 
        If intersection exists, send alert (12 ms).
        """

        danger_mask, intersection_mask, danger_types = create_dangerous_intersections_masks(
            frame_height,
            frame_width,
            boxes_centers,
            safety_radius_pixels,
            segment_danger_mask,
            dem_nodata_danger_mask,
            geofencing_danger_mask,
            slope_danger_mask,
        )

        # if cooldown has passed, check if dangers exist and report them with the appropriate string(s)
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        danger_exists = len(danger_types) > 0
        if cooldown_has_passed and danger_exists:
            danger_type_str = " & ".join(danger_types)
            send_alert(alerts_file, frame_id, danger_type_str)
            last_alert_frame_id = frame_id

        # annotate the frame and save it into the video
        # provide standalone image if danger is present and cooldown has passed to complement textual alert
        annotate_and_save_frame(
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
        )

        iteration_time = time() - iteration_start_time
        print(f"Iteration completed in {iteration_time*1000:.1f} ms. Equivalent fps = {1/iteration_time:.2f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    print(f"Danger Analysis for {processed_frames_counter} frames (out of {total_frames}) completed in {total_time:.1f} seconds")
    real_processing_rate = processed_frames_counter / total_time
    print(f"Real processing rate: {real_processing_rate:.1f} fps. Real time: {real_processing_rate >= fps}")
    apparent_processing_rate = total_frames / total_time
    print(f"Apparent processing rate: {apparent_processing_rate:.1f} fps. Real time: {apparent_processing_rate >= fps}")

    # close tifs
    dem_tif.close()
    if dem_mask_tif is not None:
        dem_mask_tif.close()

    # close files
    flight_data_file.close()
    alerts_file.close()

    # close videos
    cap.release()
    annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")
