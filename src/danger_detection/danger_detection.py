from typing import Any
from ultralytics import YOLO
from shapely import Polygon
from pathlib import Path
import cv2
from time import time
import numpy as np

from src.danger_detection.utils import *

from src.danger_detection.detection.detection import perform_detection
from src.danger_detection.segmentation.segmentation import create_onnx_segmentation_session, perform_segmentation
from src.danger_detection.output.alerts import send_alert
from src.danger_detection.output.frames import get_danger_intersect_colored_frames, annotate_and_save_frame

from src.shared.drone_utils.flight_logs import parse_drone_flight_data
from src.shared.drone_utils.gsd import get_meters_per_pixel
from src.shared.drone_utils.localization import get_objects_coordinates


def perform_danger_detection(
        input_args: dict[str, Any],
        output_args: dict[str, Any],
        detection_args: dict[str, Any],
        segmentation_args: dict[str, Any],
        drone_args: dict[str, Any],
) -> None:

    # ============== CREATE OUTPUT DIRECTORY ===================================

    output_dir = Path(output_args["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============== LOAD DEM (+ MASK) ===================================

    # Open the DEM, if provided
    # open the DEM mask, if provided and DEM provided
    dem_tif, dem_mask_tif = open_dem_tifs(dem_path=input_args["dem"], dem_mask_path=input_args["dem_mask"])

    # ============== LOAD AI MODELS ===================================

    # Load detection model
    detection_model_checkpoint = detection_args.pop("model_checkpoint")
    detector = YOLO(detection_model_checkpoint, task="detect")  # Animal detection model

    # Load segmentation model
    segmentation_model_checkpoint = segmentation_args.pop("model_checkpoint")
    segmenter_session, segmenter_input_name,  segmenter_input_shape = create_onnx_segmentation_session(segmentation_model_checkpoint)  # Dangerous terrain segmentation model

    # ============== LOAD DETECTION CLASSES INFO ===================================

    # prepare detection classes names and number
    classes_names = detector.names  # Dictionary of class names
    num_classes = len(classes_names)

    # ============== LOAD FLIGHT INFO ===================================

    # Open drone flight data
    flight_data_file_path = Path(input_args["flight_data"])
    flight_data_file = open(flight_data_file_path, "r")

    # ============== LOAD INPUT VIDEO INFO ===================================

    # Open video and get properties
    cap = cv2.VideoCapture(input_args["source"])
    assert cap.isOpened(), "Error reading video file"

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # avoids unreasonable video strides
    input_args["vid_stride"] = max(1, min(input_args["vid_stride"], total_frames))
    print(f"Processing 1 frame every {input_args['vid_stride']}")

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

    # ============== LOAD OUTPUT VIDEO WRITER ===================================

    annotated_video_path = (output_dir / output_args["annotated_video_name"]).with_suffix(".mp4")
    annotated_writer = cv2.VideoWriter(
        filename=annotated_video_path,
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=fps,
        frameSize=(frame_width, frame_height)
    )

    frame_shape = (frame_height, frame_width, 3)
    color_danger_frame, color_intersect_frame = get_danger_intersect_colored_frames(shape=frame_shape)

    # ============== BEGIN VIDEO PROCESSING ===================================

    # Frame counter
    frame_id = 0
    processed_frames_counter = 0

    # Alert cooldown initialization
    alerts_frames_cooldown = max(1, int(input_args["alerts_cooldown_seconds"] * fps))   # convert cooldown from seconds to frames
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
            continue  # skip to next frame directly (processes 1 frame every 'vid_stride' frames)

        processed_frames_counter += 1  # update the actual number of frames processed
        frame_id += 1  # Update frame ID
        print(f"\n------------- Processing frame {frame_id}/{total_frames}-----------")

        # ============== PERFORM DETECTION ===================================
        crono_start = time()
        # Detect animals in frame, in the form (X,Y) = (COL, ROW) of frame
        classes, boxes_centers, boxes_corner1, boxes_corner2 = perform_detection(detector, frame, detection_args)
        print(f"detection of animals completed in {(time() - crono_start)*1000:.1f} ms")

        # ============== PERFORM SEGMENTATION ===================================

        crono_start = time()
        # Highlight dangerous objects
        segment_roads_danger_mask, segment_vehicles_danger_mask = perform_segmentation(segmenter_session, segmenter_input_name,  segmenter_input_shape, frame, segmentation_args)
        print(f"Segmentation and danger mask creation completed in {(time() - crono_start)*1000:.1f} ms")

        # ============== COMPUTE FRAME GROUND RESOLUTION IN METERS/PIXEL  ===================================
        crono_start = time()

        # load frame flight data
        frame_flight_data = parse_drone_flight_data(flight_data_file, frame_id)

        # Perform the pixels to meters conversion using the sensor resolution
        meters_per_pixel = get_meters_per_pixel(
            rel_altitude_m=frame_flight_data["rel_alt"],
            focal_length_mm=drone_args["true_focal_len_mm"],
            sensor_width_mm=drone_args["sensor_width_mm"],
            sensor_height_mm=drone_args["sensor_height_mm"],
            sensor_width_pixels=drone_args["sensor_width_pixels"],
            sensor_height_pixels=drone_args["sensor_height_pixels"],
            image_width_pixels=frame_width,
            image_height_pixels=frame_height,
        )

        # ============== COMPUTE FRAME SIZE IN METERS  ===================================
        """
        frame_width_m = frame_width * meters_per_pixel
        frame_height_m = frame_height * meters_per_pixel
        print(f"Frame dimensions: {frame_width_m:.2f}x{frame_height_m:.2f} meters")
        """
        # ============== COMPUTE SAFETY AREA RADIUS SIZE IN PIXELS  ===================================
        safety_radius_pixels = int(input_args["safety_radius_m"] / meters_per_pixel)

        # ============== COMPUTE LOCATION (LNG,LAT) OF FRAME CORNERS  ===================================
        # get the coordinates of the 4 corners of the frame.
        # The rectangle may be oriented in any direction wrt North
        corners_coordinates = get_objects_coordinates(
            objects_coords=frame_corners,   # (X,Y) expected input
            center_lat=frame_flight_data["latitude"],
            center_lon=frame_flight_data["longitude"],
            frame_width_pixels=frame_width,
            frame_height_pixels=frame_height,
            meters_per_pixel=meters_per_pixel,
            angle_wrt_north=frame_flight_data["gb_yaw"],
        )

        # ============== COMPUTE LOCATION (LNG,LAT) OF ANIMALS  ===================================
        """
        if boxes_centers.size > 0:
            animals_coordinates = get_objects_coordinates(
                objects_coords=boxes_centers,   # (X,Y)
                center_lat=frame_flight_data["latitude"],
                center_lon=frame_flight_data["longitude"],
                frame_width_pixels=frame_width,
                frame_height_pixels=frame_height,
                meters_per_pixel=meters_per_pixel,
                angle_wrt_north=frame_flight_data["gb_yaw"],
             )
        """

        print(f"Frame location computed in {(time() - crono_start)*1000:.1f} ms. (Animals position not computed)")

        # ============== CREATE TRANSFORM TO EXTRACT THE FRAME AREA FROM THE RASTER ========================

        # compute the Affine transform to extract a portion of the raster corresponding to the area in the frame
        frame_transform = get_frame_transform(
            height=frame_height,
            width=frame_width,
            drone_ul=tuple(corners_coordinates[0]),  # (lon, lat) for upper-left corner
            drone_ur=tuple(corners_coordinates[1]),  # (lon, lat) for upper-right corner
            drone_bl=tuple(corners_coordinates[3]),  # (lon, lat) for bottom-left corner
        )

        # ============== COMPUTE DEM (+MASK) WINDOW ENCOMPASSING THE FRAME  ===================================

        crono_start = time()

        if dem_tif is not None:

            center_coords = (frame_flight_data["longitude"], frame_flight_data["latitude"])

            dem_window, dem_mask_window, dem_window_transform, dem_window_bounds, dem_window_size = extract_dem_window(
                dem_tif=dem_tif,
                dem_mask_tif=dem_mask_tif,
                center_lonlat=center_coords,
                rectangle_lonlat=corners_coordinates,
            )
            # dem_window and dem_mask_window are (1,dem_window_size,dem_window_size) arrays
            assert dem_window.shape == (1, dem_window_size, dem_window_size)
            assert dem_mask_window.shape == (1, dem_window_size, dem_window_size)

            # find the distance in meters between two points on opposite side of the window at the drone latitude
            dem_window_size_m = get_window_size_m(frame_flight_data["latitude"], dem_window_bounds)
            # compute the resolution of each dem pixel in meters
            dem_pixel_size_m = dem_window_size_m / dem_window_size

            # ============== COMPUTE SLOPE MASK FROM DEM WINDOW ===================================

            # compute the slope mask using the dem window and info about the resolution of each pixel
            slope_mask_window = compute_slope_mask_horn(
                elev_array=dem_window,
                pixel_size=dem_pixel_size_m,
                slope_threshold_deg=input_args["slope_angle_threshold"]
            )

            # ============== ROTATE & UPSCALE MASKS USING FRAME TRANSFORM ========================

            # stack the dem_nodata and dem_slope masks to form a (2, dem_window_size, dem_window_size) mask array
            masks_window = np.concatenate((dem_mask_window, slope_mask_window), axis=0)
            assert masks_window.shape == (2, dem_window_size, dem_window_size)

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
        else:
            dem_nodata_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
            slope_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        # ============== CREATE GEOFENCING MASK ========================

        # compute geofencing directly on the full size frame
        # independently of other two masks for flexibility (slower), as it should not require the dem data
        if input_args["geofencing_vertexes"] is not None:
            geofencing_danger_mask = create_geofencing_mask_runtime(
                frame_width=frame_width,
                frame_height=frame_height,
                transform=frame_transform,
                polygon=Polygon(input_args["geofencing_vertexes"])
            )
        else:
            geofencing_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

        print(f"Frame-overlapping DEM validity and slope masks computed in {(time() - crono_start)*1000:.1f} ms")

        # ============== CHECK DANGER TYPES AND CREATE DANGER/INTERSECTION MASKS ================
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
            segment_roads_danger_mask,
            segment_vehicles_danger_mask,
            dem_nodata_danger_mask,
            geofencing_danger_mask,
            slope_danger_mask,
        )

        # ============== RAISE ALERTS IF NEEDED ================

        # verify whether the cooldown has passed
        cooldown_has_passed = (frame_id - last_alert_frame_id) >= alerts_frames_cooldown
        # verify whether danger animals in danger have been detected
        danger_exists = len(danger_types) > 0

        # report danger with the appropriate string (if needed)
        if cooldown_has_passed and danger_exists:
            danger_type_str = " & ".join(danger_types)
            send_alert(alerts_file, frame_id, danger_type_str)
            last_alert_frame_id = frame_id

        # ============== ANNOTATE FRAME ==================================

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
            color_danger_frame,
            color_intersect_frame,
        )

        iteration_time = time() - iteration_start_time
        print(f"Iteration completed in {iteration_time*1000:.1f} ms. Equivalent fps = {1/iteration_time:.3f}")

    """ Processing completed, print stats and release resources"""

    total_time = time() - processing_start_time
    print(f"Danger Analysis for {processed_frames_counter} frames (out of {total_frames}) completed in {total_time:.3f} seconds")
    real_processing_rate = processed_frames_counter / total_time
    print(f"Real processing rate: {real_processing_rate:.3f} fps. Real time: {real_processing_rate >= fps}")
    apparent_processing_rate = total_frames / total_time
    print(f"Apparent processing rate: {apparent_processing_rate:.3f} fps. Real time: {apparent_processing_rate >= fps}")

    # close tifs
    close_tifs([dem_tif, dem_mask_tif])

    # close files
    flight_data_file.close()
    alerts_file.close()

    # close videos
    cap.release()
    annotated_writer.release()

    print(f"Videos and alerts log have been saved at {output_dir}")
