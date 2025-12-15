import multiprocessing as mp
from queue import Empty

import logging
import numpy as np
from shapely import Polygon
from src.shared.drone_utils.gsd import get_meters_per_pixel
from src.shared.drone_utils.localization import get_objects_coordinates
from src.danger_detection.utils import (close_tifs, compute_slope_mask_horn,
                                 create_geofencing_mask_runtime,
                                 extract_dem_window, open_dem_tifs, get_frame_transform,
                                 get_window_size_m,
                                 map_window_onto_drone_frame)
from src.danger_detection.processes.messages import GeoResult
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
from src.shared.processes.constants import POISON_PILL, QUEUE_GET_TIMEOUT

from time import time

# ================================================================

logger = logging.getLogger("main.danger_geo")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_geo.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class GeoWorker(mp.Process):

    def __init__(self, input_queue, result_queue, input_args, drone_args):

        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue

        self.input_args = input_args
        self.drone_args = drone_args

    def run(self):
        """
        Main loop of the process: 
        - instantiate files
        - processes frames.
        - Shuts down when it receives a poison pill, forwarding the termination signal to the next process in the sequence
        """
        
        logger.info("Geo-handling process started.")

        # Open the DEM, if provided
        # open the DEM mask, if provided and DEM provided
        dem_tif, dem_mask_tif = open_dem_tifs(
            dem_path=self.input_args["dem"], 
            dem_mask_path=self.input_args["dem_mask"]
        )

        # placeholders, will be set based on first frame shape
        frame_width = None
        frame_height = None
        frame_corners = None
        
        logger.info("Running...")


        while True:

            iter_start = time()

            try:
                # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                frame_telemetry_object: CombinedFrameTelemetryQueueObject  = self.input_queue.get(timeout=QUEUE_GET_TIMEOUT)
            except Empty:
                logger.warning(f"Input queue timed out ({QUEUE_GET_TIMEOUT} secsonds). Upstream producer may be stalled. Retrying...")
                continue # Go back and try to get again

            if frame_telemetry_object == POISON_PILL:
                logger.info("Found sentinel value on queue.")
                # Signal end of processing to the next process in the chain
                # Ensure the poison pill is passed on by using a blocking put (will wait until queue is free) 
                self.result_queue.put(POISON_PILL)  
                logger.info("Sentinel value has been passed on to the next process. Terminating geo data handling process...")
                break
            
            # setup frame dimensions and corners based on the first frame received
            if frame_width is None and frame_height is None and frame_corners is None:
                frame_height, frame_width, _ = frame_telemetry_object.frame
                frame_corners = np.array([
                    [0, 0],  # upper left  (0,   0   )
                    [frame_width - 1, 0],  # upper right (C-1, 0   )
                    [frame_width - 1, frame_height - 1],  # lower right (C-1, R-1 )
                    [0, frame_height - 1],  # lower left  (0,   R-1 )
                ])
                logger.debug(f"Geo-data handler got frame WxH = {frame_width} x {frame_height}")


            # If there is no telemetry associated with the frame, then no processing can be performed. 
            # Assume no dangers and return clean danger masks.
            if frame_telemetry_object.telemetry is None:
                result = GeoResult(
                    frame_id=frame_telemetry_object.frame_id,
                    safety_radius_pixels=-1,
                    nodata_dem_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                    geofencing_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                    slope_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                )
                # blocking put ensures the result is put on the target queue eventually (when space becomes available)
                # cannot discard this processing since the AI modules results must be combined later, and all resuls must be present
                self.result_queue.put(result)
                logger.warning("No telemetry match found, skipping GEO step")
                logger.debug(f"completed in {(time()-iter_start)*1000:.2f} ms")
                continue
                # iteration concluded, move on to the next one

            # OTHERWISE, continue processing based on telemetry info
            # load frame flight data
            frame_flight_data = frame_telemetry_object.telemetry

            # Perform the pixels to meters conversion using the sensor resolution
            meters_per_pixel = get_meters_per_pixel(
                rel_altitude_m=frame_flight_data["rel_alt"],
                focal_length_mm=self.drone_args["true_focal_len_mm"],
                sensor_width_mm=self.drone_args["sensor_width_mm"],
                sensor_height_mm=self.drone_args["sensor_height_mm"],
                sensor_width_pixels=self.drone_args["sensor_width_pixels"],
                sensor_height_pixels=self.drone_args["sensor_height_pixels"],
                image_width_pixels=frame_width,
                image_height_pixels=frame_height,
            )

            # ============== COMPUTE SAFETY AREA RADIUS SIZE IN PIXELS  ===================================
            safety_radius_pixels = int(self.input_args["safety_radius_m"] / meters_per_pixel)

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
                    slope_threshold_deg=self.input_args["slope_angle_threshold"]
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

            if self.input_args["geofencing_vertexes"] is not None:
                # compute geofencing directly on the full size frame
                # independently of other two masks for flexibility (slower), as it should not require the dem data
                geofencing_danger_mask = create_geofencing_mask_runtime(
                    frame_width=frame_width,
                    frame_height=frame_height,
                    transform=frame_transform,
                    polygon=Polygon(self.input_args["geofencing_vertexes"])
                )
            else:
                geofencing_danger_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

            # ============== PUT RESULTS ON QUEUE ========================

            result = GeoResult(
                frame_id=frame_telemetry_object.frame_id,
                safety_radius_pixels=safety_radius_pixels,
                nodata_dem_mask=dem_nodata_danger_mask,
                geofencing_mask=geofencing_danger_mask,
                slope_mask=slope_danger_mask,
            )
            # blocking put ensures the result is put on the target queue eventually (when space becomes available)
            # cannot discard this processing since the AI modules results must be combined later, and all resuls must be present
            self.result_queue.put(result)

            logger.debug(f"completed in {(time()-iter_start)*1000:.2f} ms")


        # end of process, cleanup and log conclusion
        close_tifs([dem_tif, dem_mask_tif])     # close tif files
        logger.info("Geo data handling process terminated successfully.")

        



