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
from src.shared.processes.constants import *
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException

from time import time, sleep

# ================================================================

logger = logging.getLogger("main.danger_geo")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_geo.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class GeoWorker(mp.Process):

    def __init__(
            self,
            input_queue: mp.Queue,
            result_queue: mp.Queue,
            error_event: mp.Event,
            input_args,
            drone_args,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue

        self.error_event = error_event

        self.input_args = input_args
        self.drone_args = drone_args

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def run(self):
        """
        Main loop of the process: 
        - instantiate files
        - processes frames.
        - Shuts down when it receives a poison pill, forwarding the termination signal to the next process in the sequence
        """
        
        logger.info("Geo-handling process started.")
        poison_pill_received = False

        # safely opens the DEM, if provided
        # safely open the DEM MASK, if provided and DEM provided
        dem_tif, dem_mask_tif = open_dem_tifs(
            dem_path=self.input_args["dem"], 
            dem_mask_path=self.input_args["dem_mask"]
        )

        # placeholders, will be set based on first frame shape
        frame_width = None
        frame_height = None
        frame_corners = None
        
        try:

            while not self.error_event.is_set():

                iter_start = time()

                # attempt to read input, retry if input queue is empty
                try:
                    # frame_telemetry_object is either a CombinedFrameTelemetryQueueObject or the POISON_PILL
                    frame_telemetry_object: CombinedFrameTelemetryQueueObject | str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug(f"Input queue timed out. Upstream producer may be stalled. Retrying...")
                    continue  # Go back and try to get again

                # check whether the input was the poison pill,
                # if yes, propagate it and terminate by breaking out of the loop
                if isinstance(frame_telemetry_object, str) and frame_telemetry_object == POISON_PILL:
                    poison_pill_received = True
                    logger.info("Found sentinel value on queue.")
                    try:
                        logger.info("Attempting to put sentinel value on output queue ...")
                        self.result_queue.put(POISON_PILL, timeout=self.poison_pill_timeout)
                        logger.info("Sentinel value has been passed on to the next process.")
                    except Exception as e:
                        logger.error(f"Error propagating Poison Pill: {e}")
                        self.error_event.set()
                        logger.warning(
                            "Error event set: force-stop application since downstream processes "
                            "are unable to receive the poison pill."
                        )
                    break

                get_time = time() - iter_start

                # setup frame dimensions and corners based on the first frame received
                if frame_width is None and frame_height is None and frame_corners is None:
                    frame_height, frame_width, _ = frame_telemetry_object.frame
                    frame_corners = np.array([
                        [0, 0],  # upper left  (0,   0   )
                        [frame_width - 1, 0],  # upper right (C-1, 0   )
                        [frame_width - 1, frame_height - 1],  # lower right (C-1, R-1 )
                        [0, frame_height - 1],  # lower left  (0,   R-1 )
                    ])
                    logger.info(f"Geo-data handler setup with frame size WxH = {frame_width} x {frame_height}")

                predict_start = time()

                # If there is no telemetry associated with the frame, then no processing can be performed.
                # Assume no dangers and set as result clean danger masks.
                if frame_telemetry_object.telemetry is None:
                    result = GeoResult(
                        frame_id=frame_telemetry_object.frame_id,
                        safety_radius_pixels=-1,
                        nodata_dem_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                        geofencing_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                        slope_mask=np.zeros((frame_height, frame_width), dtype=np.uint8),
                    )

                else:
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

                predict_time = time() - predict_start

                # put result in output queue
                append_start = time()
                try:
                    self.result_queue.put(result, timeout=self.queue_put_timeout)
                    logger.debug("Put detection results on output queue")
                except QueueFullException:
                    logger.error(
                        f"Failed to put Geo results on output queue: queue is full. "
                        f"Consumer too slow or stuck?. "
                        f"Skipping Geo results. "
                        "This might break sync between models in the next process and cause an global error event"
                    )
                append_time = time() - append_start

                iter_time = time()-iter_start

                logger.debug(
                    f"frame {frame_telemetry_object.frame_id} processed in {iter_time * 1000:.2f} ms, "
                    f"of which --> "
                    f"GET: {get_time * 1000:.2f} ms, "
                    f"PREDICT: {predict_time * 1000:.2f} ms, "
                    f"PROPAGATE: {append_time * 1000:.2f} ms."
                )
                # iteration completed correctly, move on to process next frame

        except Exception as e:
            logger.critical(f"An unexpected critical error happened in the Geo process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:

            # close tif files
            close_tifs([dem_tif, dem_mask_tif])

            # log process conclusion
            logger.info(
                "Geo process terminated successfully."
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()

