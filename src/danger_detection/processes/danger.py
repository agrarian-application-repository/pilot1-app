import multiprocessing as mp
import logging
from src.danger_detection.utils import create_dangerous_intersections_masks
from src.danger_detection.processes.messages import DangerDetectionResults, DetectionResult, SegmentationResult, GeoResult
from src.shared.processes.constants import POISON_PILL, QUEUE_GET_TIMEOUT
from time import time
# ================================================================

logger = logging.getLogger("main.danger_detection")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/danger_detection.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


class DangerDetectionWorker(mp.Process):

    def __init__(self, input_queues, result_queue):
        super().__init__()

        self.input_queues = input_queues
        self.result_queue = result_queue
        
    def run(self):

        logger.info("Danger detection process started. Running ...")

        while True:
            
            iter_start = time()
            
            # Collect one result from each model's result queue
            detection_result: DetectionResult
            segmentation_result: SegmentationResult
            geo_result: GeoResult

            # since this process can only continue once all three results are colected, use blocking
            detection_result=self.input_queues[0].get()
            segmentation_result=self.input_queues[1].get()
            geo_result=self.input_queues[2].get()
            
            # if a poison pill is found on one of the input queues, propagare the stop signal and terminate
            # the poison pill can be found:
            # - on only one of the queues, due to an unexpected processing error
            # - on all queues, when the termination signal is due to the sequential shutting down on corect app termination 
            if detection_result==POISON_PILL or segmentation_result==POISON_PILL or geo_result ==POISON_PILL:
                
                if detection_result==POISON_PILL and segmentation_result==POISON_PILL and geo_result ==POISON_PILL:
                    logger.info("Found sentinel value on all input queues, this indicates corect processes termination")
                else:
                    logger.warning("Found sentinel value on some of the input queues, but not all, this indicates a processing error in one of the prior processes")

                # Signal end of processing to the next process in the chain
                # Ensure the poison pill is passed on by using a blocking put (will wait until queue is free) 
                self.result_queue.put(POISON_PILL)  
                logger.info("Sentinel value has been passed on to the next process. Terminating the animal detection process...")
                break
            
            # since processing obects are uploaded to the processing queues atomically and in order, and since this process always reriees one processes obect per qqueue
            # the processing result should always be aligned
            assert detection_result.frame_id == segmentation_result.frame_id == geo_result.frame_id, "IDs should match"
            
            
            danger_mask, intersection_mask, danger_types = create_dangerous_intersections_masks(
                frame_height=self.video_info_dict["frame_height"],
                frame_width=self.video_info_dict["frame_width"],
                boxes_centers=detection_result.boxes_centers,
                safety_radius_pixels=geo_result.safety_radius_pixels,
                segment_roads_danger_mask=segmentation_result.roads_mask,
                segment_vehicles_danger_mask=segmentation_result.vehicles_mask,
                dem_nodata_danger_mask=geo_result.nodata_dem_mask,
                geofencing_danger_mask=geo_result.geofencing_mask,
                slope_danger_mask=geo_result.slope_mask,
            )

            danger_exists = len(danger_types) > 0
            danger_type_str = " & ".join(danger_types) if danger_exists else ""

            result = DangerDetectionResults(
                frame_id=detection_result.frame_id,
                frame=detection_result.frame,
                classes_names=detection_result.classes_names,
                num_classes=detection_result.num_classes,
                classes=detection_result.classes,
                boxes_centers=detection_result.boxes_centers,
                boxes_corner1=detection_result.boxes_corner1,
                boxes_corner2=detection_result.boxes_corner2,
                safety_radius_pixels=geo_result.safety_radius_pixels,
                danger_mask=danger_mask,
                intersection_mask=intersection_mask,
                danger_types=danger_type_str,
                timestamp=detection_result.timestamp,
            )
            self.result_queue.put(result)

            logger.debug(f"frame {detection_result.frame_id} completed in {(time() - iter_start) * 1000:.2f} ms")

    # end of process, log conclusion
        logger.info("Animal detection process terminated successfully.")
