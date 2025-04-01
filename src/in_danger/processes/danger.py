import multiprocessing as mp

from src.in_danger.utils import create_dangerous_intersections_masks
from src.in_danger.processes.results import DangerDetectionResults


class DangerDetectionWorker(mp.Process):

    def __init__(self, models_results_queues, result_queue, shared_dict):
        super().__init__()

        self.shared_dict = shared_dict

        self.models_results_queues = models_results_queues
        self.result_queue = result_queue

    def run(self):
        while True:
            # Collect one result from each model's result queue
            detection_result, segmentation_result, geo_result = [q.get() for q in self.models_results_queues]
            if detection_result is None or segmentation_result is None or geo_result is None:
                assert detection_result is None and segmentation_result is None and geo_result is None, "All three model should terminate together"
                self.result_queue.put(None)    # Signal end of processing
                print("Terminating danger detection process.")
                break

            assert detection_result.frame_id == segmentation_result.frame_id == geo_result.frame_id, "IDs should match"
            danger_mask, intersection_mask, danger_types = create_dangerous_intersections_masks(
                frame_height=self.shared_dict["frame_height"],
                frame_width=self.shared_dict["frame_width"],
                boxes_centers=detection_result.boxes_centers,
                safety_radius_pixels=geo_result.safety_radius_pixels,
                segment_danger_mask=segmentation_result.mask,
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
            )
            self.result_queue.put(result)
