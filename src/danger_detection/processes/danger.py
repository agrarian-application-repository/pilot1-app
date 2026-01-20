import multiprocessing as mp
import logging
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException
from src.danger_detection.utils import create_dangerous_intersections_masks
from src.danger_detection.processes.messages import DangerDetectionResults, ModelsAlignmentResult
from src.shared.processes.constants import *
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

    def __init__(
            self,
            input_queue: mp.Queue,
            result_queue: mp.Queue,
            error_event: mp.Event,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queue = input_queue
        self.result_queue = result_queue
        self.error_event = error_event

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def run(self):

        logger.info("Danger detection process started.")
        poison_pill_received = False

        # lazy init on first valid frame
        frame_height = frame_width = None

        try:

            while not self.error_event.is_set():

                iter_start = time()

                try:
                    # previous_step_results is either a ModelsAlignmentResult or the poison_pill
                    previous_step_results: ModelsAlignmentResult | str = self.input_queue.get(timeout=self.queue_get_timeout)
                except QueueEmptyException:
                    logger.debug(
                        "Input queue empty, retrying data fetch ... "
                        "(Previous process too slow or stuck?)"
                    )
                    continue  # Go back and try to read again from input queue, also check the error event condition

                if isinstance(previous_step_results, str) and previous_step_results == POISON_PILL:
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
                            "Error event set: "
                            "force-stop downstream processes since they are unable to receive the poison pill."
                        )
                    break
                    # exit the outer loop and terminate the process execution

                # rename for readability
                detection_result = previous_step_results.detection_result
                segmentation_result = previous_step_results.segmentation_result
                geo_result = previous_step_results.geo_result

                # lazy-init on first frame
                if frame_height is None and frame_width is None:
                    frame_height, frame_width, _ = detection_result.frame.shape

                # models outputs combining and processing
                danger_mask, intersection_mask, danger_types = create_dangerous_intersections_masks(
                    frame_height=frame_height,
                    frame_width=frame_width,
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
                    original_wh=detection_result.original_wh,
                )

                try:
                    self.result_queue.put(result, timeout=self.queue_put_timeout)
                    logger.debug(
                        f"Put danger identification results for frame {detection_result.frame_id} on output queue"
                    )
                except QueueFullException:
                    logger.error(
                        f"Failed to put danger identification result for frame {detection_result.frame_id} on output queue. "
                        "Output queue is full, consumer too slow? "
                        "Discarding frame."
                    )

                logger.debug(f"frame {detection_result.frame_id} processed in {(time() - iter_start) * 1000:.2f} ms")

        except Exception as e:
            logger.critical(f"An unexpected critical error happened in danger identification process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application")

        finally:
            # log process conclusion
            logger.info(
                "Danger identification process terminated successfully."
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()


if __name__ == "__main__":

    import numpy as np
    import random
    from src.shared.processes.consumer import Consumer
    from src.shared.processes.producer import Producer
    from time import sleep, time, perf_counter
    from src.danger_detection.processes.messages import DetectionResult, SegmentationResult, GeoResult

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30
    FREAL = 40

    QUEUE_MAX = 3

    stop_with_poison_pill = True
    stop_after = 20.0


    def generate_queue_object():
        ts=time()
        num_animals = random.randint(0,10)
        boxes_centers = np.array([(random.randint(0,639), random.randint(0,479)) for _ in range(num_animals)])
        boxes_corner1= boxes_centers - 15
        boxes_corner2 = boxes_centers + 15

        d = DetectionResult(
            frame_id=int(ts*100),
            frame=np.zeros((720,1080,3), dtype=np.uint8),
            classes_names=["goat", "sheep"],
            num_classes=2,
            classes=[random.randint(0,1) for _ in range(num_animals)],
            boxes_centers=boxes_centers,
            boxes_corner1=boxes_corner1,
            boxes_corner2=boxes_corner2,
            timestamp=ts,
            original_wh=(1920,1080),
        )
        s= SegmentationResult(
            frame_id=int(ts*100),
            roads_mask=np.random.randint(0,random.choice([1,2]),size=(720,1080), dtype=np.uint8),
            vehicles_mask=np.random.randint(0,random.choice([1,2]),size=(720,1080), dtype=np.uint8),
        )
        g= GeoResult(
            frame_id=int(ts*100),
            safety_radius_pixels=random.randint(10, 100),
            nodata_dem_mask=np.random.randint(0,random.choice([1,2]),size=(720,1080), dtype=np.uint8),
            geofencing_mask=np.random.randint(0,random.choice([1,2]),size=(720,1080), dtype=np.uint8),
            slope_mask=np.random.randint(0,random.choice([1,2]),size=(720,1080), dtype=np.uint8),
        )
        return ModelsAlignmentResult(
            detection_result=d,
            segmentation_result=s,
            geo_result=g,
        )

    queue_in = mp.Queue(maxsize=QUEUE_MAX)
    queue_out = mp.Queue(maxsize=QUEUE_MAX)

    stop_event = mp.Event()
    error_event = mp.Event()

    producer = Producer(queue_in, error_event, generate_queue_object, frequency_hz=FAST)
    
    worker = DangerDetectionWorker(
        input_queue=queue_in,
        result_queue=queue_out,
        error_event=error_event,
    )
    
    consumer = Consumer(queue_out, error_event, frequency_hz=FAST)


    print("CONSUMERS STARTED")
    consumer.start()

    sleep(1)

    print("WORKER STARTED")
    worker.start()

    sleep(1)

    sleep(1)

    print("PRODUCER STARTED")
    producer.start()

    sleep(1)

    start_at = time()
    stop_at = start_at + stop_after

    processes = [producer, worker, consumer]

    signal_set = False

    while True:
        
        if time() > stop_at and not signal_set:
            signal_set = True
            if stop_with_poison_pill:
                print("POISON PILL")
                producer.stop()
            else:
                print("ERROR EVENT")
                error_event.set()

        # Check if everyone has finished their logic
        all_finished = all(p.work_finished.is_set() for p in processes)

        # Check if an error occurred anywhere
        error_occurred = error_event.is_set()

        if all_finished or error_occurred:
            if error_occurred:
                print("[Main] Error detected. Terminating chain.")
            else:
                print("[Main] All processes finished logic. Cleaning up.")
            break

        sleep(0.5)

    print(f"[Main] Granting 5s for all processed to cleanly conclude their processing.")
    sleep(5.0)
    # The Sweep: Force everyone to join or die
    for p in processes:
        # If the logic is finished but the process is still 'alive',
        # it is 100% stuck in the queue feeder thread.
        if p.is_alive():
            print(f"[Main] {p.name} is hanging in cleanup. Work Completed: {p.work_finished.is_set()}. Terminating.")
            p.terminate()

        p.join()
        print(f"[Main] {p.name} joined.")