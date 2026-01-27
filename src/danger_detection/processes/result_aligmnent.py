import multiprocessing as mp
import logging
from src.danger_detection.processes.messages import DetectionResult, SegmentationResult, \
    GeoResult, ModelsAlignmentResult
from src.shared.processes.constants import *
from time import time, sleep
from queue import Full as QueueFullException
from queue import Empty as QueueEmptyException

# ================================================================

logger = logging.getLogger("main.models_alignment")

if not logger.handlers:  # Avoid duplicate handlers
    video_handler = logging.FileHandler('./logs/models_alignment.log', mode='w')
    video_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(video_handler)
    logger.setLevel(logging.DEBUG)


# ================================================================


class ModelsAlignmentWorker(mp.Process):

    def __init__(
            self,
            input_queues: list[mp.Queue],
            result_queue: mp.Queue,
            error_event: mp.Event,
            queue_get_timeout: float = MODELS_QUEUE_GET_TIMEOUT,
            queue_put_timeout: float = MODELS_QUEUE_PUT_TIMEOUT,
            poison_pill_timeout: float = POISON_PILL_TIMEOUT,
    ):
        super().__init__()

        self.input_queues = input_queues
        self.result_queue = result_queue
        self.error_event = error_event

        self.queue_get_timeout = queue_get_timeout
        self.queue_put_timeout = queue_put_timeout
        self.poison_pill_timeout = poison_pill_timeout

        self.work_finished = mp.Event()

    def run(self):

        logger.info("Danger detection process started.")
        poison_pill_received = False

        try:

            while not self.error_event.is_set():

                iter_start = time()

                # do not collect results until all queues have at least one entry
                # short sleep in between checks
                any_empty = any(in_queue.empty() for in_queue in self.input_queues)
                if any_empty:
                    logger.debug(f"At least one input queue empty. Retrying in {self.queue_get_timeout} seconds")
                    sleep(self.queue_get_timeout)
                    continue

                # Collect one result from each model's result queue
                detection_result: DetectionResult = self.input_queues[0].get_nowait()
                segmentation_result: SegmentationResult = self.input_queues[1].get_nowait()
                geo_result: GeoResult = self.input_queues[2].get_nowait()

                collected_results = [detection_result, segmentation_result, geo_result]
                alignment_success = False

                while not alignment_success:

                    # check whether any of the results is a poison pill
                    # if it is, propagate it and leave the loop
                    if any((isinstance(r,str) and r == POISON_PILL) for r in collected_results):
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

                    ids = [r.frame_id for r in collected_results]
                    max_id = max(ids)

                    # Check if alignment is achieved
                    if all(id_val == max_id for id_val in ids):
                        alignment_success = True
                        # All match! Exit sub-loop to process data
                    else:
                        # get the queue that is the most behind
                        min_id = min(ids)
                        queue_idx = ids.index(min_id)

                        logger.info(
                            f"Current ids: {ids}. "
                            f"For queue {queue_idx}, "
                            f"attempting to replace frame {collected_results[queue_idx].frame_id} with a new one "
                            f"to catch up to {max_id}"
                        )
                        try:
                            collected_results[queue_idx] = self.input_queues[queue_idx].get(timeout=self.queue_get_timeout)
                            logger.debug(f"Frame replaced with frame {collected_results[queue_idx].frame_id}")
                        except QueueEmptyException:
                            logger.debug(f"Queue exhausted while trying to catch up to {max_id}")

                # if previous loop was exited due to poison pill (exited due to break before alignment flag set)
                # leave the outer loop as well and terminate the process
                if not alignment_success:
                    break

                aligned_results = ModelsAlignmentResult(
                    detection_result=collected_results[0],
                    segmentation_result=collected_results[1],
                    geo_result=collected_results[2],
                )

                try:
                    self.result_queue.put(aligned_results, timeout=self.queue_put_timeout)
                    logger.debug(f"frame {detection_result.frame_id} processed in {(time() - iter_start) * 1000:.2f} ms")
                except QueueFullException:
                    logger.warning(
                        "Failed to put aligned model result on output queue. "
                        "Consumer process too slow? "
                        "Discarding aligned frames. "
                    )

        except Exception as e:
            logger.critical(f"Unforeseen critical error in alignment process: {e}")
            self.error_event.set()
            logger.warning("Error event set: force-stopping the application ")

        finally:
            # log process conclusion
            logger.info(
                "Model results alignment process terminated successfully."
                f"Poison pill received: {poison_pill_received}. "
                f"Error event: {self.error_event.is_set()}."
            )
            self.work_finished.set()



if __name__ == "__main__":

    import numpy as np
    import random
    from src.shared.processes.consumer import Consumer
    from src.shared.processes.producer import MultiQueueProducer
    from src.shared.processes.processor import Processor
    from time import sleep, time, perf_counter
    from dataclasses import dataclass

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30
    FREAL = 40

    QUEUE_MAX = 3

    stop_with_poison_pill = False
    stop_after = 20.0

    @dataclass
    class ExampleDataclass:
        frame_id: int

    def generate_queue_object():
        return ExampleDataclass(
            frame_id=int(time()*100),
        )

    detection_queue_in = mp.Queue(maxsize=QUEUE_MAX)
    segmentation_queue_in = mp.Queue(maxsize=QUEUE_MAX)
    geo_queue_in = mp.Queue(maxsize=QUEUE_MAX)

    detection_queue_out = mp.Queue(maxsize=QUEUE_MAX)
    segmentation_queue_out = mp.Queue(maxsize=QUEUE_MAX)
    geo_queue_out = mp.Queue(maxsize=QUEUE_MAX)

    models_in_queues = [detection_queue_in, segmentation_queue_in, geo_queue_in]
    models_out_queues = [detection_queue_out, segmentation_queue_out, geo_queue_out]

    out_queue = mp.Queue(maxsize=QUEUE_MAX)

    stop_event = mp.Event()
    error_event = mp.Event()

    producer = MultiQueueProducer(models_in_queues, error_event, generate_queue_object, frequency_hz=FAST)
    
    detection_model = Processor(models_in_queues[0], models_out_queues[0], error_event, frequency_hz=FREAL)
    segmentation_model = Processor(models_in_queues[1], models_out_queues[1], error_event, frequency_hz=SLOW)
    geo_model = Processor(models_in_queues[2], models_out_queues[2], error_event, frequency_hz=REAL)

    aligner = ModelsAlignmentWorker(models_out_queues, out_queue, error_event)

    consumer = Consumer(out_queue, error_event, frequency_hz=FAST)


    print("CONSUMERS STARTED")
    consumer.start()

    sleep(1)

    print("ALIGNER STARTED")
    aligner.start()

    sleep(1)

    print("MODELS STARTED")
    detection_model.start()
    segmentation_model.start()
    geo_model.start()

    sleep(1)

    print("PRODUCER STARTED")
    producer.start()

    sleep(1)

    start_at = time()
    stop_at = start_at + stop_after

    processes = [detection_model, segmentation_model, geo_model] + [aligner] + [consumer]

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