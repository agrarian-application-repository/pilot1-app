import multiprocessing
from queue import Empty as QueueEmptyException
from queue import Full as QueueFullException
from src.shared.processes.constants import POISON_PILL
import time


class Processor(multiprocessing.Process):
    def __init__(
        self,
        input_queue: multiprocessing.Queue,
        output_queue: multiprocessing.Queue,
        error_event: multiprocessing.Event,
        frequency_hz: float = 50.0,
    ):
        """
        :param input_queue: multiprocessing.Queue to get data from.
        :param output_queue: multiprocessing.Queue to put data into.
        :param error_event: Multiprocessing Event
        :param frequency_hz: How many items to generate per second.
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.interval = 1.0 / frequency_hz
        self.error_event = error_event

        self.work_finished = multiprocessing.Event()


    def run(self):
        print(f"[PROCESSOR] Starting at {1 / self.interval} Hz")

        next_time = time.perf_counter()

        try:
            while not self.error_event.is_set():

                try:
                    data = self.input_queue.get(timeout=self.interval)
                except QueueEmptyException:
                    continue
                
                try:
                    
                    if data == POISON_PILL:
                        self.output_queue.put(data, timeout=1.0)
                        print("[PROCESSOR] Propagated POISON PILL. Terminating")
                        break
                except QueueFullException:
                        print("[PROCESSOR] Unable to propagate POISON PILL. Terminating")
                        self.error_event.set()
                        break

                try:
                    self.output_queue.put(data, timeout=0.01)
                except QueueFullException:
                    continue

                now = time.perf_counter()
                if now < next_time:
                    time.sleep(next_time - now)
                next_time += self.interval

        except Exception as e:
            print(f"[PROCESSOR] Unforseen exception. {e}")

        finally:
            # DRAINING: Remove everything left so Producer doesn't hang on its join_thread()
            #try:
            #    while True:
            #        _ = self.input_queue.get_nowait()
            #except QueueEmptyException:
            #    print("[PROCESSOR] Input Queue successfully drained.")
            #finally:
            #    self.input_queue.close()

            print("[PROCESSOR] DONE")
            self.work_finished.set()
