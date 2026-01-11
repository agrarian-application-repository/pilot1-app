import multiprocessing
from queue import Full as QueueFullException
from src.shared.processes.constants import POISON_PILL
import time


class Producer(multiprocessing.Process):
    def __init__(
        self,
        output_queue: multiprocessing.Queue,
        error_event: multiprocessing.Event,
        data_fn,
        frequency_hz: float = 50.0,
    ):
        """
        :param output_queue: multiprocessing.Queue to put data into.
        :param data_fn: Callable that generates the data to enqueue.
        :param error_event: Multiprocessing Event
        :param frequency_hz: How many items to generate per second.
        """
        super().__init__()
        self.output_queue = output_queue
        self.data_fn = data_fn
        self.interval = 1.0 / frequency_hz
        self.stop_signal = multiprocessing.Event()
        self.error_event = error_event
        self.stop_with_poison = True

        self.work_finished = multiprocessing.Event()

    def run(self):
        print(f"[Producer] Starting at {1 / self.interval} Hz")

        next_time = time.perf_counter()

        try:

            while not self.error_event.is_set():

                now = time.perf_counter()
                if now < next_time:
                    time.sleep(next_time - now)
                next_time += self.interval

                if self.stop_signal.is_set():
                    if self.stop_with_poison:
                        try:
                            self.output_queue.put(POISON_PILL, timeout=0.01)
                            print("[Producer] successfully put POISON PILL in queue")
                        except QueueFullException:
                            print("[Producer] Failed to put POISON PILL in queue, set error event")
                            self.error_event.set()
                    # always break on stop signal
                    break

                # Generate data
                data = self.data_fn()

                try:
                    self.output_queue.put(data, timeout=0.01)
                except QueueFullException:
                    pass

        except Exception as e:
            print(f"[Producer] Unforseen exception. {e}")

        finally:
            print("[Producer] DONE")
            self.work_finished.set()

    def stop(self):
        self.stop_signal.set()
