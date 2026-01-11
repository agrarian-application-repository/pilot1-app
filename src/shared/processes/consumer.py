import multiprocessing
import time
from collections import deque
from queue import Empty as QueueEmptyException
from src.shared.processes.constants import POISON_PILL


class Consumer(multiprocessing.Process):
    def __init__(
            self,
            input_queue: multiprocessing.Queue,
            error_event: multiprocessing.Event,
            frequency_hz: float = 50.0,
            length: int = 10,

    ):
        """
        :param input_queue: The multiprocessing.Queue to pull data from.
        :param error_event: Multiprocessing Event
        :param frequency_hz: How many items to process per second (e.g., 30.0 for 30fps).
        :param length: length of the deque to measure processing speed
        """
        super().__init__()
        self.input_queue = input_queue
        self.interval = 1.0 / frequency_hz
        self.length = length
        self.error_event = error_event

        self.work_finished = multiprocessing.Event()

    def run(self):
        print(f"Consumer: Starting at {1 / self.interval} Hz")

        perf_dequeue = deque(maxlen=self.length)

        next_t = time.perf_counter()

        try:

            while not self.error_event.is_set():

                try:
                    # Attempt to get an item without blocking forever
                    # This allows the process to check the stop_signal regularly
                    read_start = time.perf_counter()
                    data = self.input_queue.get(timeout=self.interval)

                    if data == POISON_PILL:
                        print("[CONSUMER] poison pill found, terminating")
                        break
                    
                    # assert data.frame.shape == (720, 1280, 3), f"Unexpected frame shape: {data.frame.shape}"
                    # print(data)

                    read_time = time.perf_counter() - read_start
                    if read_time < self.interval:
                        time.sleep(self.interval - read_time)

                    perf_dequeue.appendleft(time.time())
                    if len(perf_dequeue) == self.length:
                        print(
                            f"Current processing speed over the last {self.length} samples: "
                            f"{self.length/(perf_dequeue[0] - perf_dequeue[-1])} samples/s"
                        )
                except QueueEmptyException:
                    # Queue was empty, just continue
                    continue

                # 2. Precise Timing Logic
                next_t += self.interval
                now = time.perf_counter()
                sleep_time = next_t - now

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We are running behind schedule!
                    # Option A: Don't sleep at all to catch up.
                    # Option B: Reset next_t to now to prevent a "stampede" of fast reads.
                    next_t = now

        except Exception as e:
            print("[CONSUMER]: unexpected exception {e}")

        finally:

            # DRAINING: Remove everything left so Producer doesn't hang on its join_thread()
            #try:
            #    while True:
            #        _ = self.input_queue.get_nowait()
            #except QueueEmptyException:
            #    print("[CONSUMER] Input Queue successfully drained.")
            #finally:
            #    self.input_queue.close()

            print("[CONSUMER] DONE")
            self.work_finished.set()
