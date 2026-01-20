from src.shared.processes.producer import Producer
from src.shared.processes.processor import Processor
from src.shared.processes.consumer import Consumer
from src.configs.utils import read_yaml_config
from time import time, sleep
import numpy as np
from src.shared.processes.messages import CombinedFrameTelemetryQueueObject
import multiprocessing as mp

if __name__ == "__main__":

    VSLOW = 1
    SLOW = 10
    FAST = 50
    REAL = 30

    CONSUMER_QUEUE_MAX = 10

    def generate_frame_telemetry_queue_object():
        ts = time()
        return CombinedFrameTelemetryQueueObject(
            frame_id=int(ts*100),
            frame=np.random.randint(0, 256, size=(720, 1280, 3), dtype=np.uint8),
            telemetry=None,
            timestamp=ts,
            original_wh=(1920, 1080),
        )


    stop_event = mp.Event()
    error_event = mp.Event()

    in_queue = mp.Queue()
    out_queue = mp.Queue(maxsize=CONSUMER_QUEUE_MAX)

    producer = Producer(in_queue, error_event, generate_frame_telemetry_queue_object, frequency_hz=SLOW)
    processor = Processor(in_queue, out_queue, error_event, frequency_hz=REAL)
    consumer = Consumer(out_queue, error_event, frequency_hz=SLOW)

    print("CONSUMERS STARTED")
    consumer.start()

    sleep(2)

    print("PROCESSOR STARTED")
    processor.start()

    sleep(2)

    print("PRODUCER STARTED")
    producer.start()

    """
    sleep(5)

    #print("PRODUCER STOPPED")
    #producer.stop()
    print("ERROR EVENT SET")
    error_event.set()

    sleep(5)

    producer.join()
    print("producer joined")

    producer.join()
    print("detector joined")

    consumer.join()
    print("consumer joined")

    """
    event_set = False
    start_time = time()

    processes = [producer, processor, consumer]

    while True:

        if time()-start_time > 5.0 and not event_set:
            event_set=True
            #print("PRODUCER STOPPED")
            #producer.stop()
            print("ERROR EVENT SET")
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
