from src.configs.utils import read_yaml_config
from src.danger_detection.processes.detection import DetectionWorker
from src.shared.processes.messages import CombinedFrametelemetryQueueObject
import multiprocessing as mp
from time import time
import numpy as np

def main():

    print("started")
    queue_lenght = 200
    detection_in_queue = mp.Queue()
    detection_results_queue = mp.Queue()

    for i in range(queue_lenght):
        detection_in_queue.put(
            CombinedFrametelemetryQueueObject(
                frame_id=i,
                frame=np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8),
                telemetry=None,
                timestamp=time()

            )
        )

    detection_in_queue.put(None)
    
    detection_args = read_yaml_config("configs/danger_detection/detector.yaml")

    print("Benchmarking")
    # Create DetectionWorker process
    detection_process = DetectionWorker(
        detection_args=detection_args,
        input_queue=detection_in_queue,
        result_queue=detection_results_queue
    )

    detection_process.start()

    detection_process.join()

    detection_in_queue.close()
    print("in queue closed")

    detection_results_queue.close()
    print("out queue closed")


if __name__ == "__main__":
    main()