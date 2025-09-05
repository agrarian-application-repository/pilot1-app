from src.configs.utils import read_yaml_config
from src.danger_detection.processes.segmentation import SegmentationWorker
from src.shared.processes.messages import CombinedFrametelemetryQueueObject
import multiprocessing as mp
from time import time
import numpy as np
import torch
print(torch.backends.cudnn.version())

def main():

    print("started")
    queue_lenght = 200
    segmentation_in_queue = mp.Queue()
    segmentation_results_queue = mp.Queue()

    for i in range(queue_lenght):
        segmentation_in_queue.put(
            CombinedFrametelemetryQueueObject(
                frame_id=i,
                frame=np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8),
                telemetry=None,
                timestamp=time()

            )
        )

    segmentation_in_queue.put(None)

    detection_args = read_yaml_config("configs/danger_detection/segmenter.yaml")

    print("Benchmarking")
    # Create SegmentationWorker process
    segmentation_process = SegmentationWorker(
        segmentation_args=detection_args,
        input_queue=segmentation_in_queue,
        result_queue=segmentation_results_queue
    )

    segmentation_process.start()

    segmentation_process.join()

    segmentation_in_queue.close()
    print("in queue closed")

    segmentation_results_queue.close()
    print("out queue closed")


if __name__ == "__main__":
    main()



# export LD_LIBRARY_PATH=/home/simone/miniconda3/envs/agrarian/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9:$LD_LIBRARY_PATH