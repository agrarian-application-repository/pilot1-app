from src.configs.utils import read_yaml_config
from src.shared.processes.stream_video_reader import StreamVideoReader
from src.shared.processes.messages import FrameQueueObject
from src.shared.processes.consumer import Consumer
import multiprocessing as mp
from time import time
import numpy as np
import torch
import os

def main():

    data_queue = mp.Queue()

    video_info_dict = {
        "frame_width": 1280,
        "frame_height": 720,
        "fps": 30,
    }

    urls = {
        "stream_ip": os.environ.get("STREAM_IP", "127.0.0.1"),  
        "stream_port": int(os.environ.get("STREAM_PORT", "1935")),
        "stream_name": os.environ.get("STREAM_NAME", "drone"),
    }

    source = f"rtmp://{urls['stream_ip']}:{urls['stream_port']}/{urls['stream_name']}"



    # Create and start the consumer process
    consumer = Consumer(data_queue)
    consumer.start()

    # Create StreamVideoReader process
    video_reader_process = StreamVideoReader(
        video_info_dict=video_info_dict,
        source=source,
        frame_queue=data_queue,
    )
    video_reader_process.start()



if __name__ == "__main__":
    main()



# export LD_LIBRARY_PATH=/home/simone/miniconda3/envs/agrarian/lib/python3.10/site-packages/nvidia/cudnn/lib/libcudnn.so.9:$LD_LIBRARY_PATH