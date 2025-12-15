import multiprocessing as mp
from typing import Any
import time

class Consumer(mp.Process):
    def __init__(self, input_queue: mp.Queue):
        super().__init__()
        self.input_queue = input_queue
        
    def process_data(self, data: Any) -> None:
        """
        Process the consumed data. Override this method for custom processing.
        """
        # print(f"[{self.name}] Processing: {data}")
        # Add your custom processing logic here
        #time.sleep(0.01)  # Simulate processing time
        
    def run(self) -> None:
        """
        Main consumer loop that continuously consumes data from the queue.
        Terminates when None is found on the queue.
        """
        print(f"[{self.name}] Starting consumer process...")
        
        while True:
            try:
                # Get data from queue (blocking call)
                data = self.input_queue.get()
                
                # Terminate if None is encountered
                if data is None:
                    print(f"[{self.name}] Found None, terminating consumer")
                    break
                    
                # Process the data
                self.process_data(data)
                
                # Mark task as done
                # self.input_queue.task_done()
                
            except Exception as e:
                print(f"[{self.name}] Error processing data: {e}")
                continue
                
        print(f"[{self.name}] Consumer terminated")