import cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm

def create_timestamp_video(output_filename="output.mp4", duration_sec=60, fps=30):
    # 1. Video Properties
    width, height = 1920, 1080
    total_frames = duration_sec * fps
    
    # 2. Initialize Video Writer
    # 'mp4v' is widely supported. Use 'avc1' for better web compatibility if needed.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # 3. Pre-define Font Settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (255, 255, 255)  # White
    thickness = 3
    
    print(f"Generating {duration_sec}s video ({total_frames} frames)...")
    
    # 4. Generate Frames
    for _ in tqdm(range(total_frames)):
        # Create a fresh black frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get current time string
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # Calculate text position (Centered)
        text_size = cv2.getTextSize(timestamp, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Draw the text
        cv2.putText(frame, timestamp, (text_x, text_y), font, 
                    font_scale, color, thickness, cv2.LINE_AA)
        
        # Write frame to video
        out.write(frame)
    
    # Release resources
    out.release()
    print(f"\nDone! Video saved as {output_filename}")

if __name__ == "__main__":
    create_timestamp_video()