import subprocess
import threading
import queue
import logging

logger = logging.getLogger("main.video_out")

class RTSPStreamer:
    def __init__(self, rtsp_url, width, height, fps):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=30)
        self.ffmpeg_process = None
        self.writer_thread = None
        self.running = False
        
    def start(self):
        """Start FFmpeg process and writer thread."""
        # Start FFmpeg as subprocess (isolation)
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo', 
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-pix_fmt', 'bgr24',    # depending on input (rgb24)
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-g', str(self.fps),  # GOP size = framerate for 1 second keyframes
            '-keyint_min', str(self.fps),
            '-b:v', '2M',  # Adjust based on your needs
            '-maxrate', '2M',
            '-bufsize', '4M',  # Much larger buffer
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # More reliable than UDP
            self.rtsp_url
        ]

        #cmd = [
        #    'ffmpeg', '-y',
        #    '-f', 'rawvideo',
        #    '-vcodec', 'rawvideo', 
        #    '-s', f'{self.width//2}x{self.height//2}',
        #    '-r', str(self.fps),
        #    '-pix_fmt', 'bgr24',    # depending on input (rgb24)
        #    '-i', '-',
        #    '-vf', 'yadif=1:0:1',
        #    '-c:v', 'libx264',
        #    '-pix_fmt', 'yuv420p',
        #    '-preset', 'medium',
        #    '-tune', 'zerolatency',
        #    '-profile:v', 'high',
        #    '-level', '4.1',
        #    '-g', str(self.fps*2),  # GOP size = framerate for 1 second keyframes
        #    '-keyint_min', str(self.fps),
        #    '-b:v', '8M',  # Adjust based on your needs
        #    '-maxrate', '10M',
        #    '-bufsize', '16M',  # Much larger buffer
        #    '-rc_lookahead', '40',
        #    '-refs', '5',
        #    '-crf', '20',
        #    '-vsync', 'cfr', 
        #    '-fps_mode', 'cfr',
        #    '-x264-params', 'nal-hrd=cbr:force-cfr=1:bframes=2',
        #    '-f', 'rtsp',
        #    '-rtsp_transport', 'tcp',  # More reliable than UDP
        #    self.rtsp_url
        #]
        
        self.ffmpeg_process = subprocess.Popen(
            cmd, 
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Start writer thread
        self.running = True
        self.writer_thread = threading.Thread(target=self._write_frames)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        logger.info("Started RTSP Stream process")
        
    def _write_frames(self):
        """Thread to write frames to FFmpeg process."""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)
                if frame is not None:
                    self.ffmpeg_process.stdin.write(frame.tobytes())
                    self.ffmpeg_process.stdin.flush()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                break
                
    def write_frame(self, frame):
        """Add frame to streaming queue (non-blocking)."""
        if self.running:
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                # Drop frame if queue is full (prevents blocking)
                pass
                
    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.writer_thread:
            self.writer_thread.join(timeout=2.0)
        if self.ffmpeg_process:
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
