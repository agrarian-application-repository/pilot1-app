import streamlit as st
import threading
import msgpack
import time
from PIL import Image
import numpy as np
import queue
import struct
import logging
import socket
import datetime
from constants import HOST, RECONNECT_DELAY, CONTAINER
# ================================================================

logger = logging.getLogger("ui.receiver")
logfile_name = "receiver.log"
log_path = f"/app/logs/{logfile_name}" if CONTAINER else f"../../logs/{logfile_name}"

if not logger.handlers:  # Avoid duplicate handlers
    alert_handler = logging.FileHandler(log_path, mode='w')
    alert_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(alert_handler)
    logger.setLevel(logging.DEBUG)

# ================================================================


# Utility function to check if port is available
def is_port_available(port: int):
    """Check if a port is available for binding"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, port))
            return True
    except socket.error:
        return False


class TCPAlertReceiver:
    """TCP receiver with auto-reconnection"""
    def __init__(self, port, alert_queue):
        self.port = port
        self.alert_queue = alert_queue
        self.running = False
        self.server_socket = None
        self.clients = []
        self.clients_lock = threading.Lock()
        self.restart_lock = threading.Lock()  # Prevent multiple restart attempts
        
    def start(self):
        self.running = True
        thread = threading.Thread(target=self._server_with_reconnect, name="Alerts-TCP-Server")
        thread.daemon = True
        thread.start()
        return thread
    
    def _server_with_reconnect(self):
        """Server with automatic restart on failure"""
        while self.running:
            try:
                self._run_server()
            except Exception as e:
                logger.error(f"TCP server failed: {e}")
                if self.running:
                    # Ensure proper cleanup before restart
                    self._cleanup_server()
                    logger.info(f"Restarting TCP server in {RECONNECT_DELAY} seconds...")
                    time.sleep(RECONNECT_DELAY)
    
    def _cleanup_server(self):
        """Clean up server resources"""
        # Close all client connections
        with self.clients_lock:
            for client_socket, _, addr in self.clients:
                try:
                    client_socket.close()
                    logger.debug(f"Closed client connection {addr}")
                except:
                    pass
            self.clients.clear()
            if 'connected_clients' in st.session_state:
                st.session_state.connected_clients = 0
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
                logger.info(f"TCP server socket on port {self.port} closed.")
            except Exception as close_e:
                logger.error(f"Error closing server socket: {close_e}")
            finally:
                self.server_socket = None
        
        # Give the OS time to release the port
        time.sleep(0.5)
    
    def _run_server(self):
        """Run the TCP server"""
        # Check if port is available before attempting to bind
        if not is_port_available(self.port):
            logger.warning(f"Port {self.port} appears to be in use, will attempt to bind anyway...")
        
        # Create socket with improved options
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Enable SO_REUSEPORT if available (Linux/macOS)
        try:
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except (AttributeError, OSError):
            pass  # SO_REUSEPORT not available on this platform
        
        try:
            self.server_socket.bind((HOST, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)

            logger.info(f"TCP server successfully started on {HOST}:{self.port}")
        
            while self.running:
                try:
                    client_socket, addr = self.server_socket.accept()
                    client_thread = threading.Thread(
                        target=self._handle_client_with_reconnect,
                        args=(client_socket, addr),
                        name=f"TCP-Client-{addr[0]}:{addr[1]}"
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                    with self.clients_lock:
                        self.clients.append((client_socket, client_thread, addr))
                        if 'connected_clients' in st.session_state:
                            st.session_state.connected_clients = len(self.clients)
                    
                    logger.info(f"New client connected from {addr}")
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"TCP accept error: {e}")
                    break
        
        except socket.error as e:
            if e.errno == 98:  # Address already in use
                logger.error(f"Port {self.port} is already in use. This suggests the previous server instance didn't clean up properly.")
                # Log what might be using the port
                try:
                    import subprocess
                    result = subprocess.run(['netstat', '-tulpn'], capture_output=True, text=True, timeout=5)
                    for line in result.stdout.split('\n'):
                        if f':{self.port}' in line:
                            logger.error(f"Port usage: {line.strip()}")
                except:
                    pass
                raise
            else:
                logger.error(f"Socket error: {e}")
                raise
        
        finally:
            # Cleanup is handled by _cleanup_server method
            pass
    
    def _handle_client_with_reconnect(self, client_socket, addr):
        """Handle client with error recovery"""
        try:
            # Set socket options for better reliability
            client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            while self.running:
                try:
                    # Read message length
                    length_data = self._recv_exact(client_socket, 4)
                    if not length_data:
                        break
                    
                    message_length = struct.unpack('!I', length_data)[0]
                    
                    if message_length > 50 * 1024 * 1024:  # 50MB limit
                        logger.error(f"Message too large: {message_length} bytes from {addr}")
                        break
                    
                    # Read message
                    message_data = self._recv_exact(client_socket, message_length)
                    if not message_data:
                        break
                    
                    # Process alert
                    alert_data = msgpack.unpackb(message_data, raw=False)
                    self._process_alert(alert_data)
                    
                    if 'total_alerts' in st.session_state:
                        st.session_state.total_alerts += 1
                        
                except Exception as e:
                    logger.error(f"Error handling client {addr}: {e}")
                    break
                    
        finally:
            try:
                client_socket.close()
            except:
                pass
            
            # Remove client from list
            with self.clients_lock:
                self.clients = [(sock, thread, a) for sock, thread, a in self.clients if a != addr]
                if 'connected_clients' in st.session_state:
                    st.session_state.connected_clients = len(self.clients)
            
            logger.info(f"Client {addr} disconnected")
    
    def _recv_exact(self, sock, n):
        """Receive exactly n bytes from socket"""
        data = b''
        while len(data) < n:
            try:
                sock.settimeout(5.0)
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data += packet
            except socket.timeout:
                if not self.running:
                    return None
                continue
            except socket.error:
                return None
        return data
    
    def _process_alert(self, alert_data):
        """Process alert data"""
        try:
            frame_id = alert_data.get('frame_id', 'Unknown')
            alert_str = alert_data.get('alert', 'Unknown')
            timestamp = alert_data.get('timestamp', time.time())    # use current time as default
            img_shape = alert_data.get('img_shape', [])
            img_dtype = alert_data.get('img_dtype', 'uint8')
            img_format = alert_data.get('img_format', None)
            compressed = alert_data.get('compressed', False)
            img_bytes = alert_data.get('img_bytes', b'')

            format_pattern = "%Y-%m-%d %H:%M:%S"
            timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(format_pattern)
            
            if img_bytes:
                image = self._reconstruct_image(
                    img_bytes, img_shape, img_dtype, img_format, compressed
                )
                
                if image is not None:
                    msg = {
                        'frame_id': frame_id,
                        'alert': alert_str,
                        'timestamp': timestamp,
                        'image': image,
                        'processed_time': time.time()
                    }
                    try:
                        self.alert_queue.put(msg, timeout=1.0)
                        logger.info(f"Alert processed (Frame {frame_id}): {alert_str}")
                    except queue.Full:
                        logger.warning("Alert queue full, dropping alert")

                else:
                    logger.warning("Received alert with empty image")
                        
        except Exception as e:
            logger.error(f"Error processing alert: {e}")
    
    def _reconstruct_image(self, img_bytes, img_shape, img_dtype, img_format, compressed):
        """Reconstruct image from bytes"""
        try:
            if compressed and img_format:
                import cv2
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                image_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if image_cv is not None:
                    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    return Image.fromarray(image_rgb)
            else:
                dtype = np.dtype(img_dtype)
                img_array = np.frombuffer(img_bytes, dtype=dtype)
                
                if img_shape:
                    img_array = img_array.reshape(img_shape)
                    
                    if len(img_shape) == 3:
                        if img_shape[2] == 3:
                            return Image.fromarray(img_array.astype(np.uint8))
                        elif img_shape[2] == 4:
                            return Image.fromarray(img_array.astype(np.uint8), 'RGBA')
                    else:
                        return Image.fromarray(img_array.astype(np.uint8), 'L')
                        
        except Exception as e:
            logger.error(f"Error reconstructing image: {e}")
            
        return None
    
    def stop(self):
        """Stop the TCP server and cleanup resources"""
        logger.info("Stopping TCP server...")
        self.running = False
        self._cleanup_server()
        logger.info("TCP server stopped")