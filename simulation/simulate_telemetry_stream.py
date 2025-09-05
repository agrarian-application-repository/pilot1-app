import socket
import json
import time
import random
import logging
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./telemetry.log', mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():

    # Set up argument parser
    parser = ArgumentParser(description="Send fake telemetry data via UDP.")

    # Add command-line arguments with default values
    parser.add_argument(
        '--ip', 
        type=str, 
        default="10.91.222.62", 
        help="IP address to send telemetry data to."
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=12345, 
        help="Port to send telemetry data to."
    )
    parser.add_argument(
        '--freq', 
        type=int, 
        default=50, 
        help="Update frequency in Hz (updates per second)."
    )

    # Parse arguments from the command line
    args = parser.parse_args()

    # Assign parsed arguments to variables
    telemetry_host = args.ip
    telemetry_port = args.port
    update_frequency = args.freq
    
    # Create a UDP socket, exit if creation fails
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        logger.info(f"UDP socket created successfully.")
    except socket.error as e:
        logger.error(f"Failed to create UDP socket: {e}")
        return

    logging.info(f"Sending telemetry data to {telemetry_host}:{telemetry_port}")

    counter = 0

    while True:
        counter +=1

        # Generate fake telemetry data
        telemetry_data = {
            "iso": 100,
            "shutter": 0.001724405511200014,
            "fnum": 2.8,
            "ev": 0,
            "color_md": "default",
            "ae_meter_md": 1,
            "focal_len": 24,
            "dzoom_ratio": 1,
            "latitude": round(35.427075 + random.uniform(-0.00001, 0.00001), 6),
            "longitude": round(24.174019 + random.uniform(-0.00001, 0.00001), 6),
            "rel_alt": round(27.531 + random.uniform(-5.0, 5.0), 3),
            "abs_alt": 180.975,
            "gb_yaw": round(-35.4 + random.uniform(-2.0, 2.0), 2),
            "gb_pitch": -90,
            "gb_roll": 0,
        }

        # Convert to JSON
        try:
            message = json.dumps(telemetry_data).encode("utf-8")
        except TypeError as e:
            logger.error(f"Failed to serialize telemetry data to JSON: {e}")
            time.sleep(1 / update_frequency) # to prevent a tight loop on error
            continue # Skip to next iteration

        # Send via UDP
        try:
            sock.sendto(message, (telemetry_host, telemetry_port))
            logger.debug(f"Sent: {telemetry_data}")
            if counter % (update_frequency *2) == 0:
                logger.info(f"Sent {counter} packets. Last one: {telemetry_data}")
                counter = 0
        except socket.error as e:
            logger.error(f"Failed to send UDP message to {telemetry_host}:{telemetry_port}: {e}")

        # Wait for next update
        time.sleep(1 / update_frequency)

if __name__ == "__main__":
    main()
