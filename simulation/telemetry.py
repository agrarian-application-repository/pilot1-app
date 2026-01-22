import paho.mqtt.client as mqtt
import time
import json
import random

# --- Configuration ---
BROKER = "localhost"
PORT = 1883
FREQUENCY_HZ = 40  # messages every second
#USERNAME = "your_username"  # Must match the user created in mosquitto_passwd
#PASSWORD = "your_password"

# Topics
TOPICS = {
    "lat": "telemetry/latitude",
    "lon": "telemetry/longitude",
    "alt": "telemetry/rel_alt",
    "yaw": "telemetry/gb_yaw"
}

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully to Mosquitto")
    else:
        print(f"Connection failed with code {rc}")

# Initialize Client
client = mqtt.Client()
#client.username_pw_set(USERNAME, PASSWORD)
client.on_connect = on_connect

try:
    client.connect(BROKER, PORT, 60)
except Exception as e:
    print(f"Could not connect to broker: {e}")
    exit(1)

client.loop_start()

print("Starting telemetry stream... Press Ctrl+C to stop.")

try:
    while True:
        # Generate random data
        telemetry_data = {
            TOPICS["lat"]: round(random.uniform(-90, 90), 6),
            TOPICS["lon"]: round(random.uniform(-180, 180), 6),
            TOPICS["alt"]: round(random.uniform(0, 500), 2),
            TOPICS["yaw"]: round(random.uniform(0, 360), 2)
        }

        # Publish each topic
        for topic, value in telemetry_data.items():
            client.publish(topic, value, qos=1)
            print(f"Published: {topic} -> {value}")
            time.sleep(1 / FREQUENCY_HZ)

except KeyboardInterrupt:
    print("\nStopping simulation...")
    client.loop_stop()
    client.disconnect()
