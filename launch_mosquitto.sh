#!/bin/bash

# Create an empty password file (if it doesn't exist)
# Since allow_anonymous is set to false, you won't be able to connect until you create a user. 
# Run this command while the container is active:
# docker exec mosquitto mosquitto_passwd -b /mosquitto/passwd_file <your_username>

docker rm -f mosquitto || true

echo "Starting Mosquitto Docker container..."

echo "Note: Ensure your certificates (ca.crt, mosquitto.crt, mosquitto.key) are in certificates/mosquitto/"

# Launch the container
docker run -d \
  --name mosquitto \
  -p 1883:1883 \
  -v "$(pwd)/configs/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf" \
  -v "$(pwd)/data/mosquitto:/mosquitto/data" \
  -v "$(pwd)/logs/mosquitto:/mosquitto/log" \
  eclipse-mosquitto:latest

# Launch the container
#docker run -d \
#  --name mosquitto \
#  -p 1883:1883 \
#  -p 8883:8883 \
#  -v "$(pwd)/configs/mosquitto/mosquitto.conf:/mosquitto/config/mosquitto.conf" \
#  -v "$(pwd)/certificates/mosquitto/ca.crt:/mosquitto/config/ca.crt" \
#  -v "$(pwd)/certificates/mosquitto/mosquitto.crt:/mosquitto/config/mosquitto.crt" \
#  -v "$(pwd)/certificates/mosquitto/mosquitto.key:/mosquitto/config/mosquitto.key" \
#  -v "$(pwd)/certificates/mosquitto/passwd_file:/mosquitto/passwd_file" \
#  -v "$(pwd)/data/mosquitto:/mosquitto/data" \
#  -v "$(pwd)/logs/mosquitto:/mosquitto/log" \
#  eclipse-mosquitto:latest

if [ $? -eq 0 ]; then
    echo "Once the script is running, you can "listen" to the traffic inside the Docker container to verify the messages are arriving:"
    echo "docker exec -it mosquitto mosquitto_sub [-u your_username] [-P your_password] -t "telemetry/#" -v"
    echo "docker exec -it mosquitto mosquitto_sub -t "telemetry/#" -v"
else
    echo "Failed to start Mosquitto container."
fi
