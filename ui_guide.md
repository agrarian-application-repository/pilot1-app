# Streamlit UI Container Launch Guide

This guide explains how to use the `launch-ui.sh` script to deploy your Streamlit UI container that receives video streams via WebRTC from MediaMTX and message packs via TCP.

## Prerequisites

- Docker installed and running
- The Streamlit UI Docker image built or available
- `launch-ui.sh` script with execute permissions

```bash
chmod +x launch-ui.sh
```

## Quick Start

### Basic Launch
The simplest way to start your container with default settings:

```bash
./launch-ui.sh
```

This launches the container with:
- TCP port: 54321 (host) → 54321 (container)
- Streamlit port: 8501 (host) → 8501 (container)
- MediaMTX URL: `http://mediamtx:8889`
- Stream name: `annot`
- Default STUN server

### Development Mode
For development with automatic rebuilding:

```bash
./launch-ui.sh -b -r --stream-name dev-camera
```

This will:
- Build the Docker image first (`-b`)
- Remove any existing container (`-r`)
- Use a custom stream name

### Production Mode
For production deployment with detached mode:

```bash
./launch-ui.sh -d -r -f production.env --network production-network
```

## Command Line Options

### Basic Options
| Option | Description | Default |
|--------|-------------|---------|
| `-i, --image NAME` | Docker image name | `streamlit-ui` |
| `-n, --name NAME` | Container name | `streamlit-ui` |
| `-h, --help` | Show help message | - |

### Port Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `-p, --tcp-port PORT` | Host TCP port mapping | `54321` |
| `-s, --streamlit-port PORT` | Host Streamlit port mapping | `8501` |
| `--internal-tcp-port PORT` | Internal container TCP port | `54321` |

### Service Configuration
| Option | Description | Default |
|--------|-------------|---------|
| `-m, --mediamtx-url URL` | MediaMTX WebRTC URL | `http://mediamtx:8889` |
| `--stream-name NAME` | Video stream name | `annot` |
| `--stun-server SERVER` | STUN server for WebRTC | `stun:stun.l.google.com:19302` |

### Runtime Options
| Option | Description | Default |
|--------|-------------|---------|
| `-d, --detached` | Run in background (detached mode) | Interactive |
| `-r, --remove` | Remove existing container first | Keep existing |
| `-b, --build` | Build image before running | Use existing |
| `-f, --env-file FILE` | Load environment variables from file | None |
| `--network NETWORK` | Connect to specific Docker network | Default network |

## Usage Examples

### Single Camera Setup
```bash
./launch-ui.sh --stream-name front-door -p 54321
```

### Multiple Camera Setup
For multiple camera streams, run separate containers:

```bash
# Camera 1
./launch-ui.sh -n ui-camera1 --stream-name camera1 -p 54321 -s 8501

# Camera 2  
./launch-ui.sh -n ui-camera2 --stream-name camera2 -p 54322 -s 8502

# Camera 3
./launch-ui.sh -n ui-camera3 --stream-name camera3 -p 54323 -s 8503
```

### Custom MediaMTX Server
If your MediaMTX server is running on a different host:

```bash
./launch-ui.sh -m "http://192.168.1.100:8889" --stream-name outdoor-cam
```

### Development with Custom Network
```bash
./launch-ui.sh -b -r --network dev-network --stream-name test-stream
```

## Environment File Configuration

### Creating Environment Files
Create different `.env` files for different deployment scenarios:

**production.env:**
```bash
TCP_PORT=54321
MEDIAMTX_WEBRTC_URL=http://production-mediamtx:8889
STREAM_NAME=main-camera
STUN_SERVER=stun:production-stun.example.com:19302
```

**development.env:**
```bash
TCP_PORT=54321
MEDIAMTX_WEBRTC_URL=http://dev-mediamtx:8889
STREAM_NAME=dev-camera
STUN_SERVER=stun:stun.l.google.com:19302
```

**staging.env:**
```bash
TCP_PORT=54321
MEDIAMTX_WEBRTC_URL=http://staging-mediamtx:8889
STREAM_NAME=staging-camera
STUN_SERVER=stun:staging-stun.example.com:19302
```

### Using Environment Files
```bash
# Production deployment
./launch-ui.sh -d -r -f production.env --network production

# Development
./launch-ui.sh -f development.env -b -r

# Staging
./launch-ui.sh -d -f staging.env --network staging
```

## Docker Compose Integration

The launch script works alongside Docker Compose. If you're using Docker Compose for MediaMTX:

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  mediamtx:
    image: bluenviron/mediamtx:latest
    ports:
      - "8889:8889"
      - "1935:1935"
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

**Launch UI with Compose network:**
```bash
# Start MediaMTX
docker-compose up -d mediamtx

# Launch UI connected to same network
./launch-ui.sh -d --network app-network
```

## Common Scenarios

### Scenario 1: First Time Setup
```bash
# Build and run for the first time
./launch-ui.sh -b --stream-name my-camera
```

### Scenario 2: Update and Restart
```bash
# Rebuild image and restart container
./launch-ui.sh -b -r
```

### Scenario 3: Port Conflict Resolution
```bash
# Use different ports if defaults are occupied
./launch-ui.sh -p 55555 -s 8502
```

### Scenario 4: Remote MediaMTX Server
```bash
# Connect to MediaMTX running on different machine
./launch-ui.sh -m "http://camera-server.local:8889" --stream-name main
```

### Scenario 5: Production Deployment
```bash
# Production deployment with environment file
./launch-ui.sh -d -r -f production.env --network production
```

## Troubleshooting

### Container Already Exists
If you see "Container already running":
```bash
./launch-ui.sh -r  # Remove and recreate
```

### Port Already in Use
Change the host port mapping:
```bash
./launch-ui.sh -p 55555 -s 8502
```

### MediaMTX Connection Issues
Check MediaMTX URL and ensure containers can communicate:
```bash
# Test with explicit MediaMTX URL
./launch-ui.sh -m "http://mediamtx:8889" --network your-network
```

### Environment File Not Found
Ensure the environment file exists and path is correct:
```bash
ls -la production.env  # Check file exists
./launch-ui.sh -f ./production.env  # Use explicit path
```

## Accessing Your Application

After successful launch:

- **Streamlit UI**: `http://localhost:8501` (or custom port)
- **TCP Messages**: Port `54321` (or custom port)
- **Container Logs**: `docker logs streamlit-ui`
- **Container Shell**: `docker exec -it streamlit-ui bash`

## Advanced Usage

### Custom Image Names
```bash
./launch-ui.sh -i my-custom-ui:v1.0 -n custom-ui-container
```

### Multiple Environment Variables
```bash
./launch-ui.sh \
  --stream-name production-cam \
  -m "http://prod-mediamtx:8889" \
  --stun-server "stun:prod-stun.example.com:19302" \
  -d -r
```

### Integration with CI/CD
```bash
# In your deployment pipeline
./launch-ui.sh -d -r -f $ENVIRONMENT.env --network $NETWORK_NAME
```

This launch script provides a flexible, production-ready way to deploy your Streamlit UI container across different environments and configurations.