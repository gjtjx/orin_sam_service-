# Orin SAM Service

A high-performance Segment Anything Model (SAM) inference service optimized for NVIDIA Orin platforms. This service provides a FastAPI-based REST API for running SAM model inference with support for various prompting methods including points and bounding boxes.

## Features

- 🚀 FastAPI-based REST API for easy integration
- 🎯 Support for point and box prompts
- 🔄 Multiple mask output options
- 🐳 Docker support with NVIDIA GPU acceleration
- 🔧 Configurable model selection (vit_h, vit_l, vit_b)
- 📊 Health check and monitoring endpoints
- ⚡ Optimized for NVIDIA Orin edge devices

## Requirements

- NVIDIA Orin device (AGX Orin, Orin NX, Orin Nano)
- JetPack 5.x or later
- Python 3.8+
- CUDA support
- Docker (optional, for containerized deployment)

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/gjtjx/orin_sam_service-.git
cd orin_sam_service-
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download SAM model checkpoint:
```bash
# Create models directory
mkdir -p models

# Download model (example for vit_h)
# Place your SAM model checkpoint in the models directory
# wget -P models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

5. Configure environment:
```bash
cp .env.example .env
# Edit .env file with your settings
```

### Docker Installation

1. Build the Docker image:
```bash
docker build -t orin-sam-service .
```

2. Or use Docker Compose:
```bash
docker-compose up -d
```

## Configuration

Edit the `.env` file or set environment variables:

```bash
HOST=0.0.0.0                    # Service host
PORT=8000                       # Service port
MODEL_TYPE=vit_h                # Model type (vit_h, vit_l, vit_b)
MODEL_CHECKPOINT=sam_vit_h_4b8939.pth  # Path to model checkpoint
DEVICE=cuda                     # Device (cuda or cpu)
```

## Usage

### Starting the Service

**Local:**
```bash
python main.py
```

**Docker:**
```bash
docker-compose up
```

The service will be available at `http://localhost:8000`

### API Endpoints

#### Root Endpoint
```bash
GET /
```
Returns service information and status.

#### Health Check
```bash
GET /health
```
Returns service health status.

#### Segmentation
```bash
POST /segment
```

Segment an image with optional prompts.

**Parameters:**
- `image` (file): Input image file
- `points` (optional, string): JSON string of point prompts
- `box` (optional, string): JSON string of box prompt
- `multimask_output` (optional, bool): Return multiple masks (default: true)

**Example with cURL:**

```bash
# Basic segmentation with point prompt
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test_image.jpg" \
  -F 'points=[{"x": 100, "y": 100, "label": 1}]' \
  -F "multimask_output=true"

# Segmentation with box prompt
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test_image.jpg" \
  -F 'box={"x1": 50, "y1": 50, "x2": 200, "y2": 200}'

# Segmentation with multiple points
curl -X POST "http://localhost:8000/segment" \
  -F "image=@test_image.jpg" \
  -F 'points=[{"x": 100, "y": 100, "label": 1}, {"x": 150, "y": 150, "label": 1}]'
```

**Example with Python:**

```python
import requests

# Prepare the image
files = {"image": open("test_image.jpg", "rb")}

# Point prompts
data = {
    "points": '[{"x": 100, "y": 100, "label": 1}]',
    "multimask_output": True
}

# Make request
response = requests.post("http://localhost:8000/segment", files=files, data=data)
result = response.json()

print(f"Generated {len(result['masks'])} masks")
print(f"Scores: {result['scores']}")
```

#### Reset Model
```bash
POST /reset
```
Reset the model predictor state.

### Response Format

The segmentation endpoint returns:

```json
{
  "masks": [[[true, false, ...], ...], ...],
  "scores": [0.95, 0.90, 0.85],
  "message": "Segmentation completed successfully"
}
```

- `masks`: List of binary segmentation masks (3D array)
- `scores`: Confidence scores for each mask
- `message`: Status message

## API Documentation

Once the service is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Performance Optimization

For optimal performance on Orin devices:

1. Use FP16 precision for faster inference
2. Batch multiple requests when possible
3. Consider using smaller models (vit_b) for edge deployment
4. Enable TensorRT optimization for production use

## Project Structure

```
orin_sam_service-/
├── main.py              # FastAPI application
├── model.py             # SAM model handler
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose setup
├── .env.example        # Environment variables template
├── .gitignore          # Git ignore rules
├── LICENSE             # Apache 2.0 License
└── README.md           # This file
```

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
- Use a smaller model (vit_b instead of vit_h)
- Process smaller images
- Reduce batch size

**2. Model checkpoint not found**
- Ensure the checkpoint path in `.env` is correct
- Download the model checkpoint to the specified location

**3. Docker GPU access issues**
- Ensure NVIDIA Container Runtime is installed
- Check `nvidia-docker` is properly configured

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Segment Anything Model (SAM) by Meta AI
- FastAPI framework
- NVIDIA Jetson platform
