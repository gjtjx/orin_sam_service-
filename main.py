"""
FastAPI service for SAM (Segment Anything Model) inference on NVIDIA Orin.
"""
import io
import logging
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import settings
from model import SAMPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model instance
predictor: Optional[SAMPredictor] = None


# Pydantic models for request/response
class PointPrompt(BaseModel):
    """Point prompt for segmentation."""
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    label: int = Field(..., description="Label: 1 for foreground, 0 for background")


class BoxPrompt(BaseModel):
    """Box prompt for segmentation."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")


class SegmentationRequest(BaseModel):
    """Request model for segmentation."""
    points: Optional[List[PointPrompt]] = None
    box: Optional[BoxPrompt] = None
    multimask_output: bool = True


class SegmentationResponse(BaseModel):
    """Response model for segmentation."""
    masks: List[List[List[bool]]] = Field(..., description="Segmentation masks")
    scores: List[float] = Field(..., description="Confidence scores for each mask")
    message: str = "Segmentation completed successfully"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global predictor
    logger.info("Starting SAM service...")
    logger.info(f"Configuration: model_type={settings.model_type}, device={settings.device}")
    
    predictor = SAMPredictor(
        model_type=settings.model_type,
        checkpoint=settings.model_checkpoint,
        device=settings.device
    )
    
    try:
        predictor.load_model()
        logger.info("SAM service started successfully")
    except Exception as e:
        logger.error(f"Failed to start SAM service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SAM service...")


# Create FastAPI app
app = FastAPI(
    title="Orin SAM Service",
    description="Segment Anything Model (SAM) inference service for NVIDIA Orin platforms",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Orin SAM Service",
        "version": "1.0.0",
        "status": "running",
        "model_type": settings.model_type,
        "device": settings.device
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None or predictor.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_type": settings.model_type,
        "device": settings.device
    }


@app.post("/segment", response_model=SegmentationResponse)
async def segment_image(
    image: UploadFile = File(..., description="Input image file"),
    points: Optional[str] = Form(None, description="JSON string of point prompts"),
    box: Optional[str] = Form(None, description="JSON string of box prompt"),
    multimask_output: bool = Form(True, description="Whether to output multiple masks")
):
    """
    Segment an image using SAM model with prompts.
    
    Args:
        image: Input image file
        points: JSON string of point prompts [{"x": 100, "y": 100, "label": 1}, ...]
        box: JSON string of box prompt {"x1": 50, "y1": 50, "x2": 200, "y2": 200}
        multimask_output: Whether to return multiple masks
        
    Returns:
        Segmentation masks and confidence scores
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Read and process image
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        # Convert to RGB if needed
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        # Convert to numpy array
        image_np = np.array(pil_image)
        logger.info(f"Processing image with shape {image_np.shape}")
        
        # Set image for SAM
        predictor.set_image(image_np)
        
        # Parse prompts
        point_coords = None
        point_labels = None
        box_coords = None
        
        if points:
            import json
            points_list = json.loads(points)
            if points_list:
                point_coords = np.array([[p["x"], p["y"]] for p in points_list])
                point_labels = np.array([p["label"] for p in points_list])
        
        if box:
            import json
            box_dict = json.loads(box)
            box_coords = np.array([box_dict["x1"], box_dict["y1"], box_dict["x2"], box_dict["y2"]])
        
        # Run prediction
        result = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box_coords,
            multimask_output=multimask_output
        )
        
        # Convert masks to list format
        masks_list = [mask.tolist() for mask in result["masks"]]
        scores_list = result["scores"].tolist()
        
        logger.info(f"Generated {len(masks_list)} masks")
        
        return SegmentationResponse(
            masks=masks_list,
            scores=scores_list,
            message="Segmentation completed successfully"
        )
        
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@app.post("/reset")
async def reset_model():
    """Reset the model predictor."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        predictor.reset()
        return {"message": "Model reset successfully"}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False
    )
