"""
Example client for using the SAM service.
"""
import requests
import json
import sys
from pathlib import Path


def segment_image(
    image_path: str,
    points: list = None,
    box: dict = None,
    multimask_output: bool = True,
    service_url: str = "http://localhost:8000"
):
    """
    Segment an image using the SAM service.
    
    Args:
        image_path: Path to the input image
        points: List of point prompts [{"x": 100, "y": 100, "label": 1}, ...]
        box: Box prompt {"x1": 50, "y1": 50, "x2": 200, "y2": 200}
        multimask_output: Whether to return multiple masks
        service_url: URL of the SAM service
        
    Returns:
        Response dictionary with masks and scores
    """
    # Prepare the image file
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    files = {"image": open(image_path, "rb")}
    
    # Prepare data
    data = {
        "multimask_output": multimask_output
    }
    
    if points:
        data["points"] = json.dumps(points)
    
    if box:
        data["box"] = json.dumps(box)
    
    # Make request
    response = requests.post(
        f"{service_url}/segment",
        files=files,
        data=data
    )
    
    if response.status_code != 200:
        raise Exception(f"Request failed: {response.status_code} - {response.text}")
    
    return response.json()


def main():
    """Example usage."""
    print("SAM Service Client Example")
    print("=" * 50)
    
    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python example_client.py <image_path>")
        print("\nExample:")
        print("  python example_client.py test_image.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Example 1: Segment with a point prompt
    print("\nExample 1: Segmentation with point prompt")
    print("-" * 50)
    try:
        result = segment_image(
            image_path=image_path,
            points=[
                {"x": 100, "y": 100, "label": 1},  # Foreground point
            ]
        )
        
        print(f"✓ Generated {len(result['masks'])} masks")
        print(f"  Scores: {result['scores']}")
        print(f"  Message: {result['message']}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Example 2: Segment with a box prompt
    print("\nExample 2: Segmentation with box prompt")
    print("-" * 50)
    try:
        result = segment_image(
            image_path=image_path,
            box={"x1": 50, "y1": 50, "x2": 200, "y2": 200}
        )
        
        print(f"✓ Generated {len(result['masks'])} masks")
        print(f"  Scores: {result['scores']}")
        print(f"  Message: {result['message']}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    # Example 3: Segment with multiple points
    print("\nExample 3: Segmentation with multiple points")
    print("-" * 50)
    try:
        result = segment_image(
            image_path=image_path,
            points=[
                {"x": 100, "y": 100, "label": 1},  # Foreground
                {"x": 150, "y": 150, "label": 1},  # Foreground
                {"x": 50, "y": 50, "label": 0},    # Background
            ]
        )
        
        print(f"✓ Generated {len(result['masks'])} masks")
        print(f"  Scores: {result['scores']}")
        print(f"  Message: {result['message']}")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()
