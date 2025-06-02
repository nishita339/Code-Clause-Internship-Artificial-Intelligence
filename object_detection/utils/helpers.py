import cv2
import numpy as np
import time
from PIL import Image
import torch

# Generate a list of distinct colors for bounding boxes
def generate_colors(num_classes):
    """
    Generate a list of distinct colors for visualization.
    
    Args:
        num_classes: Number of classes
        
    Returns:
        List of RGB colors
    """
    np.random.seed(42)  # For reproducibility
    colors = []
    for i in range(num_classes):
        # Generate vibrant colors
        hue = i / num_classes
        saturation = 0.8 + np.random.random() * 0.2
        value = 0.8 + np.random.random() * 0.2
        
        # Convert HSV to RGB
        h = hue * 6
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if 0 <= h < 1:
            r, g, b = c, x, 0
        elif 1 <= h < 2:
            r, g, b = x, c, 0
        elif 2 <= h < 3:
            r, g, b = 0, c, x
        elif 3 <= h < 4:
            r, g, b = 0, x, c
        elif 4 <= h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        # Convert to 0-255 range
        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        
        colors.append((r, g, b))
    
    return colors

# Format time in a human-readable way
def format_time(seconds):
    """
    Format time in a human-readable way.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"

# Create video writer
def get_video_writer(input_path, output_path):
    """
    Create a video writer based on input video specifications.
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
        
    Returns:
        cv2.VideoWriter object, frame width, frame height
    """
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.release()
    
    return out, width, height

# Convert normalized coordinates to absolute
def convert_to_absolute_coords(box, width, height):
    """
    Convert normalized coordinates (0-1) to absolute pixel coordinates.
    
    Args:
        box: Bounding box in normalized coordinates [x1, y1, x2, y2]
        width: Image width
        height: Image height
        
    Returns:
        Bounding box in absolute coordinates [x1, y1, x2, y2]
    """
    x1 = int(box[0] * width)
    y1 = int(box[1] * height)
    x2 = int(box[2] * width)
    y2 = int(box[3] * height)
    
    return [x1, y1, x2, y2]

# Check if CUDA is available and set device
def get_device():
    """
    Check if CUDA is available and return device.
    
    Returns:
        torch.device object
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')