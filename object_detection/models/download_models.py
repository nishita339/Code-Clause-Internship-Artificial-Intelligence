import torch
import torchvision
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_faster_rcnn():
    """
    Download Faster R-CNN model pretrained on COCO dataset.
    """
    logger.info("Downloading Faster R-CNN pretrained model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    logger.info("Faster R-CNN model downloaded successfully.")
    return model

def download_ssd():
    """
    Download SSD model pretrained on COCO dataset.
    """
    logger.info("Downloading SSD pretrained model...")
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
    logger.info("SSD model downloaded successfully.")
    return model

def download_all_models():
    """
    Download all models and cache them.
    """
    logger.info("Starting model downloads...")
    
    # Create models directory if it doesn't exist
    os.makedirs("weights", exist_ok=True)
    
    try:
        # Download Faster R-CNN
        faster_rcnn = download_faster_rcnn()
        
        # Download SSD
        ssd = download_ssd()
        
        logger.info("All models downloaded successfully.")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading models: {str(e)}")
        return False

if __name__ == "__main__":
    download_all_models()