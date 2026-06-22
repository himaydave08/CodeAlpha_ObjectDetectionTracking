import os
import logging
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class YOLODetector:
    """
    A wrapper class for YOLOv8 object detection using the Ultralytics library.
    """
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 model.
        
        Args:
            model_name (str): The name or path of the pre-trained YOLOv8 weights (e.g., yolov8n.pt).
        """
        logger.info(f"Initializing YOLO detector with model: {model_name}")
        try:
            # Load the model. Ultralytics automatically downloads the weights
            # if the file is not found locally.
            self.model = YOLO(model_name)
        except Exception as e:
            logger.error(f"Failed to load YOLO model '{model_name}': {e}")
            raise RuntimeError(f"Error loading YOLO model '{model_name}': {e}")

    def detect(self, frame, confidence_threshold=0.5):
        """
        Runs object detection on a single frame.
        
        Args:
            frame (numpy.ndarray): The input image frame.
            confidence_threshold (float): Minimum confidence score to filter detections.
            
        Returns:
            list: A list of detections, where each detection is a tuple of:
                  ([left, top, width, height], confidence, class_name)
        """
        if frame is None:
            return []
            
        try:
            # Run inference with verbose=False to keep logs clean
            results = self.model(frame, verbose=False)[0]
            
            detections = []
            for box in results.boxes:
                # Extract coordinates, confidence and class
                xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                conf = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names.get(class_id, "unknown")
                
                # Filter by confidence
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = xyxy
                    # Convert coordinates to [left, top, width, height] format for Deep SORT
                    w = x2 - x1
                    h = y2 - y1
                    detections.append(([x1, y1, w, h], conf, class_name))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection inference: {e}")
            return []
