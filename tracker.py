import logging
from deep_sort_realtime.deepsort_tracker import DeepSort

logger = logging.getLogger(__name__)

class ObjectTracker:
    """
    A wrapper class for Deep SORT multi-object tracking.
    """
    def __init__(self, max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.2, embedder="mobilenet", embedder_gpu=False):
        """
        Initializes the Deep SORT tracker.
        
        Args:
            max_age (int): Maximum number of frames to keep a track alive without detections.
            n_init (int): Number of consecutive detections before the track is confirmed.
            nms_max_overlap (float): Non-maximum suppression threshold.
            max_cosine_distance (float): Gating threshold for cosine distance.
            embedder (str): Feature extractor model name ("mobilenet", "torchvision", etc.).
            embedder_gpu (bool): Use GPU for embedding extraction. Default is False for safety on CPU systems.
        """
        logger.info(f"Initializing Deep SORT tracker (embedder: {embedder}, GPU: {embedder_gpu})")
        try:
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                nms_max_overlap=nms_max_overlap,
                max_cosine_distance=max_cosine_distance,
                embedder=embedder,
                embedder_gpu=embedder_gpu
            )
        except Exception as e:
            logger.error(f"Failed to initialize Deep SORT: {e}")
            raise RuntimeError(f"Error initializing Deep SORT tracker: {e}")

    def update(self, detections, frame):
        """
        Updates the tracker with new detections.
        
        Args:
            detections (list): List of detections in format ([left, top, w, h], confidence, class_name).
            frame (numpy.ndarray): The input frame.
            
        Returns:
            list: List of active, confirmed tracks as dictionaries:
                  {"track_id": track_id, "bbox": [left, top, right, bottom], "class_name": class_name}
        """
        if frame is None:
            return []
            
        try:
            # Update tracks
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            active_tracks = []
            for track in tracks:
                # Only process confirmed tracks that are currently active
                # track.is_confirmed() determines if track has been seen n_init times
                if not track.is_confirmed():
                    continue
                
                # Retrieve bounding box in [left, top, right, bottom] format
                ltrb = track.to_ltrb()
                track_id = track.track_id
                class_name = track.get_det_class()
                
                active_tracks.append({
                    "track_id": track_id,
                    "bbox": ltrb,
                    "class_name": class_name
                })
            
            return active_tracks
        except Exception as e:
            logger.error(f"Error during tracker update: {e}")
            return []
