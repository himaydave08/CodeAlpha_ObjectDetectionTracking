import os
import cv2
import argparse
import logging
import hashlib

from detector import YOLODetector
from tracker import ObjectTracker

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_color(track_id):
    """
    Generates a unique and consistent BGR color for a given track ID.
    """
    hash_object = hashlib.sha256(str(track_id).encode())
    hex_dig = hash_object.hexdigest()
    # Extract RGB values from hash
    r = int(hex_dig[0:2], 16)
    g = int(hex_dig[2:4], 16)
    b = int(hex_dig[4:6], 16)
    return (b, g, r)  # BGR for OpenCV

def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking using YOLOv8 and Deep SORT.")
    parser.add_argument(
        "--input", 
        type=str, 
        default=None, 
        help="Path to input video file. If not specified, webcam (device 0) will be used."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=os.path.join("output", "output_video.mp4"), 
        help="Path where the annotated output video will be saved."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="YOLOv8 pre-trained weights (default: yolov8n.pt)."
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.5, 
        help="Minimum confidence threshold for detections (default: 0.5)."
    )
    parser.add_argument(
        "--no-display", 
        action="store_true", 
        help="Run in headless mode (do not show cv2.imshow window)."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check input source
    if args.input is not None:
        if not os.path.exists(args.input):
            logger.error(f"Input video file not found: {args.input}")
            return
        video_source = args.input
        logger.info(f"Using video file input: {video_source}")
    else:
        video_source = 0
        logger.info("Using webcam input (device 0)")

    # Create output directory if it doesn't exist
    if args.output:
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")

    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        if video_source == 0:
            logger.error("Webcam (device 0) could not be opened. Please make sure a webcam is connected, "
                         "or run with a video file using --input <path_to_video>.")
        else:
            logger.error(f"Could not open video file: {video_source}")
        return

    # Get video properties for writer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 100:
        fps = 30.0  # Fallback FPS
        
    logger.info(f"Video resolution: {width}x{height} @ {fps} FPS")

    # Initialize video writer
    writer = None
    if args.output:
        # Use standard MP4V codec which is universally supported for writing in OpenCV.
        # We will convert it to web-compatible H.264 post-process using FFmpeg.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        logger.info(f"Saving output video to: {args.output}")

    # Initialize detector and tracker
    try:
        detector = YOLODetector(model_name=args.model)
        tracker = ObjectTracker(embedder_gpu=False)  # Run embedder on CPU for safety
    except Exception as e:
        logger.error(f"Failed to initialize detector/tracker components: {e}")
        cap.release()
        if writer:
            writer.release()
        return

    logger.info("Starting processing loop. Press 'q' in the window to quit.")
    
    frame_count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Step 1: Detect objects
            detections = detector.detect(frame, confidence_threshold=args.confidence)
            
            # Step 2: Track objects
            tracks = tracker.update(detections, frame)
            
            # Step 3: Draw annotations
            for track in tracks:
                track_id = track["track_id"]
                bbox = track["bbox"]  # [left, top, right, bottom]
                class_name = track["class_name"]
                
                # Convert coords to integers
                x1, y1, x2, y2 = map(int, bbox)
                
                # Assign consistent colors based on track ID
                color = get_color(track_id)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare text label (e.g. ID: 12 | person)
                label = f"ID: {track_id} | {class_name}"
                
                # Draw label background
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(
                    frame, 
                    (x1, y1 - label_h - 10 if y1 - label_h - 10 > 0 else 0), 
                    (x1 + label_w + 10, y1), 
                    color, 
                    -1
                )
                
                # Draw text label
                cv2.putText(
                    frame, 
                    label, 
                    (x1 + 5, y1 - 5 if y1 - label_h - 10 > 0 else label_h + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    1, 
                    cv2.LINE_AA
                )
            
            # Step 4: Save output
            if writer:
                writer.write(frame)
                
            # Step 5: Show frame
            if not args.no_display:
                cv2.imshow("CodeAlpha - Object Detection & Tracking", frame)
                # Exit if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("User requested exit.")
                    break
                    
            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames...")
                
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"Error during video processing loop: {e}")
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        logger.info(f"Finished processing. Total frames processed: {frame_count}")

        # Post-process: Convert the generated video to H.264 using FFmpeg for web compatibility
        if args.output and os.path.exists(args.output):
            temp_output = args.output.replace(".mp4", "_temp.mp4")
            try:
                import subprocess
                # Rename the generated video to a temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                os.rename(args.output, temp_output)
                
                # Run ffmpeg to convert to H.264 (libx264)
                logger.info("Converting output video to web-compatible H.264 format using FFmpeg...")
                cmd = [
                    "ffmpeg", "-y", "-i", temp_output,
                    "-vcodec", "libx264",
                    "-pix_fmt", "yuv420p",
                    args.output
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                logger.info("Video conversion complete.")
                
                # Remove the temp file
                if os.path.exists(temp_output):
                    os.remove(temp_output)
            except Exception as convert_error:
                logger.warning(f"Could not convert video to H.264 via FFmpeg: {convert_error}. "
                               f"Using original output video.")
                if os.path.exists(temp_output) and not os.path.exists(args.output):
                    os.rename(temp_output, args.output)

if __name__ == "__main__":
    main()
