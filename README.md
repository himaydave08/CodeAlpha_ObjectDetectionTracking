# Real-Time Object Detection and Tracking (YOLOv8 + Deep SORT)

This repository implements a modular, high-performance real-time object detection and multi-object tracking pipeline. It leverages the state-of-the-art **YOLOv8** model for high-accuracy detection and the **Deep SORT** (Simple Online and Realtime Tracking with a Deep Association Metric) algorithm for robust ID persistence across frames.

Developed as part of the **CodeAlpha AI Internship**.

---

## 🚀 How it Works

The pipeline follows a two-stage detection-and-tracking paradigm:

1. **Object Detection (YOLOv8):** 
   - Each video frame is processed by a pre-trained YOLOv8 model (e.g., `yolov8n.pt`).
   - The model predicts bounding boxes, class labels, and confidence scores for target objects.
   - Detections with scores below a user-defined threshold are discarded.

2. **Object Tracking (Deep SORT):**
   - The remaining bounding boxes are fed to the Deep SORT tracker.
   - Deep SORT extracts appearance features (embeddings) using a lightweight convolutional neural network (e.g., MobileNet).
   - A Kalman filter predicts the object state in the next frame.
   - The Hungarian algorithm performs data association, matching tracks to new detections using motion (Mahalanobis distance) and appearance similarity (cosine distance).
   - This ensures objects maintain consistent, unique tracking IDs even during partial occlusions or camera movements.

---

## 🛠️ Tech Stack

- **Core Language:** Python 3.8+
- **Deep Learning Model:** YOLOv8 (`ultralytics` package)
- **Tracking Algorithm:** Deep SORT (`deep-sort-realtime` package)
- **Computer Vision:** OpenCV (`opencv-python` package)
- **Array Processing:** NumPy

---

## 📂 Project Structure

```
CodeAlpha_ObjectDetectionTracking/
├── detector.py          # Wrapper for YOLOv8 model loading & inference
├── tracker.py           # Wrapper for Deep SORT initialization & updates
├── main.py              # Video processing loop, arguments, and rendering
├── requirements.txt     # Python dependencies
├── .gitignore           # File/folder exclusions for Git
└── README.md            # Project documentation (this file)
```

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CodeAlpha_ObjectDetectionTracking
```

### 2. Create and Activate a Virtual Environment
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

Run the tracking pipeline using `main.py`. The script automatically handles input from either a live webcam stream or an offline video file.

### 1. Webcam Mode (Real-Time)
Run tracking using your default connected webcam:
```bash
python main.py
```

### 2. Video File Mode
Run tracking on a specific video file (e.g. for offline analysis or sandboxed environments):
```bash
python main.py --input sample_videos/sample.mp4 --output output/output_video.mp4
```

### 3. Headless Mode (Server/No-GUI Mode)
To run tracking on a video file and save the output *without* opening a display window:
```bash
python main.py --input sample_videos/sample.mp4 --output output/output_video.mp4 --no-display
```

### Command Line Arguments
- `--input`: Path to input video file (omit to use webcam).
- `--output`: Path to save the annotated output video (default: `output/output_video.mp4`).
- `--model`: YOLOv8 weight file to use (default: `yolov8n.pt`).
- `--confidence`: Confidence score threshold for detections (default: `0.5`).
- `--no-display`: Run without opening GUI display window.

---

## 📊 Sample Output

When the script runs, it generates an annotated video. For each tracked object, a bounding box is drawn along with:
- **Track ID:** A persistent unique integer assigned to the object.
- **Class Label:** The class name (e.g. `person`, `car`, `bicycle`).
- **Color Coding:** Consistent color for each unique tracking ID.

### Output Sample Preview
*(The resulting annotated video is saved to `output/output_video.mp4`)*

![Object Tracking Visual Placeholder](https://user-images.githubusercontent.com/placeholder-tracking-output.png)
