from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Run object detection on frames
print("Running Object Detection...")
model.predict(
    source=r"C:\yt-dlp\frames1800",
    save=True,
    project=r"C:\ML-Internship",
    name="detected_frames",
    conf=0.5,
    exist_ok=True
)
print("Detection complete!")