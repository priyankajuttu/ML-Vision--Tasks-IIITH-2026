from ultralytics import YOLO
import os

# Load YOLO segmentation model
model = YOLO("yolo11n-seg.pt")

# Run segmentation on frames
print("Running Object Segmentation...")
model.predict(
    source=r"C:\yt-dlp\frames1800",
    save=True,
    project=r"C:\ML-Internship",
    name="segmented_frames",
    conf=0.5,
    exist_ok=True
)
print("Segmentation complete!")