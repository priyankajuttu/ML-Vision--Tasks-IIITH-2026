from ultralytics import YOLO

# Load YOLO detection model
model = YOLO("yolo11n.pt")

print("Running Object Detection...")
model.predict(
    source=r"C:\yt-dlp\frames2",
    save=True,
    project=r"C:\ML-Internship",
    name="detected_frames2",
    conf=0.5,
    exist_ok=True
)
print("Detection complete!")