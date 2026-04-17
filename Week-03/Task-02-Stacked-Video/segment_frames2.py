from ultralytics import YOLO

# Load YOLO segmentation model
model = YOLO("yolo11n-seg.pt")

print("Running Object Segmentation...")
model.predict(
    source=r"C:\yt-dlp\frames2",
    save=True,
    project=r"C:\ML-Internship",
    name="segmented_frames2",
    conf=0.5,
    exist_ok=True
)
print("Segmentation complete!")