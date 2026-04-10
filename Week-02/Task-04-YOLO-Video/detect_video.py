from ultralytics import YOLO
import os
import subprocess

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Path to your Week 1 frames
frames_folder = r"C:\yt-dlp\frames1800"

# Output folder for annotated frames
output_folder = r"C:\ML-Internship\annotated_frames"
os.makedirs(output_folder, exist_ok=True)

# Run YOLO detection on all frames
print("Running YOLO detection on all frames...")
results = model.predict(
    source=frames_folder,
    save=True,
    project=r"C:\ML-Internship",
    name="annotated_frames",
    conf=0.5,
    exist_ok=True
)

print(f"Detection complete! Annotated frames saved to {output_folder}")

# Count annotated frames
frames = sorted([f for f in os.listdir(output_folder) if f.endswith('.jpg')])
print(f"Total annotated frames: {len(frames)}")