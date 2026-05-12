from ultralytics import YOLO
import os
import shutil
import cv2

model = YOLO("yolo11n.pt")

all_frames_dir = r"C:\BikeDataset\all_frames"
output_dir = r"C:\BikeDataset\test_results"

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

images = sorted([f for f in os.listdir(all_frames_dir) if f.endswith('.jpg')])
print(f"Total frames: {len(images)}")

detected_count = 0

for i, img_file in enumerate(images):
    img_path = os.path.join(all_frames_dir, img_file)
    results = model.predict(img_path, conf=0.1, classes=[2, 5], verbose=False)
    annotated = results[0].plot()
    out_path = os.path.join(output_dir, img_file)
    cv2.imwrite(out_path, annotated)
    if len(results[0].boxes) > 0:
        detected_count += 1
    if (i+1) % 50 == 0:
        print(f"Processed {i+1}/{len(images)} | Detections: {detected_count}")

print(f"Done! Frames with detections: {detected_count}")