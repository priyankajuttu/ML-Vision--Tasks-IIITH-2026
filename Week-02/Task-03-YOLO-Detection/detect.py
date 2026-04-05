from ultralytics import YOLO

# Load pretrained YOLOv8 model
model = YOLO("yolo11n.pt")

# Run detection on dog image
results = model.predict(
    source=r"C:\Users\Juttu\OneDrive\Pictures\Petopia\dog.jpg.jpg",
    save=True,
    conf=0.5
)

# Print results
for result in results:
    print("Detected Objects:")
    for box in result.boxes:
        cls = result.names[int(box.cls)]
        conf = float(box.conf)
        print(f"  - {cls}: {conf:.2f} confidence")