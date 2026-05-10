from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

# Train on our dataset
results = model.train(
    data=r"C:\BikeDataset\bike_dataset.yaml",
    epochs=50,
    imgsz=384,
    batch=8,
    name="car_bus_detection",
    patience=10,
    save=True,
    plots=True
)

print("Training complete!")
print(f"Best weights saved at: {results.save_dir}")