from ultralytics import YOLO
import os

# Load YOUR trained weights
model = YOLO(r"C:\BikeDataset\weights\best.pt")

# Run detection on test images
results = model.predict(
    source=r"C:\BikeDataset\images\test",
    save=True,
    conf=0.3,
    project=r"C:\BikeDataset",
    name="test_results",
    exist_ok=True
)

print(f"\n🎉 Detection complete!")
print(f"Results saved to: C:\\BikeDataset\\test_results\\")

# Print summary
total_cars = 0
total_buses = 0
for result in results:
    for box in result.boxes:
        cls = int(box.cls)
        if cls == 0:
            total_cars += 1
        elif cls == 1:
            total_buses += 1

print(f"Total cars detected  : {total_cars}")
print(f"Total buses detected : {total_buses}")