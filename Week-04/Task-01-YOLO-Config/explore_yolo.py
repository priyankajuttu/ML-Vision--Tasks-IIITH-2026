import yaml
import os

# ── Use YAML files already in your venv ──
yaml_path = r"C:\ML-Internship\venv\Lib\site-packages\ultralytics\cfg\datasets\coco128.yaml"

# Fix encoding error with utf-8
with open(yaml_path, 'r', encoding='utf-8') as f:
    data = yaml.safe_load(f)

print("=" * 50)
print("YAML CONFIGURATION FILE ANALYSIS")
print("=" * 50)
print(f"Dataset Path  : {data.get('path', 'N/A')}")
print(f"Train Path    : {data.get('train', 'N/A')}")
print(f"Val Path      : {data.get('val', 'N/A')}")
print(f"Num Classes   : {data.get('nc', 'N/A')}")
print(f"\nAll Class Names:")
for i, name in enumerate(data.get('names', [])):
    print(f"  {i:3d}: {name}")

# ── Also read YOLO26 model yaml ──
model_yaml = r"C:\ML-Internship\venv\Lib\site-packages\ultralytics\cfg\models\26\yolo26.yaml"

print("\n" + "=" * 50)
print("YOLO26 MODEL CONFIG FILE ANALYSIS")
print("=" * 50)
with open(model_yaml, 'r', encoding='utf-8') as f:
    content = f.read()
print(content)

print("\nAnalysis Complete!")