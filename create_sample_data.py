import os
from PIL import Image
import random

data_dir = r"C:\Users\z0045n5j\Documents\tech\s3\MLO\ASSGN\2\data\processed"

# Create sample images
splits = ["train", "val", "test"]
classes = ["cats", "dogs"]
counts = {"train": 50, "val": 10, "test": 10}  # Small dataset for quick testing

for split in splits:
    for class_idx, class_name in enumerate(classes):
        dir_path = os.path.join(data_dir, split, class_name)
        count = counts[split]
        
        for i in range(count):
            # Create random RGB image
            img = Image.new('RGB', (224, 224), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            img.save(os.path.join(dir_path, f"{class_name}_{i:04d}.jpg"))
        
        print(f"Created {count} {class_name} images in {split} split")

print("Sample dataset created successfully!")
