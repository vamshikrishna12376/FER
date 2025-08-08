import os
import pandas as pd

image_dir = 'data/train'
rows = []
for label in os.listdir(image_dir):
    label_dir = os.path.join(image_dir, label)
    if not os.path.isdir(label_dir):
        continue  # Skip files like .DS_Store
    for img_file in os.listdir(label_dir):
        # ... your image processing ...

        img_path = os.path.join(label_dir, img_file)
        rows.append({'path': img_path, 'label': label})
df = pd.DataFrame(rows)
df.to_csv('train_labels.csv', index=False)
