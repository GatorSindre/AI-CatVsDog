import os
import shutil

def organize_folder(folder):
    """Create class folders and move images based on filename."""
    cat_folder = os.path.join(folder, 'cat')
    dog_folder = os.path.join(folder, 'dog')
    os.makedirs(cat_folder, exist_ok=True)
    os.makedirs(dog_folder, exist_ok=True)

    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            if 'cat' in f.lower():
                shutil.move(path, os.path.join(cat_folder, f))
            elif 'dog' in f.lower():
                shutil.move(path, os.path.join(dog_folder, f))

# Run this once for train and test folders
organize_folder('./data/train')
organize_folder('./data/test')

print("✅ All images moved into class folders. You can now comment out this script.")