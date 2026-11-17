import os
import shutil
import kagglehub

# Get downloaded dataset
source = kagglehub.dataset_download("msambare/fer2013")

# Destination inside project
destination = os.path.join("dataset", "FER2013")

os.makedirs(destination, exist_ok=True)

# Copy dataset
for item in os.listdir(source):
    src = os.path.join(source, item)
    dst = os.path.join(destination, item)

    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Dataset copied successfully!")

# Display dataset structure
for folder in os.listdir(destination):
    path = os.path.join(destination, folder)

    if os.path.isdir(path):
        print(f"\n{folder}:")
        for emotion in os.listdir(path):
            emotion_path = os.path.join(path, emotion)
            if os.path.isdir(emotion_path):
                print(emotion, ":", len(os.listdir(emotion_path)), "images")