import kagglehub

print("Downloading FER2013 dataset...")

path = kagglehub.dataset_download("msambare/fer2013")

print("Download completed!")
print("Dataset location:", path)