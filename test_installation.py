import cv2
import torch
import streamlit
import psutil

print("OpenCV:", cv2.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CPU cores:", psutil.cpu_count())