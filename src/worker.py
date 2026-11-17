
import os
import time
import cv2
import torch
import torch.nn as nn
import numpy as np

# Model architecture must match train.py
class EmotionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Global variables initialized separately in each process
model = None
emotion_classes = None


def initialize_worker(model_path):

    global model, emotion_classes

    torch.set_num_threads(1)

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model = EmotionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    emotion_classes = checkpoint["classes"]

    print(
        f"Worker initialized | PID: {os.getpid()}",
        flush=True
    )


def process_face(task):

    person_id, face_image = task

    start = time.perf_counter()

    # Convert face into grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Resize to CNN input dimensions
    gray = cv2.resize(gray, (48, 48))

    # Normalize exactly like training
    image = gray.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    tensor = torch.from_numpy(image)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    with torch.inference_mode():

        outputs = model(tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(
            probabilities, dim=1
        )

    emotion = emotion_classes[predicted.item()]

    processing_time = time.perf_counter() - start

    return {
        "person_id": person_id,
        "emotion": emotion.capitalize(),
        "confidence": round(confidence.item() * 100, 2),
        "pid": os.getpid(),
        "processing_time": round(processing_time, 5)
    }
