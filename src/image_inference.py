
import time
import cv2
import numpy as np
import torch

from src.face_detection import FaceDetector
from src.worker import initialize_worker, process_face
from src.parallel_engine import ParallelEmotionEngine


MODEL_PATH = "models/emotion_model.pth"

# Keep the sequential model loaded between image predictions
_sequential_initialized = False


def detect_and_crop_faces(image_rgb):

    image_bgr = cv2.cvtColor(
        np.array(image_rgb),
        cv2.COLOR_RGB2BGR
    )

    detector = FaceDetector()
    faces = detector.detect_faces(image_bgr)

    crops = []

    for x, y, w, h in faces:

        crop = image_bgr[y:y+h, x:x+w]

        if crop.size > 0:
            crops.append(crop)

    return image_bgr, faces, crops


def run_sequential(crops):

    global _sequential_initialized

    if not crops:
        return [], 0.0

    # Match the one-thread-per-worker CPU configuration
    torch.set_num_threads(1)

    # Load model only once
    if not _sequential_initialized:

        initialize_worker(MODEL_PATH)

        _sequential_initialized = True

    # Warm-up (excluded from measurement)
    process_face((0, crops[0]))

    results = []

    start_time = time.perf_counter()

    for index, crop in enumerate(crops):

        result = process_face((index + 1, crop))
        results.append(result)

    elapsed_time = time.perf_counter() - start_time

    return results, elapsed_time


def create_parallel_engine(workers=4):

    engine = ParallelEmotionEngine(
        num_workers=workers,
        model_path=MODEL_PATH
    )

    # Initialize and warm up worker models
    # before measuring actual predictions.
    dummy_face = np.zeros(
        (48, 48, 3),
        dtype=np.uint8
    )

    engine.warmup(dummy_face)

    return engine


def run_parallel(crops, engine):

    if not crops:
        return [], 0.0

    # Measure actual prediction time
    results, elapsed_time = engine.process_faces(crops)

    return results, elapsed_time


def annotate_image(image_bgr, faces, results):

    output = image_bgr.copy()

    for (x, y, w, h), result in zip(faces, results):

        emotion = result["emotion"]
        confidence = result["confidence"]

        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )


def calculate_speedup(sequential_time, parallel_time):

    if parallel_time <= 0:
        return 0.0

    return sequential_time / parallel_time
