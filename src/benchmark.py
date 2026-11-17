
import os
import glob
import time
import cv2
import torch
import numpy as np
import pandas as pd
import multiprocessing as mp

from src.worker import EmotionCNN
from src.parallel_engine import ParallelEmotionEngine


MODEL_PATH = "models/emotion_model.pth"
DATA_PATH = "dataset/FER2013/test"

WORKER_COUNTS = [1, 2, 4, 8]
BATCH_SIZES = [1, 2, 4, 8, 16]
REPEATS = 5


def load_model():

    torch.set_num_threads(1)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = EmotionCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def preprocess(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48))

    image = gray.astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5

    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def sequential_prediction(model, faces):

    start = time.perf_counter()

    with torch.inference_mode():

        for face in faces:

            tensor = preprocess(face)
            model(tensor)

    return time.perf_counter() - start


def main():

    print("\n===== PERFORMANCE BENCHMARK =====")

    # Load sample images
    paths = sorted(
        glob.glob(os.path.join(DATA_PATH, "*", "*.jpg"))
    )

    if len(paths) < 16:
        print("Not enough test images!")
        return

    faces = []

    for path in paths[:16]:

        image = cv2.imread(path)

        if image is not None:
            faces.append(image)

    if len(faces) < 16:
        print("Could not load 16 images!")
        return

    print("Sample images loaded:", len(faces))

    model = load_model()

    # Warm up sequential model
    sequential_prediction(model, faces[:2])

    all_results = []

    try:

        for batch_size in BATCH_SIZES:

            test_faces = faces[:batch_size]

            print(f"\nTesting {batch_size} faces")

            # Sequential benchmark
            sequential_times = []

            for _ in range(REPEATS):

                elapsed = sequential_prediction(
                    model,
                    test_faces
                )

                sequential_times.append(elapsed)

            seq_avg = np.mean(sequential_times)

            print(
                f"Sequential: {seq_avg * 1000:.3f} ms"
            )

            # Parallel benchmarks
            for workers in WORKER_COUNTS:

                engine = ParallelEmotionEngine(
                    num_workers=workers,
                    model_path=MODEL_PATH
                )

                try:

                    # Warm-up: initialize and execute tasks
                    engine.process_faces(test_faces)

                    parallel_times = []

                    for _ in range(REPEATS):

                        _, elapsed = engine.process_faces(
                            test_faces
                        )

                        parallel_times.append(elapsed)

                    par_avg = np.mean(parallel_times)

                finally:
                    engine.close()

                # Calculate performance metrics
                speedup = seq_avg / par_avg if par_avg > 0 else 0

                efficiency = (
                    speedup / workers
                ) * 100

                all_results.append({

                    "Faces": batch_size,
                    "Workers": workers,
                    "Sequential_ms": round(seq_avg * 1000, 3),
                    "Parallel_ms": round(par_avg * 1000, 3),
                    "Speedup": round(speedup, 3),
                    "Efficiency_percent": round(efficiency, 2)

                })

                print(
                    f"Workers: {workers} | "
                    f"Parallel: {par_avg * 1000:.3f} ms | "
                    f"Speedup: {speedup:.2f}x | "
                    f"Efficiency: {efficiency:.2f}%"
                )

    except Exception as error:

        print("Benchmark error:", error)
        raise

    # Save results
    df = pd.DataFrame(all_results)

    os.makedirs("results", exist_ok=True)

    df.to_csv(
        "results/benchmark_results.csv",
        index=False
    )

    print("\n===== FINAL BENCHMARK TABLE =====")
    print(df.to_string(index=False))

    print("\nResults saved to:")
    print("results/benchmark_results.csv")


if __name__ == "__main__":

    mp.freeze_support()
    main()
