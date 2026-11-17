
import cv2
import os
import glob
import multiprocessing as mp

from src.parallel_engine import ParallelEmotionEngine


def main():

    engine = ParallelEmotionEngine(num_workers=4)

    try:
        # Automatically find actual dataset images
        image_paths = glob.glob(
            "dataset/FER2013/train/*/*.jpg"
        )

        if len(image_paths) < 3:
            print("Not enough images found!")
            return

        # Select three sample images
        selected_images = image_paths[:3]

        faces = []

        for path in selected_images:
            image = cv2.imread(path)

            if image is not None:
                faces.append(image)

        if not faces:
            print("Images could not be loaded!")
            return

        # Run parallel processing
        results, total_time = engine.process_faces(faces)

        print("\n===== PARALLEL PROCESSING RESULTS =====")

        for result in results:
            print("\nPerson:", result["person_id"])
            print("Emotion:", result["emotion"])
            print("Worker PID:", result["pid"])
            print("Processing time:",
                  round(result["processing_time"], 5), "seconds")

        print("\nTotal processing time:",
              round(total_time, 5), "seconds")

    finally:
        engine.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
