
import cv2
import time
import multiprocessing as mp

from src.face_detection import FaceDetector
from src.parallel_engine import ParallelEmotionEngine


def main():

    print("Initializing Parallel Emotion Detection...")

    detector = FaceDetector()

    # Four independent CNN worker processes
    engine = ParallelEmotionEngine(
        num_workers=4,
        model_path="models/emotion_model.pth"
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Webcam not available!")
        engine.close()
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Webcam started!")
    print("Press Q to exit.")

    fps = 0
    frame_count = 0
    start_time = time.perf_counter()

    try:

        while True:

            ret, frame = camera.read()

            if not ret:
                break

            # Detect multiple faces
            faces = detector.detect_faces(frame)

            face_crops = []

            # Extract each detected face
            for (x, y, w, h) in faces:

                face = frame[y:y+h, x:x+w]

                if face.size > 0:
                    face_crops.append(face)

            # Send all faces to separate worker tasks
            if face_crops:

                results, processing_time = engine.process_faces(
                    face_crops
                )

            else:
                results = []
                processing_time = 0

            # Draw predictions
            for index, ((x, y, w, h), result) in enumerate(
                zip(faces, results)
            ):

                emotion = result["emotion"]
                confidence = result["confidence"]
                pid = result["pid"]

                label = f"{emotion}: {confidence:.1f}%"

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x, y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {index + 1} | PID: {pid}",
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1
                )

            # Calculate FPS
            frame_count += 1
            current_time = time.perf_counter()

            elapsed = current_time - start_time

            if elapsed >= 1:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = current_time

            # Display system information
            cv2.putText(
                frame,
                f"Faces: {len(faces)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Workers: 4 | Parallel: ON",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Inference: {processing_time * 1000:.1f} ms",
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "LIVE PARALLEL EMOTION DETECTION",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()

        engine.close()

        print("Application stopped successfully!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
