
import cv2
import time
import av
import threading
from collections import Counter

from src.face_detection import FaceDetector
from src.parallel_engine import ParallelEmotionEngine


class EmotionVideoProcessor:

    def __init__(self, workers=4):

        self.detector = FaceDetector()

        self.engine = ParallelEmotionEngine(
            num_workers=workers,
            model_path="models/emotion_model.pth"
        )

        self.lock = threading.Lock()

        self.total_faces = 0
        self.fps = 0
        self.inference_ms = 0
        self.emotion_counts = Counter()
        self.worker_pids = set()
        self.latest_results = []

        self.previous_time = time.perf_counter()
        self.frame_count = 0

        # Store the most recent predictions
        self.last_faces = []
        self.last_results = []

    def recv(self, frame):

        image = frame.to_ndarray(format="bgr24")

        self.frame_count += 1

        # Process every alternate frame
        if self.frame_count % 2 == 1:

            original_h, original_w = image.shape[:2]
            scale = 0.75

            small_image = cv2.resize(
                image,
                (int(original_w * scale), int(original_h * scale))
            )

            faces = self.detector.detect_faces(small_image)

            crops = []

            for x, y, w, h in faces:

                crop = small_image[y:y+h, x:x+w]

                if crop.size > 0:
                    crops.append(crop)

            results = []
            processing_time = 0

            if crops:
                results, processing_time = (
                    self.engine.process_faces(crops)
                )

            self.last_faces = faces
            self.last_results = results

            current_emotions = [
                result["emotion"] for result in results
            ]

            now = time.perf_counter()

            instant_fps = 1 / max(
                now - self.previous_time,
                0.001
            )

            self.previous_time = now

            with self.lock:

                self.total_faces = len(faces)
                self.inference_ms = processing_time * 1000
                self.fps = instant_fps
                self.latest_results = results

                for emotion in current_emotions:
                    self.emotion_counts[emotion] += 1

                for result in results:
                    self.worker_pids.add(result["pid"])

        # Draw the most recent detections on every frame
        for (x, y, w, h), result in zip(
            self.last_faces,
            self.last_results
        ):

            x = int(x / scale)
            y = int(y / scale)
            w = int(w / scale)
            h = int(h / scale)

            emotion = result["emotion"]
            confidence = result["confidence"]
            pid = result["pid"]

            cv2.rectangle(
                image,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"{emotion} {confidence:.1f}%",
                (x, max(y-10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"PID: {pid}",
                (x, y+h+25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        # Display metrics
        cv2.putText(
            image,
            f"Faces: {len(self.last_faces)}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"FPS: {self.fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )

    def close(self):

        self.engine.close()
