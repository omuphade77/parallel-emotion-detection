
import cv2
import time

from src.face_detection import FaceDetector


def main():

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam")
        return

    print("Webcam started!")
    print("Press Q to stop.")

    while True:

        start_time = time.time()

        ret, frame = camera.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Detect multiple faces
        faces = detector.detect_faces(frame)

        # Draw bounding boxes
        frame = detector.draw_faces(frame, faces)

        # Calculate FPS
        fps = 1 / max(time.time() - start_time, 0.001)

        cv2.putText(
            frame,
            f"Faces: {len(faces)} | FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        cv2.imshow("Parallel Emotion Detection - Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
