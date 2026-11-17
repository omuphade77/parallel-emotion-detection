
import multiprocessing as mp
import time
import os

from src.worker import process_face, initialize_worker


class ParallelEmotionEngine:

    def __init__(
        self,
        num_workers=4,
        model_path="models/emotion_model.pth"
    ):

        self.num_workers = num_workers
        self.closed = False

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.pool = mp.Pool(
            processes=num_workers,
            initializer=initialize_worker,
            initargs=(os.path.abspath(model_path),)
        )

        print(f"Started {num_workers} CNN workers")

    @property
    def worker_pids(self):

        if self.closed:
            return []

        return [
            process.pid
            for process in self.pool._pool
            if process.is_alive()
        ]

    def warmup(self, sample_face):

        if self.closed:
            raise RuntimeError("Worker pool is closed")

        # Give all workers warm-up tasks.
        tasks = [
            (index + 1, sample_face)
            for index in range(self.num_workers * 2)
        ]

        self.pool.map(
            process_face,
            tasks,
            chunksize=1
        )

        print("Parallel workers warmed up")

    def process_faces(self, faces):

        if self.closed:
            raise RuntimeError("Worker pool is closed")

        if not faces:
            return [], 0.0

        tasks = [
            (index + 1, face)
            for index, face in enumerate(faces)
        ]

        start = time.perf_counter()

        results = self.pool.map(
            process_face,
            tasks,
            chunksize=1
        )

        total_time = time.perf_counter() - start

        return results, total_time

    def close(self):

        if not self.closed:

            self.closed = True

            self.pool.close()
            self.pool.join()

            print("All workers stopped")
