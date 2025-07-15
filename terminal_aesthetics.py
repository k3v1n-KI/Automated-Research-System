import sys
import itertools
import threading
import time

# ——— Spinner Helper ———
class Spinner:
    def __init__(self, message="Loading", delay=0.1):
        self.message = message
        self.delay = delay
        self._running = False
        self._spinner = itertools.cycle("|/-\\")
        self._thread = None

    def _spin(self):
        while self._running:
            char = next(self._spinner)
            sys.stdout.write(f"\r{self.message}... {char}")
            sys.stdout.flush()
            time.sleep(self.delay)
        sys.stdout.write("\r" + " " * (len(self.message) + 5) + "\r")
        sys.stdout.flush()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
