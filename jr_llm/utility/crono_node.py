import time
import threading
import logging
import subprocess

class ChronoNode:
    def __init__(self):
        self.stopwatch_start = 0.0
        self.stopwatch_running = False

    def get_clock(self):
        current = time.strftime("%Y-%m-%d %H:%M:%S")
        logging.info(f"Clock: {current}")
        return current

    def toggle_stopwatch(self):
        if not self.stopwatch_running:
            self.stopwatch_start = time.time()
            self.stopwatch_running = True
            return "Stopwatch started."
        else:
            elapsed = time.time() - self.stopwatch_start
            self.stopwatch_running = False
            return f"Stopwatch stopped. Elapsed: {elapsed:.2f}s"

    def deploy_timer(self, seconds: int):
        def _thread():
            time.sleep(seconds)
            subprocess.Popen(["say", f"'Timer complete: {seconds}s'"])
            logging.warning(f"TIMER COMPLETE ({seconds}s)")
        threading.Thread(target=_thread, daemon=True).start()