"""Task-level progress and ETA based on work completed in this invocation."""

import math
import time


def duration(seconds):
    seconds = max(0, math.ceil(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class Progress:
    def __init__(self, total, completed=0):
        self.total = total
        self.completed = completed
        self.initial = completed
        self.started = time.monotonic()

    def advance(self, label, *, detail="", force=False):
        self.completed += 1
        if not (force or self.completed % 25 == 0 or self.completed == self.total):
            return
        elapsed = max(0, time.monotonic() - self.started)
        newly_completed = self.completed - self.initial
        remaining = max(0, self.total - self.completed)
        eta = (
            duration(elapsed * remaining / newly_completed)
            if elapsed > 0
            else "estimating"
        )
        if remaining == 0:
            eta = "00:00:00"
        percent = 100 * self.completed / self.total
        print(
            f"{label}: {detail + ' | ' if detail else ''}"
            f"task={self.completed}/{self.total} ({percent:.1f}%) | ETA={eta}",
            flush=True,
        )
