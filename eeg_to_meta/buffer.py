"""Thread-safe rolling EEG ring buffer for real-time streaming.

Samples arrive one-by-one (or in small batches) from the Muse stream.
The buffer stores the most recent `max_samples` and provides windowed
read-outs at the configured stride interval without copying the entire
history each time.
"""

import threading

import numpy as np

from config import (
    N_CHANNELS, SFREQ,
    MI_WINDOW_SAMPLES, BLINK_WINDOW_SAMPLES,
    STRIDE_SAMPLES,
)


class RingBuffer:
    """Lock-protected ring buffer for multi-channel EEG.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels (default 4 for Muse 2).
    max_seconds : float
        How many seconds of history to retain (default 10 s).
    sfreq : int
        Sampling rate in Hz (default 256).
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        max_seconds: float = 10.0,
        sfreq: int = SFREQ,
    ):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.max_samples = int(max_seconds * sfreq)
        self._buf = np.zeros((n_channels, self.max_samples), dtype=np.float64)
        self._write_pos = 0        # monotonically increasing sample counter
        self._lock = threading.Lock()

    @property
    def total_written(self) -> int:
        """Total number of samples ever pushed (not just stored)."""
        return self._write_pos

    # ── writing ─────────────────────────────────────────────────────

    def push(self, samples: np.ndarray):
        """Append new samples to the buffer.

        Parameters
        ----------
        samples : ndarray
            Shape (n_channels,) for a single sample, or
            (n_channels, n_new) for a batch.
        """
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]   # (n_ch, 1)

        n_new = samples.shape[1]

        with self._lock:
            for i in range(n_new):
                idx = self._write_pos % self.max_samples
                self._buf[:, idx] = samples[:, i]
                self._write_pos += 1

    # ── reading ─────────────────────────────────────────────────────

    def get_latest(self, n_samples: int) -> np.ndarray | None:
        """Return the most recent `n_samples` as (n_channels, n_samples).

        Returns None if not enough data has been collected yet.
        """
        with self._lock:
            available = min(self._write_pos, self.max_samples)
            if available < n_samples:
                return None

            end = self._write_pos % self.max_samples
            start = end - n_samples

            if start >= 0:
                return self._buf[:, start:end].copy()
            else:
                # Wraps around the ring
                return np.hstack([
                    self._buf[:, start % self.max_samples :],
                    self._buf[:, :end],
                ]).copy()

    def get_mi_window(self) -> np.ndarray | None:
        """Convenience: latest MI-length window (2 s = 512 samples)."""
        return self.get_latest(MI_WINDOW_SAMPLES)

    def get_blink_window(self) -> np.ndarray | None:
        """Convenience: latest blink-length window (0.5 s = 128 samples)."""
        return self.get_latest(BLINK_WINDOW_SAMPLES)

    def has_enough(self, n_samples: int) -> bool:
        """Check if at least `n_samples` are available."""
        return min(self._write_pos, self.max_samples) >= n_samples

    def ready_for_mi(self) -> bool:
        return self.has_enough(MI_WINDOW_SAMPLES)

    def ready_for_blink(self) -> bool:
        return self.has_enough(BLINK_WINDOW_SAMPLES)

    def clear(self):
        with self._lock:
            self._buf[:] = 0.0
            self._write_pos = 0
