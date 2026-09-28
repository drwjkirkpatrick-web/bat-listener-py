"""
Recording & Export (Module 6).

This module handles:
  - Recording the shifted (audible) audio output to a buffer
  - Exporting the recording as a WAV file (manual PCM encoding via soundfile)
  - Exporting the spectrogram as a PNG image (via pygame.Surface)

Recording works by capturing audio frames from the AudioEngine's output.
When recording is active, the main loop copies the current output buffer
into a growing list. When the user stops recording, the list is concatenated
into a single numpy array and written to a WAV file.

The spectrogram PNG export uses pygame's image.save function.
"""

import numpy as np
import soundfile as sf
import pygame
from datetime import datetime


class Recorder:
    """
    Handles audio recording and file export.

    Attributes:
        _is_recording:    Whether recording is currently active.
        _recorded_chunks: List of numpy arrays (audio chunks captured during recording).
        _start_time:      Timestamp when recording started (for the timer display).
        _sample_rate:     Sample rate of the recorded audio.
    """

    def __init__(self):
        self._is_recording: bool = False
        self._recorded_chunks: list[np.ndarray] = []
        self._start_time: float = 0.0
        self._sample_rate: int = 44100

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def elapsed_seconds(self) -> float:
        """Elapsed recording time in seconds (for the UI timer display)."""
        if not self._is_recording:
            return 0.0
        import time
        return time.time() - self._start_time

    def start(self, sample_rate: int) -> None:
        """
        Start recording.

        Clears any previous recording and begins capturing audio.

        Args:
            sample_rate: Sample rate of the audio to record.
        """
        self._is_recording = True
        self._recorded_chunks = []
        self._sample_rate = sample_rate
        import time
        self._start_time = time.time()

    def stop(self) -> None:
        """Stop recording. The recorded audio can then be exported."""
        self._is_recording = False

    def capture_chunk(self, chunk: np.ndarray) -> None:
        """
        Capture a chunk of audio data during recording.

        Called from the main loop each time new audio output is available.

        Args:
            chunk: A 1D numpy array of float32 audio samples.
        """
        if self._is_recording and chunk is not None and len(chunk) > 0:
            self._recorded_chunks.append(chunk.copy())

    def export_wav(self, filepath: str) -> bool:
        """
        Export the recorded audio to a WAV file.

        Concatenates all captured chunks into a single array and writes
        it using soundfile. The audio is normalized to prevent clipping.

        Args:
            filepath: Destination file path for the .wav file.

        Returns:
            True if export succeeded, False if no recording is available.
        """
        if len(self._recorded_chunks) == 0:
            return False

        # Concatenate all chunks into one continuous array.
        audio = np.concatenate(self._recorded_chunks)

        # Normalize to -3 dBFS to prevent clipping while maintaining headroom.
        if audio.max() > 0:
            audio = audio / max(abs(audio.max()), abs(audio.min())) * 0.7

        try:
            sf.write(filepath, audio.astype("float32"), self._sample_rate)
            return True
        except Exception as e:
            print(f"[Recorder] Error exporting WAV: {e}")
            return False

    def export_spectrogram_png(self, surface: pygame.Surface, filepath: str) -> bool:
        """
        Export a pygame Surface (the spectrogram) as a PNG image.

        Args:
            surface:   The pygame Surface to export.
            filepath:  Destination file path for the .png file.

        Returns:
            True if export succeeded, False on error.
        """
        try:
            pygame.image.save(surface, filepath)
            return True
        except Exception as e:
            print(f"[Recorder] Error exporting PNG: {e}")
            return False

    def generate_default_filename(self, ext: str) -> str:
        """
        Generate a default filename with a timestamp.

        Args:
            ext: File extension without the dot (e.g. "wav" or "png").

        Returns:
            A filename like "bat_recording_20260928_153022.wav".
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"bat_recording_{timestamp}.{ext}"