"""
Noise Suppression Module (Improvements #14, #15).

Implements two classic noise reduction techniques:

  #14 — Real-Time Spectral Subtraction:
    Estimates noise from low-energy segments via exponential moving average (EMA),
    subtracts the noise magnitude spectrum from the signal magnitude spectrum per
    FFT frame, then reconstructs with overlap-add (OLA).
    Based on the SoheilGtex/Active-Noise-Cancelling reference implementation.

  #15 — Wiener Filter:
    Computes the optimal linear MSE-estimate gain per frequency bin:
      H(f) = S_signal(f) / (S_signal(f) + S_noise(f))
    Uses scipy.signal.wiener for a fast single-pass enhancement.
    Assumes stationary Gaussian noise (good for fan hum, power supply interference).

Both methods are applied BEFORE frequency shifting and spectrogram display,
so the spectrogram shows cleaned audio and the shifted output is noise-reduced.

Usage:
    suppressor = NoiseSuppressor(method="spectral_subtraction")
    cleaned = suppressor.process(audio, sample_rate)
"""

import numpy as np
from scipy import signal as scipy_signal
from enum import Enum


class NoiseMethod(Enum):
    """Available noise suppression methods."""
    SPECTRAL_SUBTRACTION = "spectral_subtraction"
    WIENER = "wiener"
    SPECTRAL_SUBTRACTION_WIENER = "combined"  # Spectral subtraction then Wiener


class NoiseSuppressor:
    """
    Real-time-capable noise suppression for ultrasonic audio.

    The spectral subtraction method processes audio in overlapping frames:
      1. Split audio into 20ms frames with 50% overlap
      2. Apply Hann window, compute FFT
      3. Track noise floor via EMA during low-energy frames
      4. Subtract β × noise magnitude from signal magnitude (with flooring)
      5. Reconstruct via inverse FFT + overlap-add

    The Wiener method is simpler: a single-pass optimal filter that works
    well for stationary noise but cannot adapt to changing noise floors.

    Attributes:
        _method:       Which suppression algorithm to use.
        _frame_size:   Frame size in samples (20ms default).
        _hop_size:     Hop size (50% overlap → frame_size / 2).
        _beta:         Spectral subtraction over-subtraction factor (1.0–4.0).
            Higher = more aggressive noise removal but more musical noise.
        _alpha:        EMA smoothing factor for noise floor tracking (0.0–1.0).
            Lower = slower adaptation (more stable), higher = faster adaptation.
        _noise_floor:  Current noise floor estimate (magnitude spectrum).
        _first_update: Whether the noise floor has been initialized.
    """

    def __init__(self, method: NoiseMethod = NoiseMethod.SPECTRAL_SUBTRACTION,
                 frame_size_ms: float = 20.0, beta: float = 2.0, alpha: float = 0.05):
        """
        Args:
            method:       Noise suppression method.
            frame_size_ms: Frame size in milliseconds (20ms is standard for speech/bioacoustics).
            beta:         Over-subtraction factor for spectral subtraction.
            alpha:        EMA smoothing factor for noise floor tracking.
        """
        self._method = method
        self._frame_size_ms = frame_size_ms
        self._beta = beta
        self._alpha = alpha
        self._noise_floor: np.ndarray | None = None
        self._first_update = True

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply noise suppression to an audio buffer.

        Args:
            audio:        1D float32 array of audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            Cleaned audio (same shape and dtype as input).
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return audio

        if self._method == NoiseMethod.WIENER:
            return self._wiener_filter(audio, sample_rate)
        elif self._method == NoiseMethod.SPECTRAL_SUBTRACTION:
            return self._spectral_subtraction(audio, sample_rate)
        elif self._method == NoiseMethod.SPECTRAL_SUBTRACTION_WIENER:
            # Apply spectral subtraction first, then Wiener for final cleanup.
            stage1 = self._spectral_subtraction(audio, sample_rate)
            return self._wiener_filter(stage1, sample_rate)
        else:
            return audio

    # -----------------------------------------------------------------------
    # Method 1: Spectral Subtraction (Improvement #14)
    # -----------------------------------------------------------------------

    def _spectral_subtraction(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Real-time spectral subtraction with EMA noise tracking.

        Algorithm (per frame):
          1. Apply Hann window to the frame
          2. Compute FFT (magnitude spectrum)
          3. If frame energy is low → update noise floor via EMA
          4. Subtract β × noise_floor from signal magnitude (with flooring)
          5. Reconstruct with original phase + inverse FFT + overlap-add

        The flooring step (max(result, floor × noise)) prevents musical noise
        artifacts that occur when subtraction drives bins to near-zero.

        Args:
            audio:        1D float32 audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            Noise-suppressed audio.
        """
        frame_size = int(sample_rate * self._frame_size_ms / 1000.0)
        # Ensure frame size is even for 50% overlap.
        if frame_size % 2 != 0:
            frame_size += 1
        hop_size = frame_size // 2

        window = np.hanning(frame_size)
        n_frames = max(0, (len(audio) - frame_size) // hop_size + 1)

        if n_frames == 0:
            return audio

        # Output buffer (same length as input).
        output = np.zeros_like(audio, dtype="float64")
        # Overlap-add normalization: sum of applied windows at each sample.
        norm = np.zeros_like(audio, dtype="float64")

        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            if end > len(audio):
                end = len(audio)
                frame = np.zeros(frame_size, dtype="float64")
                frame[:end - start] = audio[start:end]
            else:
                frame = audio[start:end].astype("float64")

            # Apply window and compute FFT.
            windowed = frame * window
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # --- Noise floor tracking ---
            frame_energy = np.mean(magnitude)
            if self._noise_floor is None:
                # Initialize noise floor with the first frame.
                self._noise_floor = magnitude.copy()
                self._first_update = False
            elif frame_energy < np.mean(self._noise_floor) * 1.5:
                # Low-energy frame → update noise floor via EMA.
                self._noise_floor = (
                    self._alpha * magnitude + (1 - self._alpha) * self._noise_floor
                )

            # --- Spectral subtraction ---
            if self._noise_floor is not None:
                # Subtract β × noise magnitude, with flooring to prevent artifacts.
                floor = 0.01 * self._noise_floor  # Prevent complete silence.
                cleaned_mag = np.maximum(
                    magnitude - self._beta * self._noise_floor,
                    floor,
                )
            else:
                cleaned_mag = magnitude

            # Reconstruct with original phase.
            cleaned_spectrum = cleaned_mag * np.exp(1j * phase)
            cleaned_frame = np.fft.irfft(cleaned_spectrum, n=frame_size)

            # Overlap-add: accumulate windowed output and window sum.
            end_idx = min(end, len(audio))
            output[start:end_idx] += cleaned_frame[:end_idx - start] * window[:end_idx - start]
            norm[start:end_idx] += window[:end_idx - start] ** 2

        # Normalize by the squared window sum (standard COLA normalization).
        norm[norm < 1e-8] = 1e-8
        output = output / norm

        # Clip to [-1, 1] to prevent artifacts from the subtraction.
        output = np.clip(output, -1.0, 1.0)

        return output.astype("float32")

    # -----------------------------------------------------------------------
    # Method 2: Wiener Filter (Improvement #15)
    # -----------------------------------------------------------------------

    def _wiener_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply a Wiener filter for noise reduction.

        The Wiener filter computes the optimal linear MSE-estimate gain:
          H(f) = S_signal(f) / (S_signal(f) + S_noise(f))

        scipy.signal.wiener implements this by estimating the signal spectrum
        from a local mean and the noise power from the local variance.

        This method is fast (single-pass), produces no musical noise, and works
        well for stationary noise like fan hum or power supply interference.
        However, it can sound slightly muffled if over-applied.

        Args:
            audio:        1D float32 audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            Wiener-filtered audio.
        """
        # Estimate noise power from a short segment at the start (assumed silence).
        noise_len = min(int(sample_rate * 0.05), len(audio))  # 50ms noise estimate
        if noise_len > 0:
            noise_segment = audio[:noise_len]
            noise_power = float(np.mean(noise_segment ** 2))
        else:
            noise_power = 0.01

        # Choose mysize based on frame size (must be odd for scipy.signal.wiener).
        mysize = int(sample_rate * self._frame_size_ms / 1000.0)
        if mysize % 2 == 0:
            mysize += 1
        mysize = max(3, min(mysize, 1025))  # Clamp to reasonable range.

        # Apply Wiener filter.
        # scipy.signal.wiener estimates signal+noise locally and applies the
        # optimal gain per sample. The mysize parameter controls the local
        # window size — larger = smoother but less adaptive.
        filtered = scipy_signal.wiener(audio.astype("float64"), mysize=mysize)
        return filtered.astype("float32")

    # -----------------------------------------------------------------------
    # Real-time streaming interface (for live mic mode)
    # -----------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Process a single frame for real-time (streaming) noise suppression.

        Used by the live microphone mode. Maintains noise floor state across
        frames. Only spectral subtraction supports true streaming; Wiener
        requires the full buffer.

        Args:
            frame:        1D float32 audio frame (one block from the mic).
            sample_rate:  Sample rate in Hz.

        Returns:
            Noise-suppressed frame.
        """
        if self._method == NoiseMethod.WIENER:
            # Wiener can process short frames but works best on longer segments.
            mysize = min(255, len(frame) if len(frame) % 2 == 1 else len(frame) - 1)
            mysize = max(3, mysize)
            return scipy_signal.wiener(frame.astype("float64"), mysize=mysize).astype("float32")
        else:
            # For spectral subtraction in streaming mode, process the frame
            # as a mini-buffer.
            return self._spectral_subtraction(frame, sample_rate)

    def reset(self) -> None:
        """Reset the noise floor estimate (e.g., when switching audio sources)."""
        self._noise_floor = None
        self._first_update = True

    @property
    def method(self) -> NoiseMethod:
        return self._method

    @method.setter
    def method(self, value: NoiseMethod) -> None:
        self._method = value
        self.reset()