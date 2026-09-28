"""
Bat Call Parameter Extraction (Improvement #16).

Extracts standard quantitative parameters from bat echolocation calls using
spectrogram analysis. These parameters — defined by SonoBat and the Montana
Bat Call Key — are the foundation of manual and automated species identification.

Parameters extracted:
  - Fpeak:  Frequency of greatest power (Hz)
  - Fc:     Characteristic frequency (frequency at the lowest slope point)
  - Fhi:    Highest frequency in the call (Hz)
  - Flo:    Lowest frequency in the call (Hz)
  - Duration: Call duration (ms)
  - Bandwidth: Fhi - Flo (Hz)
  - Sweep slope: Rate of frequency change (kHz/ms) — upper, lower, total
  - IPI:    Inter-pulse interval (ms) — time between consecutive calls

Algorithm:
  1. Compute STFT of the audio with fine time resolution
  2. For each call (detected via energy thresholding), extract the spectrogram
     ridge (peak frequency at each time frame)
  3. From the ridge, compute Fpeak, Fhi, Flo, duration, bandwidth, slope
  4. For IPI: detect call onsets and measure time between them

Sources: SonoBat Western NA Bat Acoustic Table; Montana Bat Call Key
"""

import numpy as np
from scipy import signal as scipy_signal
from dataclasses import dataclass
from typing import List


@dataclass
class CallParameters:
    """
    Standard parameters for a single bat echolocation call.

    All frequencies in Hz, durations in ms, slopes in kHz/ms.
    """
    f_peak: float = 0.0       # Frequency of greatest power (Hz)
    f_characteristic: float = 0.0  # Frequency at lowest slope (Hz)
    f_high: float = 0.0       # Highest frequency (Hz)
    f_low: float = 0.0        # Lowest frequency (Hz)
    duration_ms: float = 0.0  # Call duration (ms)
    bandwidth_hz: float = 0.0  # Fhi - Flo (Hz)
    slope_upper_khz_ms: float = 0.0  # Initial sweep slope
    slope_lower_khz_ms: float = 0.0  # Final sweep slope
    slope_total_khz_ms: float = 0.0  # Overall sweep slope
    start_time_ms: float = 0.0  # When this call starts (ms from audio start)


@dataclass
class CallSequence:
    """
    A sequence of bat calls with inter-pulse intervals.
    """
    calls: List[CallParameters]
    # Inter-pulse intervals between consecutive calls (ms).
    inter_pulse_intervals_ms: List[float]

    @property
    def num_calls(self) -> int:
        return len(self.calls)

    @property
    def mean_ipi_ms(self) -> float:
        """Mean inter-pulse interval (ms)."""
        if not self.inter_pulse_intervals_ms:
            return 0.0
        return float(np.mean(self.inter_pulse_intervals_ms))

    @property
    def mean_duration_ms(self) -> float:
        """Mean call duration (ms)."""
        if not self.calls:
            return 0.0
        return float(np.mean([c.duration_ms for c in self.calls]))


class CallParameterExtractor:
    """
    Extracts bat call parameters from audio using STFT-based ridge tracking.

    The extractor works on a numpy array of audio samples. It:
      1. Computes a high-resolution STFT
      2. Detects individual calls via energy thresholding
      3. For each call, tracks the frequency ridge (peak bin per frame)
      4. Extracts standard parameters from the ridge contour
    """

    def __init__(self, fft_size: int = 256, hop_size: int = 64,
                 energy_threshold: float = 0.15):
        """
        Args:
            fft_size:        FFT window size (small = good time resolution for short calls).
            hop_size:        STFT hop size (samples between frames).
            energy_threshold: Fraction of max energy to use as call detection threshold.
        """
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._energy_threshold = energy_threshold

    def extract(self, audio: np.ndarray, sample_rate: int) -> CallSequence:
        """
        Extract call parameters from an audio buffer.

        Args:
            audio:        1D float32 array of audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            A CallSequence containing all detected calls and their IPIs.
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return CallSequence(calls=[], inter_pulse_intervals_ms=[])

        # --- Step 1: Compute STFT ---
        # Use a Hann window for smooth frequency response.
        window = np.hanning(self._fft_size)
        # scipy spectrogram gives us frequency and time bins.
        freqs, times, sxx = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=self._fft_size,
            noverlap=self._fft_size - self._hop_size,
            mode="magnitude",
        )

        # --- Step 2: Detect calls via energy thresholding ---
        # Compute energy per time frame (sum of magnitudes).
        energy_per_frame = np.sum(sxx, axis=0)
        max_energy = np.max(energy_per_frame) if len(energy_per_frame) > 0 else 0.0
        if max_energy <= 0:
            return CallSequence(calls=[], inter_pulse_intervals_ms=[])

        threshold = max_energy * self._energy_threshold
        is_active = energy_per_frame > threshold

        # Find contiguous active regions (individual calls).
        call_regions = self._find_contiguous_regions(is_active)

        # --- Step 3: Extract parameters for each call ---
        calls = []
        for start_frame, end_frame in call_regions:
            call_params = self._extract_call_parameters(
                sxx, freqs, times, start_frame, end_frame, sample_rate
            )
            calls.append(call_params)

        # --- Step 4: Compute inter-pulse intervals ---
        ipis = []
        for i in range(1, len(calls)):
            # IPI = time between end of call[i-1] and start of call[i].
            prev_end = calls[i - 1].start_time_ms + calls[i - 1].duration_ms
            curr_start = calls[i].start_time_ms
            ipi = curr_start - prev_end
            if ipi > 0:  # Ignore overlaps (shouldn't happen but guard).
                ipis.append(ipi)

        return CallSequence(calls=calls, inter_pulse_intervals_ms=ipis)

    def _find_contiguous_regions(self, is_active: np.ndarray) -> list[tuple[int, int]]:
        """
        Find contiguous True regions in a boolean array.

        Args:
            is_active: 1D boolean array (True = call present).

        Returns:
            List of (start_frame, end_frame) tuples for each contiguous region.
        """
        regions = []
        in_region = False
        start = 0

        for i, active in enumerate(is_active):
            if active and not in_region:
                # Start of a new call.
                start = i
                in_region = True
            elif not active and in_region:
                # End of a call.
                regions.append((start, i))
                in_region = False

        # Handle call that extends to the end.
        if in_region:
            regions.append((start, len(is_active)))

        # Filter out very short regions (noise / artifacts < 3 frames).
        return [(s, e) for s, e in regions if e - s >= 3]

    def _extract_call_parameters(
        self, sxx: np.ndarray, freqs: np.ndarray, times: np.ndarray,
        start_frame: int, end_frame: int, sample_rate: int
    ) -> CallParameters:
        """
        Extract parameters for a single call from its spectrogram region.

        The ridge (peak frequency at each time frame) is tracked, then
        standard parameters are computed from the ridge contour.

        Args:
            sxx:          Spectrogram magnitude (shape: [freq_bins, time_frames]).
            freqs:        Frequency axis values (Hz).
            times:        Time axis values (s).
            start_frame:  Start frame index of this call.
            end_frame:    End frame index of this call.
            sample_rate:  Audio sample rate (Hz).

        Returns:
            CallParameters for this call.
        """
        # Extract the spectrogram slice for this call.
        call_sxx = sxx[:, start_frame:end_frame]
        call_times = times[start_frame:end_frame]

        if call_sxx.shape[1] == 0:
            return CallParameters()

        # --- Track the frequency ridge ---
        # The ridge is the peak frequency at each time frame.
        ridge_indices = np.argmax(call_sxx, axis=0)
        ridge_freqs = freqs[ridge_indices]

        # --- Fpeak: frequency of greatest power ---
        # Find the time frame with maximum energy, then its peak frequency.
        max_energy_frame = np.argmax(np.sum(call_sxx, axis=0))
        f_peak = ridge_freqs[max_energy_frame]

        # --- Fhi, Flo: highest and lowest frequencies on the ridge ---
        f_high = float(np.max(ridge_freqs))
        f_low = float(np.min(ridge_freqs))

        # --- Bandwidth ---
        bandwidth = f_high - f_low

        # --- Duration ---
        if len(call_times) >= 2:
            duration_s = call_times[-1] - call_times[0]
        else:
            duration_s = self._fft_size / sample_rate
        duration_ms = duration_s * 1000.0

        # --- Sweep slopes ---
        # Upper slope: slope of the first 25% of the call.
        # Lower slope: slope of the last 25% of the call.
        # Total slope: overall slope from start to end.
        n_frames = len(ridge_freqs)
        if n_frames >= 4:
            quarter = max(1, n_frames // 4)
            # Upper slope (first quarter).
            upper_freqs = ridge_freqs[:quarter]
            upper_times = np.arange(quarter) * (duration_s / n_frames)
            slope_upper = self._compute_slope(upper_freqs, upper_times)

            # Lower slope (last quarter).
            lower_freqs = ridge_freqs[-quarter:]
            lower_times = np.arange(quarter) * (duration_s / n_frames)
            slope_lower = self._compute_slope(lower_freqs, lower_times)

            # Total slope.
            all_times = np.arange(n_frames) * (duration_s / n_frames)
            slope_total = self._compute_slope(ridge_freqs, all_times)
        else:
            slope_upper = slope_lower = slope_total = 0.0

        # --- Fc: characteristic frequency ---
        # The characteristic frequency is the frequency at the lowest slope point
        # (the "flattest" part of the call). We find the frame where the
        # ridge frequency changes the least.
        if n_frames >= 3:
            # Compute the absolute derivative of the ridge.
            ridge_diff = np.abs(np.diff(ridge_freqs))
            # Find the frame with the minimum derivative (flattest point).
            # Use the middle of a 3-frame window for stability.
            min_diff_idx = np.argmin(ridge_diff)
            f_characteristic = ridge_freqs[min_diff_idx]
        else:
            f_characteristic = f_peak

        # --- Start time ---
        start_time_ms = call_times[0] * 1000.0 if len(call_times) > 0 else 0.0

        return CallParameters(
            f_peak=float(f_peak),
            f_characteristic=float(f_characteristic),
            f_high=f_high,
            f_low=f_low,
            duration_ms=duration_ms,
            bandwidth_hz=float(bandwidth),
            slope_upper_khz_ms=slope_upper,
            slope_lower_khz_ms=slope_lower,
            slope_total_khz_ms=slope_total,
            start_time_ms=start_time_ms,
        )

    def _compute_slope(self, freqs: np.ndarray, times_s: np.ndarray) -> float:
        """
        Compute the sweep slope in kHz/ms from frequency and time arrays.

        Uses linear regression: slope = cov(t, f) / var(t).

        Args:
            freqs:  Frequency values (Hz).
            times_s: Time values (seconds).

        Returns:
            Slope in kHz/ms (can be negative for downsweeps).
        """
        if len(freqs) < 2 or np.std(times_s) == 0:
            return 0.0
        # Linear regression slope.
        slope_hz_per_s = np.polyfit(times_s, freqs, 1)[0]
        # Convert: Hz/s → kHz/ms = (Hz/s) / 1e6
        return float(slope_hz_per_s / 1e6)

    def format_parameters(self, seq: CallSequence) -> str:
        """
        Format extracted parameters as a human-readable string for the UI.

        Args:
            seq: A CallSequence from extract().

        Returns:
            Multi-line string with parameters for each call + IPI summary.
        """
        if seq.num_calls == 0:
            return "No calls detected."

        lines = [f"Calls detected: {seq.num_calls}"]
        if seq.inter_pulse_intervals_ms:
            lines.append(f"Mean IPI: {seq.mean_ipi_ms:.1f} ms")
            lines.append(f"Mean duration: {seq.mean_duration_ms:.1f} ms")

        for i, call in enumerate(seq.calls):
            lines.append(f"--- Call {i+1} ---")
            lines.append(f"  Fpeak: {call.f_peak/1000:.1f} kHz")
            lines.append(f"  Fc:    {call.f_characteristic/1000:.1f} kHz")
            lines.append(f"  Fhi:   {call.f_high/1000:.1f} kHz")
            lines.append(f"  Flo:   {call.f_low/1000:.1f} kHz")
            lines.append(f"  Dur:   {call.duration_ms:.2f} ms")
            lines.append(f"  BW:    {call.bandwidth_hz/1000:.1f} kHz")
            lines.append(f"  Slope: {call.slope_total_khz_ms:.2f} kHz/ms")

        return "\n".join(lines)