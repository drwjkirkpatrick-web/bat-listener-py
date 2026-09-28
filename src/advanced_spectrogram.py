"""
Advanced Spectrogram Techniques (Improvements #12, #13, #20, #5, #6, #7).

This module provides multiple spectrogram rendering modes beyond the basic
STFT used in the original spectrogram.py:

  #12 — PCEN (Per-Channel Energy Normalization):
    Adaptive gain control that replaces static log compression. Tracks
    stationary background noise per frequency band via IIR smoothing, then
    applies dynamic range compression. Reduces false alarm rates 5–50×
    compared to log-mel spectrograms in bioacoustic detection tasks.
    Reference: Lostanlen et al., IEEE 2018. Params: T=60ms, α=0.8, δ=10, r=0.25.

  #13 — Constant-Q Transform (CQT):
    Uses geometrically-spaced frequency bins (Q = f/Δf is constant). Frequency
    resolution increases at low frequencies and time resolution improves at
    high frequencies. Ideal for bat calls spanning 15–120 kHz with narrowband
    CF components. Reference: librosa.cqt.

  #20 — Mel-Scale Spectrogram:
    Compresses high frequencies (where bat calls are densely packed) into fewer
    bins while preserving low-frequency detail. 128 mel bands between 15–120 kHz.
    Reference: librosa.feature.melspectrogram.

  #5 — Continuous Wavelet Transform (CWT):
    Multiresolution analysis matching bat cochlear processing. Morlet wavelets
    provide sharper ridges for fast FM sweeps than fixed-window STFT.
    Reference: Lin, JASA 1992 (Purdue University).

  #6 — Wigner-Ville Distribution (WVD):
    Time-frequency representation with no windowing artifact. Higher
    concentration for FM signals than STFT. Minimizes FM slope estimation
    variance. Reference: Herman, Archives of Acoustics 2008 (Wrocław UT).

  #7 — Cepstrogram:
    Windowed short-time cepstral analysis. Resolves sub-microsecond echo
    fine structure ("glints") below auditory temporal resolution.
    Reference: Buck & Simmons, JASA (UMass Dartmouth/Brown).

Each method returns a 2D magnitude array (frequency bins × time frames)
that can be rendered by the spectrogram canvas renderer.
"""

import numpy as np
from scipy import signal as scipy_signal
from enum import Enum


class SpectrogramMode(Enum):
    """Available spectrogram rendering modes."""
    STFT_LINEAR = "stft_linear"       # Standard STFT (linear frequency)
    STFT_LOG = "stft_log"             # STFT with log-frequency axis
    PCEN = "pcen"                      # Per-Channel Energy Normalization
    CQT = "cqt"                        # Constant-Q Transform
    MEL = "mel"                        # Mel-scale spectrogram
    WAVELET = "wavelet"                # Continuous Wavelet Transform (Morlet)
    WIGNER_VILLE = "wigner_ville"     # Wigner-Ville Distribution
    CEPSTROGRAM = "cepstrogram"        # Short-time cepstral analysis


class AdvancedSpectrogram:
    """
    Multi-mode spectrogram processor.

    Computes spectrograms using various time-frequency representations.
    Each mode is selected by the user and rendered on the canvas.

    The class operates on numpy audio arrays and returns 2D magnitude
    arrays suitable for the spectrogram renderer.
    """

    def __init__(self, fft_size: int = 2048, hop_size: int = 512):
        """
        Args:
            fft_size:  FFT window size for STFT-based modes.
            hop_size:  STFT hop size (samples between frames).
        """
        self._fft_size = fft_size
        self._hop_size = hop_size

        # PCEN state (IIR smoother state per frequency band).
        self._pcen_state: np.ndarray | None = None
        self._pcen_T = 0.06    # Time constant (60ms)
        self._pcen_alpha = 0.8  # Gain normalization exponent
        self._pcen_delta = 10.0  # Bias / floor
        self._pcen_r = 0.25      # Dynamic range compression

    def compute(self, audio: np.ndarray, sample_rate: int,
                mode: SpectrogramMode) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute a spectrogram in the specified mode.

        Args:
            audio:        1D float32 audio samples.
            sample_rate:  Sample rate in Hz.
            mode:         Which spectrogram mode to compute.

        Returns:
            (magnitude, frequencies) where:
              magnitude: 2D array [freq_bins × time_frames], float32, 0–255 scaled.
              frequencies: 1D array of frequency values (Hz) for each bin.
        """
        if audio is None or len(audio) == 0 or sample_rate <= 0:
            return np.zeros((1, 1), dtype="float32"), np.array([0.0])

        if mode == SpectrogramMode.STFT_LINEAR:
            return self._compute_stft(audio, sample_rate, log_scale=False)
        elif mode == SpectrogramMode.STFT_LOG:
            return self._compute_stft(audio, sample_rate, log_scale=True)
        elif mode == SpectrogramMode.PCEN:
            return self._compute_pcen(audio, sample_rate)
        elif mode == SpectrogramMode.CQT:
            return self._compute_cqt(audio, sample_rate)
        elif mode == SpectrogramMode.MEL:
            return self._compute_mel(audio, sample_rate)
        elif mode == SpectrogramMode.WAVELET:
            return self._compute_wavelet(audio, sample_rate)
        elif mode == SpectrogramMode.WIGNER_VILLE:
            return self._compute_wigner_ville(audio, sample_rate)
        elif mode == SpectrogramMode.CEPSTROGRAM:
            return self._compute_cepstrogram(audio, sample_rate)
        else:
            return self._compute_stft(audio, sample_rate, log_scale=False)

    # -----------------------------------------------------------------------
    # STFT (Linear and Log-frequency)
    # -----------------------------------------------------------------------

    def _compute_stft(self, audio: np.ndarray, sample_rate: int,
                      log_scale: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard Short-Time Fourier Transform spectrogram.

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.
            log_scale:    If True, remap frequency axis to logarithmic spacing.

        Returns:
            (magnitude_0_255, frequency_axis)
        """
        window = np.hanning(self._fft_size)
        freqs, times, sxx = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=self._fft_size,
            noverlap=self._fft_size - self._hop_size,
            mode="magnitude",
        )

        if log_scale:
            # Remap to log-spaced frequency bins.
            # Create log-spaced bins from 1 kHz to Nyquist.
            nyquist = sample_rate / 2.0
            n_bins = len(freqs)
            log_freqs = np.logspace(
                np.log10(max(1000, freqs[1])),
                np.log10(nyquist),
                n_bins,
            )
            # Interpolate each time frame to log-spaced bins.
            log_sxx = np.zeros((n_bins, len(times)))
            for t in range(len(times)):
                log_sxx[:, t] = np.interp(log_freqs, freqs, sxx[:, t])
            freqs = log_freqs
            sxx = log_sxx

        # Scale to 0–255.
        max_val = sxx.max() if sxx.max() > 0 else 1.0
        scaled = (sxx / max_val * 255).astype("float32")
        return scaled, freqs

    # -----------------------------------------------------------------------
    # PCEN (Improvement #12)
    # -----------------------------------------------------------------------

    def _compute_pcen(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-Channel Energy Normalization (PCEN).

        PCEN replaces static log compression with adaptive gain control:
          1. Compute STFT magnitude
          2. For each frequency band, track the smoothed energy via IIR filter:
             S[t] = (1 - α) * S[t-1] + α * |X[t]|²  (where α = 1/T)
          3. Apply gain normalization: G = (S / (ε + S[t]))^α
          4. Apply dynamic range compression: PCEN = (G * |X|² + δ)^r - δ^r

        This normalizes for varying signal levels (distance from bat, mic
        sensitivity drift), making faint calls more visible while suppressing
        steady background noise.

        Reference: Lostanlen et al., "Per-Channel Energy Normalization: Why
        and How", IEEE 2018.

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (pcen_magnitude_0_255, frequency_axis)
        """
        # Step 1: Compute STFT magnitude squared (power).
        window = np.hanning(self._fft_size)
        freqs, times, sxx = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=self._fft_size,
            noverlap=self._fft_size - self._hop_size,
            mode="magnitude",
        )
        power = sxx ** 2  # |X|²

        n_bins, n_frames = power.shape

        # Step 2: IIR smoothing per frequency band.
        # α = hop_duration / T  (where T = time constant in seconds)
        hop_duration = self._hop_size / sample_rate
        alpha = hop_duration / self._pcen_T
        alpha = min(alpha, 1.0)  # Clamp for stability.

        if self._pcen_state is None or len(self._pcen_state) != n_bins:
            self._pcen_state = np.zeros(n_bins, dtype="float64")

        smoothed = np.zeros_like(power, dtype="float64")
        for t in range(n_frames):
            self._pcen_state = (
                (1 - alpha) * self._pcen_state + alpha * power[:, t]
            )
            smoothed[:, t] = self._pcen_state

        # Step 3: Gain normalization.
        eps = 1e-6
        gain = (smoothed / (eps + smoothed)) ** self._pcen_alpha

        # Step 4: Dynamic range compression.
        pcen = (gain * power + self._pcen_delta) ** self._pcen_r - self._pcen_delta ** self._pcen_r

        # Scale to 0–255.
        max_val = pcen.max() if pcen.max() > 0 else 1.0
        scaled = (np.abs(pcen) / max_val * 255).astype("float32")
        return scaled, freqs

    # -----------------------------------------------------------------------
    # Constant-Q Transform (Improvement #13)
    # -----------------------------------------------------------------------

    def _compute_cqt(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Constant-Q Transform (CQT).

        The CQT uses geometrically-spaced frequency bins where Q = f / Δf is
        constant. This means:
          - Low frequencies: narrow bandwidth → fine frequency resolution
          - High frequencies: wide bandwidth → fine time resolution

        This is ideal for bat calls:
          - CF calls (horseshoe bats at 80–110 kHz) need fine freq resolution
          - FM sweeps need good time resolution at high frequencies

        Implementation: We approximate the CQT by computing a high-resolution
        STFT and then aggregating bins into log-spaced bands. A true CQT uses
        a different kernel, but this approximation is visually equivalent for
        display purposes.

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (cqt_magnitude_0_255, log_spaced_frequencies)
        """
        # Compute a high-resolution STFT.
        fft_size = max(self._fft_size, 4096)
        window = np.hanning(fft_size)
        freqs, times, sxx = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=fft_size,
            noverlap=fft_size - self._hop_size,
            mode="magnitude",
        )

        # Define log-spaced frequency bins from 15 kHz to Nyquist.
        nyquist = sample_rate / 2.0
        f_min = max(15000, freqs[1])  # 15 kHz minimum
        f_max = min(nyquist, 120000)  # 120 kHz maximum
        if f_min >= f_max:
            f_min = freqs[1]
            f_max = nyquist

        # 12 bins per octave, 6 octaves from 15 kHz → ~960 kHz.
        n_octaves = np.log2(f_max / f_min)
        n_bins = max(24, int(n_octaves * 12))
        log_freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)

        # Aggregate STFT bins into log-spaced bands.
        cqt_sxx = np.zeros((n_bins, len(times)), dtype="float64")
        for i in range(n_bins):
            # Find STFT bins within this CQT band.
            band_low = log_freqs[i] * (2 ** (-1 / 24))  # Half-step below
            band_high = log_freqs[i] * (2 ** (1 / 24))   # Half-step above
            mask = (freqs >= band_low) & (freqs <= band_high)
            if np.any(mask):
                cqt_sxx[i, :] = np.max(sxx[mask, :], axis=0)

        # Scale to 0–255.
        max_val = cqt_sxx.max() if cqt_sxx.max() > 0 else 1.0
        scaled = (cqt_sxx / max_val * 255).astype("float32")
        return scaled, log_freqs

    # -----------------------------------------------------------------------
    # Mel-Scale Spectrogram (Improvement #20)
    # -----------------------------------------------------------------------

    def _compute_mel(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Mel-scale spectrogram.

        The mel scale compresses high frequencies (where bat calls are densely
        packed) into fewer bins while preserving low-frequency detail. This
        produces a perceptually-weighted spectrogram that better matches how
        bat call energy is distributed.

        Implementation: Compute STFT, then apply a mel filterbank (triangular
        filters spaced on the mel scale) to aggregate linear frequency bins
        into mel bands.

        128 mel bands between 15–120 kHz, as recommended for bioacoustic
        deep learning models.

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (mel_magnitude_0_255, mel_frequencies)
        """
        # Compute STFT.
        window = np.hanning(self._fft_size)
        freqs, times, sxx = scipy_signal.spectrogram(
            audio,
            fs=sample_rate,
            window=window,
            nperseg=self._fft_size,
            noverlap=self._fft_size - self._hop_size,
            mode="magnitude",
        )

        # Build mel filterbank.
        n_mels = 128
        f_min = 15000  # 15 kHz
        f_max = min(sample_rate / 2.0, 120000)  # 120 kHz or Nyquist

        mel_freqs = self._mel_filterbank_freqs(n_mels, f_min, f_max)
        filterbank = self._build_mel_filterbank(
            n_mels, f_min, f_max, freqs
        )

        # Apply filterbank: mel_sxx = filterbank @ sxx
        mel_sxx = filterbank @ sxx

        # Scale to 0–255.
        max_val = mel_sxx.max() if mel_sxx.max() > 0 else 1.0
        scaled = (mel_sxx / max_val * 255).astype("float32")
        return scaled, mel_freqs

    def _mel(self, freq_hz: float) -> float:
        """Convert frequency in Hz to mel scale."""
        return 2595.0 * np.log10(1.0 + freq_hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel scale to frequency in Hz."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _mel_filterbank_freqs(self, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
        """Compute the center frequencies of mel filterbank bands."""
        mel_min = self._mel(f_min)
        mel_max = self._mel(f_max)
        mel_points = np.linspace(mel_min, mel_max, n_mels)
        return np.array([self._mel_to_hz(m) for m in mel_points])

    def _build_mel_filterbank(self, n_mels: int, f_min: float, f_max: float,
                               freqs: np.ndarray) -> np.ndarray:
        """
        Build a triangular mel filterbank matrix.

        Each row is a triangular filter centered on a mel frequency.
        The matrix shape is [n_mels × n_freq_bins].

        Args:
            n_mels:  Number of mel bands.
            f_min:   Minimum frequency (Hz).
            f_max:   Maximum frequency (Hz).
            freqs:   STFT frequency axis (Hz).

        Returns:
            Filterbank matrix [n_mels × len(freqs)].
        """
        # Mel center frequencies, with edges for triangular filters.
        mel_min = self._mel(f_min)
        mel_max = self._mel(f_max)
        # n_mels + 2 points to get n_mels triangular filters.
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])

        n_freqs = len(freqs)
        filterbank = np.zeros((n_mels, n_freqs), dtype="float64")

        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            # Triangular filter: ramp up from left to center, down to right.
            for j in range(n_freqs):
                f = freqs[j]
                if left <= f <= center:
                    filterbank[i, j] = (f - left) / max(1e-6, center - left)
                elif center < f <= right:
                    filterbank[i, j] = (right - f) / max(1e-6, right - center)

        return filterbank

    # -----------------------------------------------------------------------
    # Continuous Wavelet Transform (Improvement #5)
    # -----------------------------------------------------------------------

    def _compute_wavelet(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Continuous Wavelet Transform with Morlet wavelets.

        The CWT provides multiresolution analysis:
          - At low frequencies: wide wavelet → fine frequency resolution
          - At high frequencies: narrow wavelet → fine time resolution

        This matches the bat cochlear filter bank structure, where different
        regions respond to different frequency ranges with different time/freq
        trade-offs.

        We use the complex Morlet wavelet:
          ψ(t) = exp(iω₀t) * exp(-t²/2)

        where ω₀ controls the number of oscillations (default 5, giving a
        good time-frequency trade-off).

        Reference: Lin, "Some aspects of wavelet transform in biosonar",
        J. Acoust. Soc. Am. 1992 (Purdue University).

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (cwt_magnitude_0_255, wavelet_frequencies)
        """
        # Define frequency range: 15 kHz to Nyquist, or min detectable.
        nyquist = sample_rate / 2.0
        f_min = max(15000, 1000)
        f_max = min(nyquist, 120000)
        if f_min >= f_max:
            return np.zeros((1, 1), dtype="float32"), np.array([0.0])

        # Log-spaced frequencies for the wavelet scales.
        n_scales = 64
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), n_scales)

        # For each frequency, compute the CWT at that scale.
        # Morlet wavelet: ψ(t) = exp(i*2π*f*t) * exp(-t²/(2*σ²))
        # where σ = number_of_cycles / (2π * f)
        omega0 = 5.0  # Number of cycles (time-frequency trade-off)
        dt = 1.0 / sample_rate
        n_samples = len(audio)

        cwt_matrix = np.zeros((n_scales, n_samples), dtype="float64")

        # Process in chunks if the audio is long (memory management).
        max_chunk = 65536
        n_chunks = (n_samples + max_chunk - 1) // max_chunk

        for i, f in enumerate(freqs):
            sigma = omega0 / (2.0 * np.pi * f)
            # Wavelet support: ±3σ.
            support = int(3 * sigma / dt)
            t_wavelet = np.arange(-support, support + 1) * dt
            # Complex Morlet wavelet.
            wavelet = (
                np.exp(1j * 2 * np.pi * f * t_wavelet)
                * np.exp(-t_wavelet ** 2 / (2 * sigma ** 2))
            )
            # Normalize.
            wavelet = wavelet / np.sqrt(np.sum(np.abs(wavelet) ** 2))

            # Convolve (use FFT for speed).
            conv = scipy_signal.fftconvolve(audio.astype("float64"), wavelet, mode="same")
            cwt_matrix[i, :] = np.abs(conv)

        # Subsample time axis to a manageable size for display.
        n_display = min(400, n_samples)
        step = max(1, n_samples // n_display)
        cwt_display = cwt_matrix[:, ::step]

        # Scale to 0–255.
        max_val = cwt_display.max() if cwt_display.max() > 0 else 1.0
        scaled = (cwt_display / max_val * 255).astype("float32")
        return scaled, freqs

    # -----------------------------------------------------------------------
    # Wigner-Ville Distribution (Improvement #6)
    # -----------------------------------------------------------------------

    def _compute_wigner_ville(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Wigner-Ville Distribution (WVD).

        The WVD is a quadratic time-frequency representation with maximal
        concentration for FM signals. Unlike the STFT, it has no windowing
        artifact — the time-frequency resolution is not limited by a
        time-bandwidth product.

        Definition:
          W(t, ω) = ∫ x(t + τ/2) * x*(t - τ/2) * e^(-iωτ) dτ

        For discrete signals, this becomes a sum over lag τ.

        The WVD has cross-terms (interference patterns between multiple
        components), which can be reduced using the pseudo-Wigner-Ville
        distribution (windowed in lag).

        Reference: Herman, "Analysis of bat echolocation signals using
        time-frequency distributions", Archives of Acoustics 2008
        (Wrocław University of Technology).

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (wvd_magnitude_0_255, frequencies)
        """
        # Limit the audio length (WVD is O(N²) — expensive).
        max_samples = 8192
        if len(audio) > max_samples:
            # Take a representative segment.
            audio = audio[:max_samples].copy()

        n = len(audio)
        # Use the analytic signal (Hilbert transform) to remove negative
        # frequencies and avoid cross-terms with the conjugate.
        analytic = scipy_signal.hilbert(audio.astype("float64"))

        # Lag range (window for pseudo-Wigner-Ville).
        lag_max = min(256, n // 2)
        freq_bins = lag_max  # Frequency resolution = lag_max

        wvd = np.zeros((freq_bins, n), dtype="float64")

        for t in range(n):
            # Compute the lag window centered at t.
            for lag in range(-lag_max // 2, lag_max // 2):
                t1 = t + lag
                t2 = t - lag
                if 0 <= t1 < n and 0 <= t2 < n:
                    # Kernel: x(t+τ/2) * x*(t-τ/2)
                    kernel = analytic[t1] * np.conj(analytic[t2])
                    # Accumulate into the WVD.
                    freq_idx = lag + lag_max // 2
                    if 0 <= freq_idx < freq_bins:
                        wvd[freq_idx, t] = np.real(kernel)

            # FFT over lag to get frequency content at time t.
            wvd[:, t] = np.abs(np.fft.fft(wvd[:, t]))

        # Frequency axis.
        freqs = np.fft.fftfreq(freq_bins, d=1.0 / sample_rate)[:freq_bins // 2]
        # Take only positive frequencies.
        wvd = wvd[:freq_bins // 2, :]

        # Subsample time for display.
        n_display = min(400, n)
        step = max(1, n // n_display)
        wvd_display = wvd[:, ::step]

        # Scale to 0–255.
        max_val = wvd_display.max() if wvd_display.max() > 0 else 1.0
        scaled = (wvd_display / max_val * 255).astype("float32")
        return scaled, np.abs(freqs)

    # -----------------------------------------------------------------------
    # Cepstrogram (Improvement #7)
    # -----------------------------------------------------------------------

    def _compute_cepstrogram(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Short-time cepstral analysis (cepstrogram).

        The cepstrum is the inverse Fourier transform of the log magnitude
        spectrum. It separates source (vocal tract / echo glints) from
        filter (environmental reflections).

        C_m = IFFT(log(|FFT(x)|))

        The cepstrogram shows cepstral coefficients over time. Peaks in the
        cepstrum correspond to echo "glints" (closely-spaced reflecting
        surfaces) that are below the auditory temporal resolution of the
        spectrogram.

        For bat calls, the cepstrogram can reveal:
          - Glint spacing (distance between echo-producing structures)
          - Call periodicity (pulse repetition)
          - Harmonic structure (separate from formant-like filtering)

        Reference: Buck & Simmons, "A biologically-inspired sensor for
        echo processing", JASA (UMass Dartmouth/Brown University).

        Args:
            audio:        1D audio samples.
            sample_rate:  Sample rate in Hz.

        Returns:
            (cepstrum_magnitude_0_255, quefrency_axis)
        """
        frame_size = self._fft_size
        hop = self._hop_size
        window = np.hanning(frame_size)
        n_frames = max(0, (len(audio) - frame_size) // hop + 1)

        if n_frames == 0:
            return np.zeros((1, 1), dtype="float32"), np.array([0.0])

        # We compute half the FFT size for the cepstrum (real signal → symmetric).
        n_cepstrum = frame_size // 2
        cepstrogram = np.zeros((n_cepstrum, n_frames), dtype="float64")

        for i in range(n_frames):
            start = i * hop
            frame = audio[start:start + frame_size]
            if len(frame) < frame_size:
                frame = np.pad(frame, (0, frame_size - len(frame)))

            # Window and FFT.
            windowed = frame * window
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum) + 1e-10  # Avoid log(0)

            # Log magnitude → cepstrum.
            log_mag = np.log(magnitude)
            cepstrum = np.fft.irfft(log_mag, n=frame_size)

            # Take the first n_cepstrum coefficients.
            cepstrogram[:, i] = cepstrum[:n_cepstrum]

        # Quefrency axis (in samples, convertible to ms: 1 sample = 1/sr seconds).
        quefrency = np.arange(n_cepstrum) / sample_rate * 1000  # ms

        # Scale to 0–255.
        max_val = cepstrogram.max() if cepstrogram.max() > 0 else 1.0
        scaled = (np.abs(cepstrogram) / max_val * 255).astype("float32")
        return scaled, quefrency

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Reset stateful parameters (e.g., PCEN smoother)."""
        self._pcen_state = None