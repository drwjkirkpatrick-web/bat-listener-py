"""
Audio Engine — Modules 1, 2, and 5 of the Bat Listener.

This is the core audio processing module. It handles:

  Module 1 — Audio Input & Decoding:
    - Loading WAV files via soundfile
    - Live microphone input via sounddevice (PortAudio)
    - Transport controls: play, pause, stop
    - Audio info: sample rate, duration, channels, Nyquist frequency

  Module 2 — Frequency Shift Engine (3 methods):
    - Time Expansion:  play back at reduced rate (1/factor). Preserves harmonics.
    - Heterodyne:      multiply signal by local oscillator sine, low-pass the
                       difference frequency. Real-time, but destroys harmonic
                       structure.
    - Frequency Division: integer down-conversion (same as time expansion with
                       integer factors).

  Module 5 — Audio Filters & Enhancement:
    - High-pass, low-pass, band-pass (scipy Biquad / IIR filters)
    - Noise gate (threshold-based muting)
    - Volume boost (linear gain)
    - Filter chain: input → highpass → lowpass → bandpass → noise_gate → gain → output

Audio playback uses sounddevice's OutputStream with a callback that reads from
a ring buffer of pre-processed samples. This decouples processing from playback
and allows real-time frequency shifting for the heterodyne method.

Design notes:
  - For time expansion and frequency division, we pre-compute the slowed audio
    by resampling (scipy.signal.resample or simple sample dropping).
  - For heterodyne, we process in real-time within the playback callback:
    multiply incoming samples by sin(2π·f_osc·t), then low-pass filter.
  - Filters are applied as scipy.signal.butter + sosfilt on the audio buffer
    before playback (for file mode) or in real-time chunks (for live mic mode).
"""

import numpy as np
import soundfile as sf
import sounddevice as sd
from scipy import signal as scipy_signal
from dataclasses import dataclass, field
from enum import Enum
import threading
import time


# ---------------------------------------------------------------------------
# Enums and Data Structures
# ---------------------------------------------------------------------------

class ShiftMethod(Enum):
    """The three frequency-shift methods for making ultrasonic audio audible."""
    TIME_EXPANSION = "time_expansion"
    HETERODYNE = "heterodyne"
    FREQUENCY_DIVISION = "frequency_division"


@dataclass
class FilterState:
    """
    State for a single audio filter.
    All filters share this structure: an active toggle + parameters.
    """
    active: bool = False
    # High-pass / low-pass: cutoff in Hz
    # Band-pass: center frequency in Hz + Q factor
    # Noise gate: threshold in dB
    # Gain: linear multiplier (1.0 = unity, 10.0 = 10x boost)
    cutoff: float = 5000.0
    center: float = 45000.0
    q: float = 5.0
    threshold: float = -40.0
    gain: float = 1.0


@dataclass
class AudioInfo:
    """
    Metadata about the currently loaded audio file or live mic stream.
    Displayed on the canvas UI.
    """
    source: str = "none"           # "file", "mic", or "none"
    filename: str = ""
    sample_rate: int = 0           # Hz
    duration: float = 0.0          # seconds
    channels: int = 0
    nyquist: float = 0.0           # Hz (sample_rate / 2)


# ---------------------------------------------------------------------------
# Audio Engine
# ---------------------------------------------------------------------------

class AudioEngine:
    """
    The main audio processing engine.

    Responsibilities:
      - Load WAV files and decode them into numpy arrays
      - Manage live microphone input
      - Apply frequency shifting (3 methods)
      - Apply the filter chain
      - Handle playback via sounddevice OutputStream
      - Provide real-time frequency data for the spectrogram
    """

    def __init__(self):
        # --- Audio data ---
        self._raw_audio: np.ndarray | None = None   # Original decoded audio (mono)
        self._processed_audio: np.ndarray | None = None  # After shift + filters
        self._audio_info = AudioInfo()

        # --- Playback state ---
        self._is_playing = False
        self._playback_position = 0.0  # seconds into the processed audio
        self._playback_sample_idx = 0  # sample index in processed_audio
        self._stream: sd.OutputStream | None = None

        # --- Live mic state ---
        self._mic_stream: sd.InputStream | None = None
        self._mic_active = False
        self._mic_buffer: np.ndarray | None = None  # circular buffer for mic data
        self._mic_write_idx = 0
        self._mic_buffer_size = 0

        # --- Frequency shift settings ---
        self._method = ShiftMethod.TIME_EXPANSION
        self._method_value = 10.0  # expansion factor, oscillator freq (kHz), or divisor

        # --- Filter states ---
        self._filters = {
            "highpass": FilterState(active=False, cutoff=5000),
            "lowpass": FilterState(active=False, cutoff=80000),
            "bandpass": FilterState(active=False, center=45000, q=5),
            "noise_gate": FilterState(active=False, threshold=-40),
            "gain": FilterState(active=True, gain=1.0),
        }

        # --- Real-time frequency data for spectrogram ---
        # Updated by the playback callback; read by the spectrogram renderer.
        self._freq_data: np.ndarray | None = None
        self._fft_size = 2048

        # --- Heterodyne state ---
        self._heterodyne_phase = 0.0  # accumulated phase for oscillator

        # --- Lock for thread-safe access to shared buffers ---
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Module 1: Audio Input & Decoding
    # -----------------------------------------------------------------------

    def load_wav_file(self, filepath: str) -> bool:
        """
        Load a WAV file and decode it into a numpy array.

        The audio is converted to mono (averaging channels) for simpler
        processing. Sample rate and metadata are stored in AudioInfo.

        Args:
            filepath: Path to the .wav file.

        Returns:
            True if loading succeeded, False on error.
        """
        try:
            audio_data, sample_rate = sf.read(filepath, dtype="float32")

            # Convert to mono if multi-channel (average all channels).
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)

            self._raw_audio = audio_data
            self._audio_info = AudioInfo(
                source="file",
                filename=filepath.split("/")[-1],
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate,
                channels=1,
                nyquist=sample_rate / 2.0,
            )

            # Stop any current playback before loading new file.
            self.stop()
            self._process_audio()
            return True

        except Exception as e:
            print(f"[AudioEngine] Error loading WAV file: {e}")
            return False

    def start_live_mic(self) -> bool:
        """
        Start live microphone input.

        Uses sounddevice.InputStream with a callback that fills a circular
        buffer. The Nyquist frequency is determined by the mic's sample rate.

        Returns:
            True if microphone started successfully, False on error.
        """
        try:
            # Use a high sample rate if possible (384 kHz is ideal for bat
            # detection, but most standard mics cap at 48 kHz).
            # We request the device's default sample rate and display the
            # resulting Nyquist limit so the user understands the constraint.
            device_info = sd.query_devices(sd.default.device[0], "input")
            sample_rate = int(device_info["default_samplerate"])

            self._mic_buffer_size = sample_rate * 2  # 2-second circular buffer
            self._mic_buffer = np.zeros(self._mic_buffer_size, dtype="float32")
            self._mic_write_idx = 0

            self._mic_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                callback=self._mic_callback,
            )
            self._mic_stream.start()
            self._mic_active = True

            self._audio_info = AudioInfo(
                source="mic",
                filename="Live Microphone",
                sample_rate=sample_rate,
                duration=0.0,  # Live — no fixed duration
                channels=1,
                nyquist=sample_rate / 2.0,
            )
            return True

        except Exception as e:
            print(f"[AudioEngine] Error starting live mic: {e}")
            return False

    def stop_live_mic(self) -> None:
        """Stop the live microphone stream and clean up."""
        if self._mic_stream is not None:
            self._mic_stream.stop()
            self._mic_stream.close()
            self._mic_stream = None
        self._mic_active = False
        if self._audio_info.source == "mic":
            self._audio_info = AudioInfo()

    def _mic_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        sounddevice InputStream callback.

        Called by PortAudio when new microphone data is available.
        We copy the incoming samples into our circular buffer.

        Args:
            indata:     Input audio data (shape: [frames, channels]).
            frames:     Number of samples in this block.
            time_info:  Timing info (unused).
            status:     PortAudio status flags (unused).
        """
        if not self._mic_active:
            return

        # Convert to mono if needed.
        mono = indata[:, 0] if indata.ndim > 1 else indata

        with self._lock:
            # Write into circular buffer.
            end_idx = self._mic_write_idx + frames
            if end_idx <= self._mic_buffer_size:
                self._mic_buffer[self._mic_write_idx:end_idx] = mono
            else:
                # Wrap around.
                first_part = self._mic_buffer_size - self._mic_write_idx
                self._mic_buffer[self._mic_write_idx:] = mono[:first_part]
                self._mic_buffer[:end_idx - self._mic_buffer_size] = mono[first_part:]
            self._mic_write_idx = (self._mic_write_idx + frames) % self._mic_buffer_size

    # -----------------------------------------------------------------------
    # Module 2: Frequency Shift Engine
    # -----------------------------------------------------------------------

    def _process_audio(self) -> None:
        """
        Process the raw audio through the frequency shift + filter chain.

        For time expansion and frequency division, this pre-computes the
        slowed-down audio buffer. For heterodyne, the raw audio is kept
        and the shift is applied in real-time during playback.

        This method is called whenever:
          - A new file is loaded
          - The shift method or its parameter changes
          - Filter settings change
        """
        if self._raw_audio is None:
            return

        audio = self._raw_audio
        sr = self._audio_info.sample_rate

        if self._method == ShiftMethod.TIME_EXPANSION:
            # Time expansion: slow playback by 1/factor.
            # A 45 kHz call at 10x becomes 4.5 kHz (audible).
            # Duration increases proportionally.
            factor = max(1.0, self._method_value)
            # Resample by inserting zeros + low-pass, or simply repeat samples.
            # Simple approach: repeat each sample 'factor' times (nearest-neighbor).
            # This is equivalent to reducing the playback rate.
            expanded = np.repeat(audio, int(round(factor)))
            audio = expanded
            # Update the effective sample rate for playback.
            # When we play this expanded buffer at the original sample rate,
            # the frequencies are divided by 'factor'.
            self._processed_audio = self._apply_filters(audio, sr)

        elif self._method == ShiftMethod.FREQUENCY_DIVISION:
            # Frequency division: integer down-conversion.
            # Same as time expansion but with integer divisors only (2, 4, 8, 10, 16).
            divisor = max(2, int(round(self._method_value)))
            expanded = np.repeat(audio, divisor)
            audio = expanded
            self._processed_audio = self._apply_filters(audio, sr)

        elif self._method == ShiftMethod.HETERODYNE:
            # Heterodyne: multiply by local oscillator, then low-pass to extract
            # the difference frequency.
            #   output = input * sin(2π · f_osc · t)
            #   Then low-pass at ~8 kHz to keep only the difference tone.
            osc_freq = self._method_value * 1000.0  # kHz → Hz
            t = np.arange(len(audio), dtype="float64") / sr
            mixed = audio * np.sin(2.0 * np.pi * osc_freq * t)
            # Low-pass filter at 8 kHz to extract the audible difference frequency.
            nyq = sr / 2.0
            cutoff = min(8000.0, nyq * 0.9)
            sos = scipy_signal.butter(4, cutoff / nyq, btype="low", output="sos")
            heterodyned = scipy_signal.sosfilt(sos, mixed).astype("float32")
            self._processed_audio = self._apply_filters(heterodyned, sr)

        # Reset playback position.
        self._playback_sample_idx = 0
        self._playback_position = 0.0

    def _apply_filters(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply the filter chain to an audio buffer.

        Chain order: highpass → lowpass → bandpass → noise_gate → gain

        Args:
            audio:        Input audio samples (float32, mono).
            sample_rate:  Sample rate in Hz.

        Returns:
            Filtered audio samples.
        """
        result = audio.copy()
        nyq = sample_rate / 2.0

        # High-pass filter: removes low-frequency noise below cutoff.
        if self._filters["highpass"].active:
            cutoff = min(self._filters["highpass"].cutoff, nyq * 0.99)
            if cutoff > 0:
                sos = scipy_signal.butter(4, cutoff / nyq, btype="high", output="sos")
                result = scipy_signal.sosfilt(sos, result)

        # Low-pass filter: removes high-frequency hiss above cutoff.
        if self._filters["lowpass"].active:
            cutoff = min(self._filters["lowpass"].cutoff, nyq * 0.99)
            if cutoff > 0:
                sos = scipy_signal.butter(4, cutoff / nyq, btype="low", output="sos")
                result = scipy_signal.sosfilt(sos, result)

        # Band-pass filter: isolates a frequency band around center, width = center/Q.
        if self._filters["bandpass"].active:
            center = self._filters["bandpass"].center
            q = max(0.1, self._filters["bandpass"].q)
            low = max(1.0, center - center / q)
            high = min(nyq * 0.99, center + center / q)
            if low < high:
                sos = scipy_signal.butter(4, [low / nyq, high / nyq], btype="band", output="sos")
                result = scipy_signal.sosfilt(sos, result)

        # Noise gate: mute samples below a threshold (in dB).
        if self._filters["noise_gate"].active:
            threshold_db = self._filters["noise_gate"].threshold
            threshold_linear = 10.0 ** (threshold_db / 20.0)
            # Compute envelope using a simple RMS window.
            window_size = max(1, int(sample_rate * 0.005))  # 5 ms window
            envelope = np.abs(result)
            # Smooth the envelope with a moving average.
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(envelope, kernel, mode="same")
            # Gate: set samples below threshold to silence.
            gate_mask = smoothed > threshold_linear
            result = result * gate_mask.astype("float32")

        # Volume boost (gain).
        if self._filters["gain"].active:
            result = result * self._filters["gain"].gain

        # Clip to prevent clipping artifacts.
        result = np.clip(result, -1.0, 1.0)
        return result.astype("float32")

    # -----------------------------------------------------------------------
    # Transport Controls (Module 1)
    # -----------------------------------------------------------------------

    def play(self) -> None:
        """Start playback of the processed audio."""
        if self._processed_audio is None and not self._mic_active:
            return

        if self._is_playing:
            return

        self._is_playing = True

        if self._mic_active:
            # For live mic, we process in real-time via the output stream callback.
            self._start_output_stream(self._audio_info.sample_rate, is_mic=True)
        else:
            # For file playback, stream the pre-processed buffer.
            sr = self._audio_info.sample_rate
            self._start_output_stream(sr, is_mic=False)

    def pause(self) -> None:
        """Pause playback (stops the stream but remembers position)."""
        self._is_playing = False
        if self._stream is not None:
            self._stream.stop()
            # Don't close — we may resume.

    def stop(self) -> None:
        """Stop playback and reset position to the beginning."""
        self._is_playing = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._playback_sample_idx = 0
        self._playback_position = 0.0

    def _start_output_stream(self, sample_rate: int, is_mic: bool) -> None:
        """
        Open a sounddevice OutputStream for playback.

        Args:
            sample_rate: Sample rate for the output stream.
            is_mic:      If True, process live mic data in real-time.
        """
        if self._stream is not None:
            self._stream.close()

        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=1024,
            callback=self._playback_callback,
        )
        self._stream.start()

    def _playback_callback(self, outdata: np.ndarray, frames: int, time_info, status) -> None:
        """
        sounddevice OutputStream callback.

        Called by PortAudio when it needs more samples to play.
        For file playback: read from the pre-processed buffer.
        For live mic: read from the mic circular buffer and apply real-time
        heterodyne processing if that method is selected.

        Args:
            outdata:    Output buffer to fill (shape: [frames, channels]).
            frames:     Number of samples needed.
            time_info:  Timing info (unused).
            status:     PortAudio status flags (unused).
        """
        if not self._is_playing:
            outdata.fill(0)
            return

        if self._mic_active:
            # --- Live mic real-time processing ---
            with self._lock:
                # Read from the circular buffer.
                read_data = np.zeros(frames, dtype="float32")
                for i in range(frames):
                    read_data[i] = self._mic_buffer[self._mic_write_idx - frames + i]
                    # Note: this reads the most recent 'frames' samples.

            if self._method == ShiftMethod.HETERODYNE:
                # Apply real-time heterodyne mixing.
                osc_freq = self._method_value * 1000.0
                sr = self._audio_info.sample_rate
                t = np.arange(frames, dtype="float64") / sr
                mixed = read_data * np.sin(2.0 * np.pi * osc_freq * t)
                # Simple one-pole low-pass for real-time difference extraction.
                alpha = 0.1  # Smoothing factor for ~8 kHz cutoff at 48 kHz SR.
                output = np.zeros(frames, dtype="float32")
                prev = 0.0
                for i in range(frames):
                    prev = prev + alpha * (mixed[i] - prev)
                    output[i] = prev
                read_data = output * 5.0  # Boost the difference tone.
            elif self._method in (ShiftMethod.TIME_EXPANSION, ShiftMethod.FREQUENCY_DIVISION):
                # For live mic, time expansion isn't truly real-time.
                # We slow the output by repeating samples.
                factor = max(1, int(round(self._method_value)))
                slowed = np.repeat(read_data, factor)
                # Take only 'frames' samples from the slowed output.
                if len(slowed) >= frames:
                    read_data = slowed[:frames]
                else:
                    read_data = np.pad(slowed, (0, frames - len(slowed)))

            outdata[:, 0] = read_data

        elif self._processed_audio is not None:
            # --- File playback from pre-processed buffer ---
            audio = self._processed_audio
            idx = self._playback_sample_idx
            remaining = len(audio) - idx

            if remaining <= 0:
                # Reached end of audio — stop.
                self._is_playing = False
                outdata.fill(0)
                return

            # Copy the next 'frames' samples (or pad with zeros if at end).
            to_copy = min(frames, remaining)
            outdata[:to_copy, 0] = audio[idx:idx + to_copy]
            if to_copy < frames:
                outdata[to_copy:, 0] = 0.0

            self._playback_sample_idx += to_copy
            sr = self._audio_info.sample_rate
            self._playback_position = self._playback_sample_idx / sr

    # -----------------------------------------------------------------------
    # Real-time frequency data for spectrogram (Module 3 interface)
    # -----------------------------------------------------------------------

    def get_frequency_data(self) -> np.ndarray | None:
        """
        Get the current frequency-domain data for the spectrogram.

        Computes an FFT on the most recent audio samples (from either the
        playback buffer or the live mic buffer) and returns the magnitude
        spectrum.

        Returns:
            numpy array of FFT bin magnitudes (0–255 scaled), or None if
            no audio is available.
        """
        # Get the most recent audio samples.
        if self._mic_active and self._mic_buffer is not None:
            # Read the last 'fft_size' samples from the circular buffer.
            n = min(self._fft_size, self._mic_buffer_size)
            with self._lock:
                if self._mic_write_idx >= n:
                    samples = self._mic_buffer[self._mic_write_idx - n:self._mic_write_idx].copy()
                else:
                    # Wrap around.
                    samples = np.concatenate([
                        self._mic_buffer[self._mic_buffer_size - (n - self._mic_write_idx):],
                        self._mic_buffer[:self._mic_write_idx],
                    ])
            sr = self._audio_info.sample_rate
        elif self._processed_audio is not None and self._is_playing:
            # Read from the playback buffer around the current position.
            idx = self._playback_sample_idx
            n = min(self._fft_size, len(self._processed_audio))
            start = max(0, idx - n)
            samples = self._processed_audio[start:idx].copy()
            if len(samples) < n:
                samples = np.pad(samples, (0, n - len(samples)))
            sr = self._audio_info.sample_rate
        elif self._raw_audio is not None:
            # Static display: show the spectrum of the original (unshifted) audio.
            n = min(self._fft_size, len(self._raw_audio))
            samples = self._raw_audio[:n].copy()
            sr = self._audio_info.sample_rate
        else:
            return None

        if len(samples) == 0:
            return None

        # Apply a Hann window to reduce spectral leakage.
        window = np.hanning(len(samples))
        windowed = samples * window

        # Compute FFT (only the positive-frequency half).
        fft_result = np.fft.rfft(windowed)
        magnitude = np.abs(fft_result)

        # Scale to 0–255 for the spectrogram color map.
        if magnitude.max() > 0:
            scaled = (magnitude / magnitude.max() * 255).astype("float32")
        else:
            scaled = np.zeros_like(magnitude)

        return scaled

    def get_peak_frequency(self) -> tuple[float, float]:
        """
        Get the current peak frequency and its magnitude.

        Returns:
            (peak_frequency_hz, peak_magnitude) or (0, 0) if no data.
        """
        freq_data = self.get_frequency_data()
        if freq_data is None or len(freq_data) == 0:
            return 0.0, 0.0

        max_idx = int(np.argmax(freq_data))
        sr = self._audio_info.sample_rate if self._audio_info.sample_rate > 0 else 44100
        # rfft produces bins from 0 to Nyquist, with num_bins = fft_size/2 + 1.
        num_bins = len(freq_data)
        freq_per_bin = (sr / 2.0) / (num_bins - 1) if num_bins > 1 else 0
        peak_freq = max_idx * freq_per_bin
        peak_mag = float(freq_data[max_idx])
        return peak_freq, peak_mag

    # -----------------------------------------------------------------------
    # Setters — called by the UI when the user changes settings
    # -----------------------------------------------------------------------

    def set_method(self, method: ShiftMethod) -> None:
        """Set the frequency-shift method and reprocess audio."""
        self._method = method
        self.stop()
        self._process_audio()

    def set_method_value(self, value: float) -> None:
        """Set the method parameter (expansion factor, oscillator kHz, or divisor)."""
        self._method_value = value
        self.stop()
        self._process_audio()

    def set_fft_size(self, size: int) -> None:
        """Set the FFT size for the spectrogram (must be power of 2)."""
        self._fft_size = size

    def set_filter(self, name: str, active: bool | None = None,
                   cutoff: float | None = None, center: float | None = None,
                   q: float | None = None, threshold: float | None = None,
                   gain: float | None = None) -> None:
        """
        Update a filter's parameters. Only provided values are changed.

        Args:
            name:      Filter name ("highpass", "lowpass", "bandpass",
                       "noise_gate", "gain").
            active:    Toggle the filter on/off.
            cutoff:    New cutoff frequency (Hz) for highpass/lowpass.
            center:    New center frequency (Hz) for bandpass.
            q:         New Q factor for bandpass.
            threshold: New threshold (dB) for noise gate.
            gain:      New gain multiplier for the gain stage.
        """
        if name not in self._filters:
            return

        f = self._filters[name]
        if active is not None:
            f.active = active
        if cutoff is not None:
            f.cutoff = cutoff
        if center is not None:
            f.center = center
        if q is not None:
            f.q = q
        if threshold is not None:
            f.threshold = threshold
        if gain is not None:
            f.gain = gain

        # Reprocess the audio with new filter settings.
        self._process_audio()

    def get_filter(self, name: str) -> FilterState | None:
        """Get the current state of a filter."""
        return self._filters.get(name)

    def get_all_filters(self) -> dict:
        """Get all filter states."""
        return self._filters

    # -----------------------------------------------------------------------
    # Getters — read-only state for the UI
    # -----------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._is_playing

    @property
    def is_mic_active(self) -> bool:
        return self._mic_active

    @property
    def audio_info(self) -> AudioInfo:
        return self._audio_info

    @property
    def playback_position(self) -> float:
        return self._playback_position

    @property
    def method(self) -> ShiftMethod:
        return self._method

    @property
    def method_value(self) -> float:
        return self._method_value

    @property
    def has_audio(self) -> bool:
        return self._raw_audio is not None or self._mic_active

    @property
    def fft_size(self) -> int:
        return self._fft_size

    def get_shifted_freq_range(self) -> tuple[float, float]:
        """
        Calculate the audible frequency range after shifting.

        Returns:
            (min_khz, max_khz) of the shifted audio.
        """
        if self._audio_info.sample_rate == 0:
            return 0.0, 0.0

        original_max = self._audio_info.nyquist / 1000.0  # kHz

        if self._method == ShiftMethod.TIME_EXPANSION:
            factor = max(1.0, self._method_value)
            return 0.0, original_max / factor
        elif self._method == ShiftMethod.FREQUENCY_DIVISION:
            divisor = max(2, int(round(self._method_value)))
            return 0.0, original_max / divisor
        elif self._method == ShiftMethod.HETERODYNE:
            # Heterodyne produces difference tones up to ~8 kHz (our low-pass cutoff).
            return 0.0, 8.0
        return 0.0, original_max