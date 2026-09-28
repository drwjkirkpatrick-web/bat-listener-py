"""
Triggered Recording Module (Improvement #18).

Implements AudioMoth-style amplitude threshold triggering for automatic
detection and recording of bat calls.

AudioMoth's bat trigger algorithm:
  1. Apply a high-pass filter (~10 kHz, simple 1-pole IIR) to remove
     low-frequency noise
  2. Compute per-millisecond energy (sum of squared amplitudes)
  3. Maintain a running average over ~500 ms (background noise level)
  4. Trigger recording when instantaneous energy exceeds the running average
     by a configurable factor for Y consecutive milliseconds
  5. Continue recording for a configurable post-trigger duration

This dramatically reduces storage by capturing only bat-relevant segments
instead of continuous recording.

Source: AudioMoth firmware (OpenAcousticDevices/AudioMoth-Project)
        RevSpace AudioMoth bat trigger documentation
"""

import numpy as np
from collections import deque
from dataclasses import dataclass


@dataclass
class TriggerEvent:
    """
    A detected trigger event (potential bat call).

    Attributes:
        start_sample:  Sample index where the trigger fired.
        end_sample:    Sample index where recording ended.
        duration_ms:   Duration of the triggered segment (ms).
        peak_energy:   Maximum energy during the event.
        timestamp_s:   Time in seconds from the start of monitoring.
    """
    start_sample: int = 0
    end_sample: int = 0
    duration_ms: float = 0.0
    peak_energy: float = 0.0
    timestamp_s: float = 0.0


class TriggeredRecorder:
    """
    Amplitude-threshold triggered recorder for bat call detection.

    The recorder continuously buffers audio but only "triggers" (saves)
    segments where the energy exceeds the background noise level by a
    configurable factor.

    Algorithm (per the AudioMoth firmware):
      1. High-pass filter at 10 kHz (1-pole IIR: y[n] = α * (y[n-1] + x[n] - x[n-1]))
      2. Compute energy per 1 ms block (sum of squared samples)
      3. Running average over 500 ms window (background noise estimate)
      4. If energy > factor × running_average for Y consecutive ms → trigger
      5. Record from trigger point + pre-trigger buffer for post-trigger duration

    Attributes:
        _sensitivity:     Trigger factor (energy must exceed average × this).
        _min_duration_ms: Minimum consecutive ms above threshold to trigger.
        _pre_trigger_ms:  How many ms to save before the trigger point.
        _post_trigger_ms: How many ms to save after the trigger point.
        _hp_cutoff_hz:    High-pass filter cutoff frequency (Hz).
        _running_avg:     Running average of energy (noise floor estimate).
        _consecutive_ms:  Count of consecutive ms above threshold.
        _triggered:       Whether currently in a triggered (recording) state.
        _events:          List of detected trigger events.
        _pre_buffer:      Ring buffer for pre-trigger audio.
    """

    def __init__(self, sensitivity: float = 4.0,
                 min_duration_ms: float = 3.0,
                 pre_trigger_ms: float = 50.0,
                 post_trigger_ms: float = 200.0,
                 hp_cutoff_hz: float = 10000.0):
        """
        Args:
            sensitivity:     Energy threshold factor (default 4× background).
            min_duration_ms: Minimum ms above threshold to trigger (default 3 ms).
            pre_trigger_ms:  Pre-trigger buffer length (default 50 ms).
            post_trigger_ms: Post-trigger recording duration (default 200 ms).
            hp_cutoff_hz:    High-pass filter cutoff (default 10 kHz).
        """
        self._sensitivity = sensitivity
        self._min_duration_ms = min_duration_ms
        self._pre_trigger_ms = pre_trigger_ms
        self._post_trigger_ms = post_trigger_ms
        self._hp_cutoff_hz = hp_cutoff_hz

        # High-pass filter state (1-pole IIR).
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0
        self._hp_alpha = 0.0

        # Noise floor tracking.
        self._running_avg = 0.0
        self._running_avg_window = 500  # 500 ms window

        # Trigger state.
        self._consecutive_ms = 0
        self._triggered = False
        self._trigger_start_sample = 0
        self._trigger_peak_energy = 0.0
        self._post_trigger_samples_remaining = 0

        # Pre-trigger ring buffer.
        self._pre_buffer_size = 0
        self._pre_buffer: deque = deque(maxlen=0)
        self._sample_rate = 0
        self._total_samples = 0

        # Events.
        self._events: list[TriggerEvent] = []

    def initialize(self, sample_rate: int) -> None:
        """
        Initialize the recorder for a new monitoring session.

        Args:
            sample_rate: Audio sample rate in Hz.
        """
        self._sample_rate = sample_rate
        ms_samples = sample_rate // 1000  # Samples per millisecond.

        # High-pass filter coefficient.
        # α = RC / (RC + dt), where RC = 1 / (2π * fc)
        rc = 1.0 / (2.0 * np.pi * self._hp_cutoff_hz)
        dt = 1.0 / sample_rate
        self._hp_alpha = rc / (rc + dt)

        # Pre-trigger buffer (in samples).
        self._pre_buffer_size = int(self._pre_trigger_ms * ms_samples)
        self._pre_buffer = deque(maxlen=self._pre_buffer_size)

        # Running average window (in samples).
        self._running_avg_window = int(500 * ms_samples)

        # Reset state.
        self._hp_prev_x = 0.0
        self._hp_prev_y = 0.0
        self._running_avg = 0.0
        self._consecutive_ms = 0
        self._triggered = False
        self._total_samples = 0
        self._events.clear()

    def process_block(self, block: np.ndarray) -> list[np.ndarray]:
        """
        Process a block of audio samples through the trigger detector.

        Returns any triggered segments that completed during this block.
        The returned segments include the pre-trigger buffer + the triggered
        audio + post-trigger audio.

        Args:
            block: 1D float32 audio samples (one processing block).

        Returns:
            List of triggered audio segments (each a 1D float32 array).
        """
        if self._sample_rate == 0:
            return []

        ms_samples = self._sample_rate // 1000
        triggered_segments = []

        for i in range(len(block)):
            sample = float(block[i])

            # --- Step 1: High-pass filter (1-pole IIR) ---
            # Removes low-frequency noise below the cutoff.
            filtered = self._hp_alpha * (self._hp_prev_y + sample - self._hp_prev_x)
            self._hp_prev_x = sample
            self._hp_prev_y = filtered

            # --- Step 2: Add to pre-trigger buffer ---
            self._pre_buffer.append(filtered)

            # --- Step 3: Compute per-ms energy ---
            # Accumulate energy for the current millisecond.
            if (self._total_samples % ms_samples) == 0 and self._total_samples > 0:
                # Compute energy for the previous ms block.
                ms_block = np.array(list(self._pre_buffer)[-ms_samples:], dtype="float64")
                energy = float(np.sum(ms_block ** 2))

                # --- Step 4: Update running average (noise floor) ---
                if self._running_avg == 0.0:
                    self._running_avg = energy
                else:
                    # Exponential moving average with the running average window.
                    alpha = 1.0 / self._running_avg_window
                    self._running_avg = (1 - alpha) * self._running_avg + alpha * energy

                # --- Step 5: Trigger detection ---
                threshold = self._running_avg * self._sensitivity

                if energy > threshold:
                    self._consecutive_ms += 1
                    if energy > self._trigger_peak_energy:
                        self._trigger_peak_energy = energy

                    # Check if we should trigger.
                    if (not self._triggered and
                        self._consecutive_ms >= int(self._min_duration_ms)):
                        # Trigger!
                        self._triggered = True
                        self._trigger_start_sample = self._total_samples - len(self._pre_buffer)
                        self._post_trigger_samples_remaining = int(
                            self._post_trigger_ms * ms_samples
                        )
                else:
                    self._consecutive_ms = 0
                    if not self._triggered:
                        self._trigger_peak_energy = 0.0

            # --- Step 6: Handle triggered recording ---
            if self._triggered:
                self._post_trigger_samples_remaining -= 1
                if self._post_trigger_samples_remaining <= 0:
                    # End of triggered segment — collect the audio.
                    # The segment is: pre-trigger buffer + current position.
                    segment = np.array(list(self._pre_buffer), dtype="float32")
                    triggered_segments.append(segment)

                    # Record the event.
                    event = TriggerEvent(
                        start_sample=self._trigger_start_sample,
                        end_sample=self._total_samples,
                        duration_ms=(self._total_samples - self._trigger_start_sample) / ms_samples,
                        peak_energy=self._trigger_peak_energy,
                        timestamp_s=self._trigger_start_sample / self._sample_rate,
                    )
                    self._events.append(event)

                    # Reset trigger state.
                    self._triggered = False
                    self._trigger_peak_energy = 0.0
                    self._consecutive_ms = 0

            self._total_samples += 1

        return triggered_segments

    @property
    def events(self) -> list[TriggerEvent]:
        """Return all detected trigger events."""
        return self._events

    @property
    def is_triggered(self) -> bool:
        """Whether currently recording a triggered event."""
        return self._triggered

    def reset(self) -> None:
        """Reset all state (e.g., when switching audio sources)."""
        self._pre_buffer.clear()
        self._running_avg = 0.0
        self._consecutive_ms = 0
        self._triggered = False
        self._total_samples = 0
        self._events.clear()