"""
Ultrasonic Microphone Hardware Module.

Provides a comprehensive interface for connecting to and recording from
professional ultrasonic microphones used in bat detection. This module handles:

  - Device detection and capability assessment via PortAudio/sounddevice
  - Known microphone database with specs for all major ultrasonic USB mics
  - Automatic device identification by name matching
  - Sample rate negotiation (request highest supported rate for max bandwidth)
  - Gain/sensitivity calibration and signal level monitoring
  - Real-time streaming with configurable buffer sizes
  - Connection status and health monitoring
  - Fallback handling for non-ultrasonic devices (built-in mics, smartphones)

Supported Microphones (auto-detected by device name):

  Professional (192+ kHz):
    - Dodotronic Ultramic 384K EVO  — 384 kHz, 190 kHz bandwidth, 16-bit
    - Dodotronic Ultramic 192K      — 192 kHz, 95 kHz bandwidth, 16-bit
    - Dodotronic Ultramic 250K      — 250 kHz, 125 kHz bandwidth, 16-bit
    - Pettersson M500-384           — 384 kHz, 160 kHz bandwidth, 16-bit
    - Pettersson M500-384 (192 mode) — 192 kHz, 95 kHz bandwidth, 16-bit
    - Avisoft UltraSoundGate 116Un + CM16/CMPA — 384 kHz, 180 kHz, 16-bit
    - Avisoft CM16/CMPA             — 200 kHz bandwidth, condenser
    - Wildlife Acoustics Echo Meter Touch 2 Pro — 256 kHz, 128 kHz, 16-bit
    - Elekon BATLOGGER A / A2       — 500 kHz, 250 kHz bandwidth
    - StudioMic DM2 / D500X         — 192 kHz, 96 kHz bandwidth

  Consumer / Extended Range (96 kHz):
    - Zoom H2n / H5 / H6           — 96 kHz, 48 kHz bandwidth
    - Focusrite Scarlett Solo      — 192 kHz, 96 kHz bandwidth
    - Sound Devices MixPre-3 II    — 192 kHz, 96 kHz bandwidth

  Smartphone / Built-in (44.1–48 kHz):
    - Built-in microphones          — 44.1/48 kHz, ~22 kHz bandwidth
    - USB microphones (general)     — 48 kHz, ~24 kHz bandwidth

Usage:
    mic = UltrasonicMicrophone()
    mic.detect_device()
    print(mic.device_info)
    mic.start_streaming()
    chunk = mic.read_chunk()
    mic.stop_streaming()

Sources:
  - Dodotronic Ultramic384 EVO User Guide (dodotronic.com)
  - Pettersson M500-384 User Guide (batsound.com)
  - Avisoft Bioacoustics CM16/CMPA + UltraSoundGate 116Un (avisoft.com)
  - Wildlife Acoustics Echo Meter Touch 2 specifications
  - AudioMoth Project (OpenAcoustic Devices)
  - sounddevice/PortAudio API documentation
"""

import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import threading
import time


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MicTier(Enum):
    """Microphone quality tier by ultrasonic capability."""
    PROFESSIONAL = "professional"    # 192+ kHz sample rate, full bat range
    EXTENDED = "extended"            # 96 kHz sample rate, partial bat range
    CONSUMER = "consumer"           # 48 kHz, low-freq bats only
    BUILTIN = "builtin"             # Built-in mic, limited
    UNKNOWN = "unknown"


class ConnectionState(Enum):
    """Microphone connection state."""
    DISCONNECTED = "disconnected"
    DETECTED = "detected"
    STREAMING = "streaming"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Known Microphone Database
# ---------------------------------------------------------------------------

@dataclass
class MicSpec:
    """
    Specifications for a known ultrasonic microphone model.

    Used for automatic device identification and capability display.
    """
    name: str                      # Display name
    manufacturer: str              # Company
    max_sample_rate_khz: float     # Maximum sample rate (kHz)
    bandwidth_khz: float           # Maximum recordable frequency (kHz)
    bit_depth: int                 # ADC resolution (bits)
    interface: str                 # Connection type (USB, XLR, etc.)
    frequency_range_khz: tuple     # (min, max) useful frequency range
    polar_pattern: str             # Omnidirectional, directional, etc.
    power: str                     # Power source
    notes: str                     # Additional info
    # Keywords in the PortAudio device name that identify this mic
    name_keywords: list = field(default_factory=list)


# Database of known ultrasonic microphones.
# Used for auto-identification when a device is connected.
KNOWN_MICROPHONES: list[MicSpec] = [
    MicSpec(
        name="Dodotronic Ultramic 384K EVO",
        manufacturer="Dodotronic",
        max_sample_rate_khz=384.0,
        bandwidth_khz=190.0,
        bit_depth=16,
        interface="USB 2.0/3.0",
        frequency_range_khz=(5, 190),
        polar_pattern="Omnidirectional (directional with horn)",
        power="USB bus powered (5V, 150mA)",
        notes="MEMS microphone, 8th order anti-aliasing filter at 190 kHz. "
              "Gain settings: -33, -10, 0, +10 dB. Works on Android, iOS, "
              "Windows, macOS, Linux.",
        name_keywords=["ultramic", "384", "dodotronic"],
    ),
    MicSpec(
        name="Dodotronic Ultramic 192K",
        manufacturer="Dodotronic",
        max_sample_rate_khz=192.0,
        bandwidth_khz=95.0,
        bit_depth=16,
        interface="USB 2.0",
        frequency_range_khz=(5, 95),
        polar_pattern="Omnidirectional",
        power="USB bus powered (5V, 150mA)",
        notes="Same hardware as 384K but configured for 192 kHz. "
              "Covers most bat species except some horseshoe bats (108 kHz).",
        name_keywords=["ultramic", "192", "dodotronic"],
    ),
    MicSpec(
        name="Dodotronic Ultramic 250K",
        manufacturer="Dodotronic",
        max_sample_rate_khz=250.0,
        bandwidth_khz=125.0,
        bit_depth=16,
        interface="USB",
        frequency_range_khz=(5, 125),
        polar_pattern="Omnidirectional",
        power="USB bus powered",
        notes="MEMS high sensitivity wide-band ultrasonic sensor. "
              "8th order low-pass anti-aliasing filter.",
        name_keywords=["ultramic", "250", "dodotronic"],
    ),
    MicSpec(
        name="Pettersson M500-384",
        manufacturer="Pettersson Elektronik",
        max_sample_rate_khz=384.0,
        bandwidth_khz=160.0,
        bit_depth=16,
        interface="USB 2.0 full-speed, OTG/host",
        frequency_range_khz=(10, 160),
        polar_pattern="Directional (with horn) / Omnidirectional (without)",
        power="USB bus powered (5V, 200mA)",
        notes="Advanced electret microphone, same as D500X detector. "
              "8th order anti-aliasing filter at 160 kHz. "
              "Works with Android, iOS, Windows, macOS, Linux. "
              "43x114x13mm, 60g. No custom drivers needed.",
        name_keywords=["pettersson", "m500", "384"],
    ),
    MicSpec(
        name="Pettersson M500-192",
        manufacturer="Pettersson Elektronik",
        max_sample_rate_khz=192.0,
        bandwidth_khz=95.0,
        bit_depth=16,
        interface="USB 2.0",
        frequency_range_khz=(10, 95),
        polar_pattern="Directional / Omnidirectional",
        power="USB bus powered",
        notes="M500 configured for 192 kHz mode. Good for most species.",
        name_keywords=["pettersson", "m500", "192"],
    ),
    MicSpec(
        name="Avisoft UltraSoundGate 116Un + CM16/CMPA",
        manufacturer="Avisoft Bioacoustics",
        max_sample_rate_khz=384.0,
        bandwidth_khz=180.0,
        bit_depth=16,
        interface="USB 1.1/2.0 full-speed",
        frequency_range_khz=(0.02, 180),
        polar_pattern="Omnidirectional (CM16/CMPA condenser)",
        power="USB bus powered (100mA)",
        notes="Condenser microphone with 200V polarization voltage. "
              "18 dB SPL self-noise. 500 mV/Pa sensitivity. "
              "Sample rates: 384, 256, 192, 128, 96, 48 kHz. "
              "XLR-5 connector allows 200m+ extension cables. "
              "Rugged aluminum enclosure, physical trigger button. "
              "Condenser mic recovers after moisture (unlike electret).",
        name_keywords=["avisoft", "ultrasoundgate", "116", "cm16"],
    ),
    MicSpec(
        name="Avisoft CM16/CMPA",
        manufacturer="Avisoft Bioacoustics",
        max_sample_rate_khz=300.0,
        bandwidth_khz=200.0,
        bit_depth=16,
        interface="XLR-5 (via UltraSoundGate base unit)",
        frequency_range_khz=(2, 200),
        polar_pattern="Omnidirectional",
        power="5V, 7mA from UltraSoundGate base unit",
        notes="Externally polarized metallized film diaphragm. "
              "Replaceable diaphragm. Optional USB-powered heating "
              "for field use in high humidity. 500 mV/Pa sensitivity. "
              "18 dB SPL self-noise (30-50 kHz bandwidth).",
        name_keywords=["avisoft", "cm16", "cmpa"],
    ),
    MicSpec(
        name="Wildlife Acoustics Echo Meter Touch 2 Pro",
        manufacturer="Wildlife Acoustics",
        max_sample_rate_khz=256.0,
        bandwidth_khz=128.0,
        bit_depth=16,
        interface="USB (micro-USB / USB-C / Lightning)",
        frequency_range_khz=(8, 128),
        polar_pattern="Directional",
        power="USB bus powered",
        notes="Designed for mobile devices. Companion app available. "
              "Covers full bat range up to 128 kHz. "
              "Split gain paths for ultrasonic and audible.",
        name_keywords=["echo", "meter", "touch", "wildlife", "acoustics"],
    ),
    MicSpec(
        name="Elekon BATLOGGER A2",
        manufacturer="Elekon AG",
        max_sample_rate_khz=500.0,
        bandwidth_khz=250.0,
        bit_depth=16,
        interface="USB",
        frequency_range_khz=(10, 250),
        polar_pattern="Directional",
        power="Battery + USB",
        notes="Highest sample rate in our database. "
              "Covers all known bat species including 200+ kHz. "
              "Built-in GPS for automatic location tagging. "
              "Automatic recording trigger with pre-trigger buffer.",
        name_keywords=["elekon", "batlogger"],
    ),
    MicSpec(
        name="AudioMoth (USB mode)",
        manufacturer="Open Acoustic Devices",
        max_sample_rate_khz=384.0,
        bandwidth_khz=192.0,
        bit_depth=16,
        interface="USB (micro-USB)",
        frequency_range_khz=(0.5, 192),
        polar_pattern="Omnidirectional",
        power="Battery / USB",
        notes="Open-source, low-cost ultrasonic recorder. "
              "Originally designed for passive deployment. "
              "Configurable amplitude threshold triggering. "
              "LED indicators for recording status. "
              "Firmware configurable via USB.",
        name_keywords=["audiomoth", "open", "acoustic"],
    ),
    MicSpec(
        name="Zoom H2n / H5 / H6 (96 kHz mode)",
        manufacturer="Zoom",
        max_sample_rate_khz=96.0,
        bandwidth_khz=48.0,
        bit_depth=24,
        interface="USB Audio Interface",
        frequency_range_khz=(0.02, 48),
        polar_pattern="Configurable (XY, AB, MS)",
        power="Battery / USB",
        notes="General-purpose handheld recorder with USB interface. "
              "48 kHz bandwidth captures low-frequency bats (noctule, "
              "serotine, Leisler's) but misses most species. "
              "Good for citizen science and audible-range monitoring.",
        name_keywords=["zoom", "h2n", "h5", "h6"],
    ),
    MicSpec(
        name="Sound Devices MixPre-3 II",
        manufacturer="Sound Devices",
        max_sample_rate_khz=192.0,
        bandwidth_khz=96.0,
        bit_depth=32,
        interface="USB Audio Interface",
        frequency_range_khz=(0.01, 96),
        polar_pattern="Depends on connected microphone",
        power="Battery / USB",
        notes="Professional audio recorder with 32-bit float recording. "
              "192 kHz sample rate gives 96 kHz bandwidth — enough for "
              "most bat species. High-quality preamps. "
              "Use with an ultrasonic capsule for best results.",
        name_keywords=["mixpre", "sound devices"],
    ),
    MicSpec(
        name="Focusrite Scarlett Solo (192 kHz)",
        manufacturer="Focusrite",
        max_sample_rate_khz=192.0,
        bandwidth_khz=96.0,
        bit_depth=24,
        interface="USB Audio Interface",
        frequency_range_khz=(0.01, 96),
        polar_pattern="Depends on connected microphone",
        power="USB bus powered",
        notes="Budget USB audio interface with 192 kHz support. "
              "Use with an ultrasonic microphone (e.g. Avisoft CM16) "
              "connected via XLR. 96 kHz bandwidth covers most bats. "
              "Air microphone preamp with 50 dB gain range.",
        name_keywords=["scarlett", "focusrite"],
    ),
]


# ---------------------------------------------------------------------------
# Device Info
# ---------------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """
    Information about a detected audio input device.

    Combines PortAudio device info with our microphone knowledge base.
    """
    # From PortAudio
    index: int = -1                # PortAudio device index
    name: str = ""                 # Device name (from PortAudio)
    max_channels: int = 0          # Max input channels
    default_sample_rate: int = 0   # Default sample rate (Hz)
    host_api: str = ""             # ALSA, CoreAudio, etc.

    # Computed / assessed
    max_sample_rate: int = 0       # Highest known/configured sample rate (Hz)
    nyquist_khz: float = 0.0      # Max recordable frequency (kHz)
    tier: MicTier = MicTier.UNKNOWN
    is_ultrasonic: bool = False

    # Matched known microphone (if any)
    matched_mic: MicSpec | None = None
    matched_mic_name: str = ""

    # Calibration
    gain_db: float = 0.0           # Current gain setting
    sensitivity_mv_pa: float = 0.0 # Microphone sensitivity (mV/Pa)
    noise_floor_db_spl: float = 0.0 # Self-noise (dB SPL)

    # Signal levels (updated during streaming)
    current_level_dbfs: float = -100.0  # Current signal level (dBFS)
    peak_level_dbfs: float = -100.0     # Peak since reset
    clip_count: int = 0                  # Number of clipping events

    def to_display_string(self) -> str:
        """Format as readable text for the UI."""
        lines = [
            f"Device: {self.name}",
            f"  Tier: {self.tier.value}",
            f"  Max sample rate: {self.max_sample_rate / 1000:.0f} kHz",
            f"  Nyquist: {self.nyquist_khz:.0f} kHz",
            f"  Channels: {self.max_channels}",
            f"  Ultrasonic: {'Yes' if self.is_ultrasonic else 'No'}",
        ]
        if self.matched_mic:
            lines.append(f"  Identified: {self.matched_mic.name}")
            lines.append(f"  Manufacturer: {self.matched_mic.manufacturer}")
            lines.append(f"  Bandwidth: {self.matched_mic.bandwidth_khz:.0f} kHz")
            lines.append(f"  Bit depth: {self.matched_mic.bit_depth}")
            lines.append(f"  Interface: {self.matched_mic.interface}")
            lines.append(f"  Polar pattern: {self.matched_mic.polar_pattern}")
            lines.append(f"  Power: {self.matched_mic.power}")
            if self.matched_mic.notes:
                lines.append(f"  Notes: {self.matched_mic.notes[:100]}")
        else:
            if self.tier == MicTier.BUILTIN:
                lines.append("  Recommendation: Connect a USB ultrasonic microphone")
                lines.append("  for full bat call detection (192+ kHz).")
                lines.append("  This device can capture low-freq bats only (noctule, serotine).")
            elif self.tier == MicTier.CONSUMER:
                lines.append("  Recommendation: A 192+ kHz USB mic is recommended")
                lines.append("  for full-spectrum bat recording.")
        return "\n".join(lines)

    def get_detectable_species_summary(self) -> str:
        """
        Summarize which bat species are detectable with this device.

        Checks the species database against the device's Nyquist frequency.
        """
        try:
            from species_guide import SPECIES_DATABASE
        except ImportError:
            return "Species database not available."

        max_khz = self.nyquist_khz
        detectable = []
        undetectable = []

        for sp in SPECIES_DATABASE:
            if sp.peak_freq_khz <= max_khz:
                detectable.append(f"  ✓ {sp.name} ({sp.peak_freq_khz:.0f} kHz)")
            else:
                undetectable.append(f"  ✗ {sp.name} ({sp.peak_freq_khz:.0f} kHz) — above {max_khz:.0f} kHz")

        lines = [f"Detectable species ({len(detectable)}/{len(detectable)+len(undetectable)}):"]
        lines.extend(detectable[:10])  # Show first 10
        if len(detectable) > 10:
            lines.append(f"  ... and {len(detectable)-10} more")
        if undetectable:
            lines.append(f"\nNot detectable ({len(undetectable)}):")
            lines.extend(undetectable[:5])
            if len(undetectable) > 5:
                lines.append(f"  ... and {len(undetectable)-5} more")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ultrasonic Microphone Interface
# ---------------------------------------------------------------------------

class UltrasonicMicrophone:
    """
    Hardware interface for ultrasonic microphone recording.

    This class manages:
      1. Device detection — scans PortAudio devices, identifies known mics
      2. Capability assessment — determines max sample rate, Nyquist, tier
      3. Sample rate negotiation — requests the highest supported rate
      4. Real-time streaming — manages an InputStream with configurable buffers
      5. Signal monitoring — tracks input levels, clipping, and noise floor
      6. Calibration — gain adjustment and sensitivity compensation

    The class is designed to work with:
      - Professional USB ultrasonic mics (Dodotronic, Pettersson, Avisoft)
      - Consumer USB audio interfaces at 96/192 kHz (Zoom, Focusrite, MixPre)
      - Built-in microphones (limited, but never blocked)

    For non-ultrasonic devices, the class provides clear warnings about which
    species are detectable and recommends upgrading to an ultrasonic microphone.
    """

    # Common sample rates to try (highest first) for negotiation.
    NEGOTIATION_RATES = [384000, 256000, 250000, 192000, 96000, 48000, 44100]

    def __init__(self):
        self._device_info = DeviceInfo()
        self._state = ConnectionState.DISCONNECTED
        self._stream: sd.InputStream | None = None
        self._sample_rate: int = 0
        self._blocksize: int = 1024
        self._channels: int = 1

        # Circular buffer for captured audio.
        self._buffer: np.ndarray | None = None
        self._buffer_size: int = 0
        self._write_idx: int = 0
        self._lock = threading.Lock()

        # Signal monitoring.
        self._level_rms: float = 0.0
        self._peak_rms: float = 0.0
        self._clip_count: int = 0
        self._last_clip_time: float = 0.0

        # Callback for real-time data consumers (spectrogram, triggered recorder, etc.)
        self._data_callback: Callable | None = None

    # -----------------------------------------------------------------------
    # Device Detection
    # -----------------------------------------------------------------------

    def detect_device(self, device_index: int | None = None) -> DeviceInfo:
        """
        Detect and identify the connected audio input device.

        Scans all PortAudio input devices, optionally selects a specific one,
        and attempts to identify it against the known microphone database.

        Args:
            device_index: Optional PortAudio device index. If None, uses default.

        Returns:
            DeviceInfo with full assessment.
        """
        try:
            devices = sd.query_devices()

            # Select device.
            if device_index is not None:
                idx = device_index
            else:
                idx = sd.default.device[0]
                if idx is None or idx < 0:
                    # Find first input device.
                    for i, d in enumerate(devices):
                        if d["max_input_channels"] > 0:
                            idx = i
                            break

            if idx is None or idx < 0 or idx >= len(devices):
                self._device_info = DeviceInfo(name="No input device found")
                self._state = ConnectionState.ERROR
                return self._device_info

            dev = devices[idx]
            name = dev["name"]
            max_sr = int(dev["default_samplerate"])

            # Try to negotiate a higher sample rate.
            negotiated_sr = self._negotiate_sample_rate(idx)
            if negotiated_sr > max_sr:
                max_sr = negotiated_sr

            # Assess device tier.
            nyquist_khz = max_sr / 2000.0
            if max_sr >= 192000:
                tier = MicTier.PROFESSIONAL
                is_ultrasonic = True
            elif max_sr >= 96000:
                tier = MicTier.EXTENDED
                is_ultrasonic = True
            elif max_sr >= 48000:
                tier = MicTier.CONSUMER
                is_ultrasonic = False
            else:
                tier = MicTier.BUILTIN
                is_ultrasonic = False

            # Match against known microphone database.
            matched_mic = self._identify_microphone(name)

            # Get host API name.
            try:
                host_apis = sd.query_hostapis()
                host_api_idx = dev.get("hostapi", 0)
                host_api_name = host_apis[host_api_idx]["name"] if host_api_idx < len(host_apis) else "Unknown"
            except Exception:
                host_api_name = "Unknown"

            self._device_info = DeviceInfo(
                index=idx,
                name=name,
                max_channels=dev["max_input_channels"],
                default_sample_rate=int(dev["default_samplerate"]),
                max_sample_rate=max_sr,
                nyquist_khz=nyquist_khz,
                tier=tier,
                is_ultrasonic=is_ultrasonic,
                matched_mic=matched_mic,
                matched_mic_name=matched_mic.name if matched_mic else "",
                host_api=host_api_name,
            )

            # If we matched a known mic, use its specs for calibration values.
            if matched_mic:
                self._device_info.sensitivity_mv_pa = 500.0  # Default for matched mics
                self._device_info.noise_floor_db_spl = 18.0

            self._state = ConnectionState.DETECTED
            return self._device_info

        except Exception as e:
            self._device_info = DeviceInfo(name=f"Detection error: {e}")
            self._state = ConnectionState.ERROR
            return self._device_info

    def _negotiate_sample_rate(self, device_index: int) -> int:
        """
        Try to negotiate the highest supported sample rate.

        PortAudio doesn't directly report max supported rate, so we try
        check_input_settings() at increasing rates until one works.

        Args:
            device_index: PortAudio device index.

        Returns:
            Highest sample rate that the device supports (Hz).
        """
        best_rate = 44100  # Fallback

        for rate in self.NEGOTIATION_RATES:
            try:
                sd.check_input_settings(
                    device=device_index,
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                )
                best_rate = rate
                break  # Found the highest that works
            except Exception:
                continue

        return best_rate

    def _identify_microphone(self, name: str) -> MicSpec | None:
        """
        Identify a connected microphone by matching its PortAudio device name
        against the known microphone database.

        Args:
            name: PortAudio device name string.

        Returns:
            Matching MicSpec, or None if no match.
        """
        name_lower = name.lower()

        for mic in KNOWN_MICROPHONES:
            # Check if all keywords for this mic appear in the device name.
            # We use a flexible match: any 2+ keywords matching is sufficient.
            matches = sum(1 for kw in mic.name_keywords if kw in name_lower)
            if matches >= 2:
                return mic
            # Single keyword match for distinctive names.
            if matches >= 1 and len(mic.name_keywords) <= 2:
                return mic

        return None

    # -----------------------------------------------------------------------
    # Streaming
    # -----------------------------------------------------------------------

    def start_streaming(self, sample_rate: int | None = None,
                        blocksize: int = 1024,
                        buffer_seconds: float = 5.0) -> bool:
        """
        Start real-time audio streaming from the microphone.

        Opens a sounddevice InputStream with the negotiated sample rate and
        a circular buffer of the specified duration.

        Args:
            sample_rate:    Override sample rate (None = use detected max).
            blocksize:      PortAudio block size (samples per callback).
            buffer_seconds: Circular buffer length in seconds.

        Returns:
            True if streaming started successfully.
        """
        if self._state == ConnectionState.STREAMING:
            return True

        if self._device_info.index < 0:
            self.detect_device()

        if self._device_info.index < 0:
            return False

        sr = sample_rate or self._device_info.max_sample_rate
        self._sample_rate = sr
        self._blocksize = blocksize
        self._channels = 1

        # Allocate circular buffer.
        self._buffer_size = int(sr * buffer_seconds)
        self._buffer = np.zeros(self._buffer_size, dtype="float32")
        self._write_idx = 0

        try:
            self._stream = sd.InputStream(
                device=self._device_info.index,
                samplerate=sr,
                channels=1,
                dtype="float32",
                blocksize=blocksize,
                callback=self._stream_callback,
            )
            self._stream.start()
            self._state = ConnectionState.STREAMING
            return True

        except Exception as e:
            print(f"[UltrasonicMic] Error starting stream: {e}")
            # Try with a lower sample rate as fallback.
            if sr > 48000:
                print(f"[UltrasonicMic] Retrying at 48000 Hz...")
                return self.start_streaming(sample_rate=48000, blocksize=blocksize,
                                           buffer_seconds=buffer_seconds)
            self._state = ConnectionState.ERROR
            return False

    def stop_streaming(self) -> None:
        """Stop the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._state = ConnectionState.DETECTED

    def _stream_callback(self, indata: np.ndarray, frames: int,
                         time_info, status) -> None:
        """
        PortAudio InputStream callback.

        Called when new audio data is available from the microphone.
        Copies data into the circular buffer and updates signal monitoring.

        Args:
            indata:     Input audio data [frames, channels].
            frames:     Number of samples.
            time_info:  Timing info (unused).
            status:     PortAudio status flags.
        """
        if self._state != ConnectionState.STREAMING:
            return

        # Convert to mono.
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        # Write to circular buffer.
        with self._lock:
            end_idx = self._write_idx + frames
            if end_idx <= self._buffer_size:
                self._buffer[self._write_idx:end_idx] = mono
            else:
                first_part = self._buffer_size - self._write_idx
                self._buffer[self._write_idx:] = mono[:first_part]
                self._buffer[:end_idx - self._buffer_size] = mono[first_part:]
            self._write_idx = (self._write_idx + frames) % self._buffer_size

        # Signal monitoring.
        rms = float(np.sqrt(np.mean(mono ** 2)))
        if rms > 0:
            self._level_rms = rms
            self._device_info.current_level_dbfs = 20 * np.log10(rms + 1e-10)
            if rms > self._peak_rms:
                self._peak_rms = rms
                self._device_info.peak_level_dbfs = self._device_info.current_level_dbfs

            # Detect clipping.
            if np.max(np.abs(mono)) >= 0.99:
                now = time.time()
                if now - self._last_clip_time > 0.1:  # Debounce at 100ms
                    self._clip_count += 1
                    self._device_info.clip_count = self._clip_count
                    self._last_clip_time = now

        # Notify external consumers (spectrogram, triggered recorder, etc.)
        if self._data_callback is not None:
            try:
                self._data_callback(mono.copy())
            except Exception:
                pass  # Don't let callback errors crash the stream.

    # -----------------------------------------------------------------------
    # Data Access
    # -----------------------------------------------------------------------

    def read_chunk(self, n_samples: int = 1024) -> np.ndarray | None:
        """
        Read the most recent n_samples from the circular buffer.

        Args:
            n_samples: Number of samples to read.

        Returns:
            1D float32 array of the most recent audio, or None if not streaming.
        """
        if self._buffer is None or self._state != ConnectionState.STREAMING:
            return None

        n = min(n_samples, self._buffer_size)

        with self._lock:
            if self._write_idx >= n:
                return self._buffer[self._write_idx - n:self._write_idx].copy()
            else:
                # Wrap around.
                return np.concatenate([
                    self._buffer[self._buffer_size - (n - self._write_idx):],
                    self._buffer[:self._write_idx],
                ]).copy()

    def read_buffer(self) -> np.ndarray | None:
        """
        Read the entire circular buffer.

        Returns:
            Full buffer contents, or None if not streaming.
        """
        if self._buffer is None or self._state != ConnectionState.STREAMING:
            return None

        with self._lock:
            return self._buffer.copy()

    def set_data_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """
        Set a callback to receive real-time audio chunks.

        The callback is called from the PortAudio thread for each block of
        audio data. It should be fast and non-blocking.

        Args:
            callback: Function taking a 1D float32 numpy array.
        """
        self._data_callback = callback

    # -----------------------------------------------------------------------
    # Calibration & Monitoring
    # -----------------------------------------------------------------------

    def calibrate_noise_floor(self, duration_s: float = 1.0) -> float:
        """
        Measure the current noise floor in dBFS.

        Records 'duration_s' seconds of audio and computes the RMS level.
        This should be done with no bat activity (background noise only).

        Args:
            duration_s: How long to sample for the measurement.

        Returns:
            Noise floor in dBFS (negative value).
        """
        n_samples = int(self._sample_rate * duration_s)
        chunk = self.read_chunk(n_samples)

        if chunk is None or len(chunk) == 0:
            return -100.0

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > 0:
            return 20 * np.log10(rms)
        return -100.0

    def reset_peak(self) -> None:
        """Reset the peak level tracker."""
        self._peak_rms = 0.0
        self._device_info.peak_level_dbfs = -100.0
        self._clip_count = 0
        self._device_info.clip_count = 0

    def set_gain(self, gain_db: float) -> None:
        """
        Set the recording gain (informational — actual gain is hardware-controlled).

        For USB ultrasonic mics like the Dodotronic, gain is set via physical
        switches. For audio interfaces, gain is set via the preamp knob.
        This method records the gain for metadata purposes.

        Args:
            gain_db: Gain in dB.
        """
        self._device_info.gain_db = gain_db

    # -----------------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------------

    @property
    def device_info(self) -> DeviceInfo:
        return self._device_info

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._state == ConnectionState.STREAMING

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def is_ultrasonic(self) -> bool:
        return self._device_info.is_ultrasonic

    @property
    def nyquist_khz(self) -> float:
        return self._device_info.nyquist_khz

    @property
    def current_level_dbfs(self) -> float:
        return self._device_info.current_level_dbfs

    @property
    def peak_level_dbfs(self) -> float:
        return self._device_info.peak_level_dbfs

    @property
    def clip_count(self) -> int:
        return self._clip_count


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def list_all_input_devices() -> list[DeviceInfo]:
    """
    List all available audio input devices with capability assessment.

    Useful for showing a device selection menu in the UI.

    Returns:
        List of DeviceInfo objects, one per input device.
    """
    devices = sd.query_devices()
    results = []

    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            # Create a temporary UltrasonicMicrophone to assess each device.
            mic = UltrasonicMicrophone()
            info = mic.detect_device(device_index=i)
            results.append(info)

    return results


def get_recommended_microphones(budget: str = "any") -> list[MicSpec]:
    """
    Return recommended microphones for bat recording, sorted by value.

    Args:
        budget: "any", "low", "mid", or "high" — filters by price range.

    Returns:
        List of MicSpec objects, best value first.
    """
    # Price tiers (approximate):
    low_budget = ["AudioMoth", "Dodotronic Ultramic 192K"]
    mid_budget = ["Dodotronic Ultramic 384K", "Dodotronic Ultramic 250K",
                  "Pettersson M500-384", "Wildlife Acoustics Echo Meter Touch 2 Pro"]
    high_budget = ["Avisoft UltraSoundGate 116Un + CM16/CMPA",
                   "Avisoft CM16/CMPA", "Elekon BATLOGGER A2"]

    if budget == "low":
        names = low_budget
    elif budget == "mid":
        names = mid_budget
    elif budget == "high":
        names = high_budget
    else:
        names = [m.name for m in KNOWN_MICROPHONES]

    return [m for m in KNOWN_MICROPHONES if m.name in names]