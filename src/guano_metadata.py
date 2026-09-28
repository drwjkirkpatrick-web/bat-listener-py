"""
GUANO Metadata Module (Improvements #17, #21).

GUANO (Grand Unified Acoustic Notation Ontology) is an open metadata standard
for bat acoustic recordings. It embeds key metadata directly into WAV file
headers, making recordings self-describing and interoperable with the broader
bat research ecosystem (Kaleidoscope, Wildlife Acoustics, BatSync, NABat).

This module provides:
  - Reading GUANO metadata from existing WAV files
  - Writing GUANO metadata to WAV exports
  - A GuanoMetadata dataclass for structured access to fields

The GUANO format spec: https://github.com/riggsd/guano-spec
The guano-py package: https://pypi.org/project/guano/

Format overview:
  GUANO metadata is stored as ASCII key-value pairs in the WAV file's RIFF
  chunks. The chunk ID is "guan" (or "GUAN" in some implementations).
  Lines are terminated with CRLF (\\r\\n), like HTTP headers.

  Example:
    GUANO|1.0
    Timestamp|2026-09-28T22:30:00-07:00
    Species Auto ID|Myotis lucifugus
    Lat|44.0521
    Lon|-121.4158
    Detector|Bat Listener Python
    Samplerate|192000
    Note|Recorded at sunset near river
"""

import struct
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GuanoMetadata:
    """
    Structured representation of GUANO metadata fields.

    This covers the most commonly used fields from the GUANO 1.0 spec.
    Custom/namespace fields can be stored in the `extra` dict.

    Field names match the GUANO spec exactly (case-sensitive, with spaces).
    """
    # --- Core GUANO header ---
    version: str = "1.0"

    # --- Temporal ---
    timestamp: str = ""          # ISO 8601: 2026-09-28T22:30:00-07:00

    # --- Spatial ---
    latitude: Optional[float] = None    # GPS latitude (decimal degrees)
    longitude: Optional[float] = None   # GPS longitude (decimal degrees)

    # --- Species ---
    species_auto_id: str = ""    # Automated species ID (scientific name)
    species_manual_id: str = ""  # Manual species ID (scientific name)
    species_auto_p: float = 0.0  # Confidence (0.0–1.0) for auto ID

    # --- Equipment ---
    detector_model: str = ""     # Detector / recorder model
    detector_serial: str = ""   # Serial number
    microphone: str = ""        # Microphone type/model
    samplerate: int = 0         # Sample rate in Hz

    # --- Recording parameters ---
    gain: float = 0.0           # Recording gain (dB)
    filter_highpass: float = 0.0  # High-pass filter cutoff (Hz)
    filter_lowpass: float = 0.0  # Low-pass filter cutoff (Hz)

    # --- Survey metadata ---
    surveyor: str = ""          # Person who made the recording
    survey_id: str = ""         # Survey identifier
    transect_id: str = ""       # Transect identifier
    point_id: str = ""          # Point count identifier

    # --- Environmental ---
    temperature: float = 0.0    # Temperature (°C)
    humidity: float = 0.0       # Humidity (%)
    pressure: float = 0.0       # Barometric pressure (hPa)
    moon_phase: float = 0.0     # Moon phase (0=new, 0.5=full, 1=new)

    # --- Free-form notes ---
    note: str = ""

    # --- Extra/custom fields ---
    extra: dict = field(default_factory=dict)

    def to_guano_string(self) -> str:
        """
        Serialize metadata to the GUANO text format (ASCII key-value pairs).

        The first line is always "GUANO|<version>". All subsequent lines are
        "Key|Value" pairs, CRLF-terminated.

        Returns:
            GUANO-formatted string (without trailing CRLF).
        """
        lines = [f"GUANO|{self.version}"]

        # Build key-value pairs in spec order.
        kv = []
        if self.timestamp:
            kv.append(("Timestamp", self.timestamp))
        if self.latitude is not None:
            kv.append(("Lat", f"{self.latitude:.6f}"))
        if self.longitude is not None:
            kv.append(("Lon", f"{self.longitude:.6f}"))
        if self.species_auto_id:
            kv.append(("Species Auto ID", self.species_auto_id))
        if self.species_manual_id:
            kv.append(("Species Manual ID", self.species_manual_id))
        if self.species_auto_p > 0:
            kv.append(("Species Auto ID P", f"{self.species_auto_p:.2f}"))
        if self.detector_model:
            kv.append(("Detector", self.detector_model))
        if self.detector_serial:
            kv.append(("Detector Serial", self.detector_serial))
        if self.microphone:
            kv.append(("Microphone", self.microphone))
        if self.samplerate > 0:
            kv.append(("Samplerate", str(self.samplerate)))
        if self.gain > 0:
            kv.append(("Gain", f"{self.gain:.1f}"))
        if self.filter_highpass > 0:
            kv.append(("HP Filter", f"{self.filter_highpass:.0f}"))
        if self.filter_lowpass > 0:
            kv.append(("LP Filter", f"{self.filter_lowpass:.0f}"))
        if self.surveyor:
            kv.append(("Surveyor", self.surveyor))
        if self.survey_id:
            kv.append(("Survey ID", self.survey_id))
        if self.transect_id:
            kv.append(("Transect ID", self.transect_id))
        if self.point_id:
            kv.append(("Point ID", self.point_id))
        if self.temperature != 0:
            kv.append(("Temperature", f"{self.temperature:.1f}"))
        if self.humidity != 0:
            kv.append(("Humidity", f"{self.humidity:.1f}"))
        if self.pressure != 0:
            kv.append(("Pressure", f"{self.pressure:.1f}"))
        if self.moon_phase != 0:
            kv.append(("Moon Phase", f"{self.moon_phase:.2f}"))
        if self.note:
            kv.append(("Note", self.note))
        # Extra custom fields.
        for key, value in self.extra.items():
            kv.append((key, str(value)))

        for key, value in kv:
            lines.append(f"{key}|{value}")

        return "\r\n".join(lines)

    @classmethod
    def from_guano_string(cls, text: str) -> "GuanoMetadata":
        """
        Parse a GUANO-formatted string into a GuanoMetadata object.

        Args:
            text: GUANO-formatted text (key-value pairs separated by |, lines by CRLF or LF).

        Returns:
            GuanoMetadata with parsed fields.
        """
        meta = cls()
        lines = text.replace("\r\n", "\n").split("\n")

        for line in lines:
            line = line.strip()
            if not line or "|" not in line:
                continue

            # Split on first | only (values may contain |).
            key, value = line.split("|", 1)

            if key == "GUANO":
                meta.version = value
            elif key == "Timestamp":
                meta.timestamp = value
            elif key == "Lat":
                try: meta.latitude = float(value)
                except: pass
            elif key == "Lon":
                try: meta.longitude = float(value)
                except: pass
            elif key == "Species Auto ID":
                meta.species_auto_id = value
            elif key == "Species Manual ID":
                meta.species_manual_id = value
            elif key == "Species Auto ID P":
                try: meta.species_auto_p = float(value)
                except: pass
            elif key == "Detector":
                meta.detector_model = value
            elif key == "Detector Serial":
                meta.detector_serial = value
            elif key == "Microphone":
                meta.microphone = value
            elif key == "Samplerate":
                try: meta.samplerate = int(value)
                except: pass
            elif key == "Gain":
                try: meta.gain = float(value)
                except: pass
            elif key == "HP Filter":
                try: meta.filter_highpass = float(value)
                except: pass
            elif key == "LP Filter":
                try: meta.filter_lowpass = float(value)
                except: pass
            elif key == "Surveyor":
                meta.surveyor = value
            elif key == "Survey ID":
                meta.survey_id = value
            elif key == "Transect ID":
                meta.transect_id = value
            elif key == "Point ID":
                meta.point_id = value
            elif key == "Temperature":
                try: meta.temperature = float(value)
                except: pass
            elif key == "Humidity":
                try: meta.humidity = float(value)
                except: pass
            elif key == "Pressure":
                try: meta.pressure = float(value)
                except: pass
            elif key == "Moon Phase":
                try: meta.moon_phase = float(value)
                except: pass
            elif key == "Note":
                meta.note = value
            else:
                # Unknown field → store in extra.
                meta.extra[key] = value

        return meta

    def to_display_string(self) -> str:
        """
        Format metadata as a readable multi-line string for the UI panel.

        Returns:
            Human-readable metadata summary.
        """
        lines = ["GUANO Metadata:"]
        if self.timestamp:
            lines.append(f"  Timestamp: {self.timestamp}")
        if self.latitude is not None and self.longitude is not None:
            lines.append(f"  GPS: {self.latitude:.4f}, {self.longitude:.4f}")
        if self.species_auto_id:
            p = f" ({self.species_auto_p:.0%})" if self.species_auto_p > 0 else ""
            lines.append(f"  Auto ID: {self.species_auto_id}{p}")
        if self.species_manual_id:
            lines.append(f"  Manual ID: {self.species_manual_id}")
        if self.detector_model:
            lines.append(f"  Detector: {self.detector_model}")
        if self.samplerate > 0:
            lines.append(f"  Sample Rate: {self.samplerate} Hz")
        if self.surveyor:
            lines.append(f"  Surveyor: {self.surveyor}")
        if self.temperature != 0:
            lines.append(f"  Temp: {self.temperature:.1f}°C")
        if self.note:
            lines.append(f"  Note: {self.note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# WAV File GUANO Chunk Read/Write
# ---------------------------------------------------------------------------

def write_wav_with_guano(filepath: str, audio: "np.ndarray", sample_rate: int,
                         metadata: GuanoMetadata) -> bool:
    """
    Write a WAV file with embedded GUANO metadata.

    The GUANO data is stored as a RIFF chunk with ID "guan" in the WAV file.
    This makes the file readable by standard WAV tools (the GUANO chunk is
    ignored by tools that don't understand it) while carrying full metadata
    for tools that do.

    Args:
            filepath:    Destination file path.
            audio:       1D numpy array of float32 audio samples.
            sample_rate: Sample rate in Hz.
            metadata:    GuanoMetadata to embed.

    Returns:
            True if the file was written successfully.
    """
    try:
        import soundfile as sf
        # Write the WAV file first.
        sf.write(filepath, audio, sample_rate)

        # Now append the GUANO chunk by rewriting the file with the extra chunk.
        # WAV files are RIFF containers: RIFF<size>WAVE<fmt ><data>[<guan>]
        _append_guano_chunk(filepath, metadata)
        return True
    except Exception as e:
        print(f"[GUANO] Error writing WAV with metadata: {e}")
        return False


def _append_guano_chunk(filepath: str, metadata: GuanoMetadata) -> None:
    """
    Append a GUANO chunk to an existing WAV file.

    The GUANO chunk format in a RIFF/WAV container:
      4 bytes: "guan" (chunk ID)
      4 bytes: chunk size (uint32, little-endian)
      N bytes: GUANO text data (ASCII, CRLF-terminated lines)

    This function reads the file, finds the data chunk, and inserts the
    GUANO chunk after it, updating the RIFF size.

    Args:
            filepath: Path to the WAV file (must already exist).
            metadata: GUANO metadata to embed.
    """
    guano_text = metadata.to_guano_string().encode("ascii")
    chunk_data = guano_text + b"\r\n"
    # Pad to even length (RIFF chunks must be word-aligned).
    if len(chunk_data) % 2 != 0:
        chunk_data += b"\x00"

    chunk_header = b"guan" + struct.pack("<I", len(chunk_data))
    guano_chunk = chunk_header + chunk_data

    with open(filepath, "r+b") as f:
        # Read RIFF header.
        riff_id = f.read(4)
        if riff_id != b"RIFF":
            return
        riff_size = struct.unpack("<I", f.read(4))[0]
        wave_id = f.read(4)
        if wave_id != b"WAVE":
            return

        # Find the data chunk to insert after it.
        pos = f.tell()
        found_data = False
        while pos < 12 + riff_size:
            f.seek(pos)
            chunk_id = f.read(4)
            chunk_size = struct.unpack("<I", f.read(4))[0]
            if chunk_id == b"data":
                found_data = True
                # Skip past the data chunk.
                data_end = pos + 8 + chunk_size + (chunk_size % 2)
                break
            pos = pos + 8 + chunk_size + (chunk_size % 2)

        if not found_data:
            return

        # Read everything after the data chunk.
        f.seek(data_end)
        trailing = f.read()

        # Write the GUANO chunk + trailing data.
        f.seek(data_end)
        f.write(guano_chunk)
        f.write(trailing)

        # Update the RIFF size.
        new_riff_size = riff_size + len(guano_chunk)
        f.seek(4)
        f.write(struct.pack("<I", new_riff_size))


def read_guano_from_wav(filepath: str) -> GuanoMetadata | None:
    """
    Read GUANO metadata from a WAV file.

    Scans the RIFF chunks for a "guan" chunk and parses it.

    Args:
            filepath: Path to the WAV file.

    Returns:
            GuanoMetadata if found, None otherwise.
    """
    try:
        with open(filepath, "rb") as f:
            riff_id = f.read(4)
            if riff_id != b"RIFF":
                return None
            riff_size = struct.unpack("<I", f.read(4))[0]
            wave_id = f.read(4)
            if wave_id != b"WAVE":
                return None

            # Scan chunks for "guan".
            pos = f.tell()
            while pos < 12 + riff_size:
                f.seek(pos)
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack("<I", f.read(4))[0]
                if chunk_id == b"guan":
                    # Found the GUANO chunk — read and parse it.
                    guano_data = f.read(chunk_size).decode("ascii", errors="replace")
                    # Remove padding null byte if present.
                    guano_data = guano_data.rstrip("\x00")
                    return GuanoMetadata.from_guano_string(guano_data)
                # Skip to next chunk (word-aligned).
                pos = pos + 8 + chunk_size + (chunk_size % 2)

        return None
    except Exception as e:
        print(f"[GUANO] Error reading GUANO metadata: {e}")
        return None


def create_default_metadata(sample_rate: int, filename: str = "",
                            species: str = "", confidence: float = 0.0,
                            latitude: float | None = None,
                            longitude: float | None = None) -> GuanoMetadata:
    """
    Create a GuanoMetadata with sensible defaults for a new recording.

    Args:
            sample_rate: Sample rate of the recording.
            filename:    Original filename (stored in Note).
            species:     Predicted species scientific name.
            confidence:  Classification confidence (0.0–1.0).
            latitude:    GPS latitude (optional).
            longitude:   GPS longitude (optional).

    Returns:
            A populated GuanoMetadata object.
    """
    import time as _time
    from datetime import datetime, timezone
    # ISO 8601 timestamp with local timezone offset.
    import time as _t
    utc_offset = _t.timezone if _t.daylight == 0 else _t.altzone
    tz_str = f"{'+' if -utc_offset >= 0 else '-'}{abs(-utc_offset // 3600):02d}:{abs(utc_offset % 3600 // 60):02d}"
    # Actually, let's use a simpler approach:
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

    return GuanoMetadata(
        version="1.0",
        timestamp=timestamp,
        latitude=latitude,
        longitude=longitude,
        species_auto_id=species,
        species_auto_p=confidence,
        detector_model="Bat Listener Python",
        samplerate=sample_rate,
        surveyor="",
        note=filename,
    )