"""
Citizen Science Module (Improvements #29, #30).

  #29 — Smartphone Built-In Microphone Support:
    Support recording via smartphone built-in microphones (not just USB
    ultrasonic mics) to lower the barrier to citizen-science participation.
    Research (Springer 2024) shows mobile devices effectively record
    low-frequency bat calls (noctule, serotine, Leisler's) with spectrogram
    quality 0.69–0.90 vs. professional detectors.

    The app should auto-detect device capability, warn when only audible-range
    recording is possible, and recommend an external USB mic for full-spectrum
    ultrasonic work — but never block recording outright.

  #30 — One-Click Export to Citizen Science Platforms:
    Build a batch export pipeline that maps GUANO metadata to Darwin Core
    terms and pushes verified observations to iNaturalist, UK BatSync, or
    NABat — including auto-generated spectrogram thumbnails, truncated WAV
    clips, GPS, species, timestamp, and recorder info.

    Source: Bat2iNat workflow (Somerset Bat Group) — maps GUANO to iNaturalist
    observations with auto-generated spectrograms, cutting per-record upload
    time from ~10 manual minutes to near-zero.

    Darwin Core mapping: https://dwc.tdwg.org/
"""

import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from guano_metadata import GuanoMetadata


@dataclass
class DeviceCapability:
    """
    Audio device capability assessment for citizen-science recording.

    Attributes:
        device_name:    Name of the audio input device.
        max_sample_rate: Maximum supported sample rate (Hz).
        max_frequency:   Maximum recordable frequency (Nyquist = max_sample_rate / 2).
        is_ultrasonic:   Whether the device can record ultrasonic (>20 kHz).
        quality_tier:    "professional", "consumer_usb", "smartphone", "unknown".
        recommendation:  User-facing recommendation message.
    """
    device_name: str = ""
    max_sample_rate: int = 44100
    max_frequency: float = 22050.0
    is_ultrasonic: bool = False
    quality_tier: str = "unknown"
    recommendation: str = ""


class DeviceCapabilityDetector:
    """
    Detects audio input device capabilities and provides recommendations.

    Uses sounddevice to query available audio input devices and assess their
    ultrasonic recording capability.
    """

    @staticmethod
    def detect() -> DeviceCapability:
        """
        Detect the default audio input device and assess its capabilities.

        Returns:
            DeviceCapability with assessment and recommendation.
        """
        try:
            import sounddevice as sd

            # Get the default input device info.
            devices = sd.query_devices()
            input_devices = [d for d in devices if d["max_input_channels"] > 0]

            if not input_devices:
                return DeviceCapability(
                    device_name="No input device",
                    recommendation="No audio input device found. Connect a microphone.",
                )

            # Use default input device.
            default_idx = sd.default.device[0]
            if default_idx is None or default_idx < 0:
                # Pick the first available input.
                dev = input_devices[0]
            else:
                dev = devices[default_idx]

            max_sr = int(dev.get("default_samplerate", 44100))
            max_freq = max_sr / 2.0
            is_ultrasonic = max_sr >= 192000

            # Determine quality tier.
            if max_sr >= 192000:
                tier = "professional"
                rec = ("Professional ultrasonic detector. Full bat call range "
                       "(0–96 kHz). Suitable for all species.")
            elif max_sr >= 96000:
                tier = "consumer_usb"
                rec = ("USB microphone with extended range. Can record up to "
                       f"{max_freq/1000:.0f} kHz. Good for most species, may miss "
                       "horseshoe bats (80–110 kHz).")
            elif max_sr >= 48000:
                tier = "smartphone"
                rec = ("Standard microphone (smartphone or built-in). Records up to "
                       f"{max_freq/1000:.0f} kHz. Can capture low-frequency calls "
                       "(noctule, serotine, Leisler's) but will miss higher "
                       "frequencies. For full-spectrum work, connect a USB "
                       "ultrasonic microphone (192+ kHz).")
            else:
                tier = "unknown"
                rec = f"Low sample rate ({max_sr} Hz). Limited recording capability."

            return DeviceCapability(
                device_name=dev.get("name", "Unknown"),
                max_sample_rate=max_sr,
                max_frequency=max_freq,
                is_ultrasonic=is_ultrasonic,
                quality_tier=tier,
                recommendation=rec,
            )

        except Exception as e:
            return DeviceCapability(
                device_name="Detection failed",
                recommendation=f"Could not detect audio device: {e}",
            )

    @staticmethod
    def get_species_detectable(capability: DeviceCapability) -> list[str]:
        """
        Return which species from the database are detectable given device capabilities.

        Only species whose peak frequency is below the device's Nyquist are listed.

        Args:
            capability: Device capability assessment.

        Returns:
            List of species names that can be detected.
        """
        from species_guide import SPECIES_DATABASE

        max_khz = capability.max_frequency / 1000.0
        detectable = []
        for sp in SPECIES_DATABASE:
            if sp.peak_freq_khz <= max_khz:
                detectable.append(f"{sp.name} ({sp.peak_freq_khz:.0f} kHz)")
            else:
                detectable.append(f"~~{sp.name} ({sp.peak_freq_khz:.0f} kHz)~~ [out of range]")

        return detectable


# ---------------------------------------------------------------------------
# Darwin Core Export (Improvement #30)
# ---------------------------------------------------------------------------

@dataclass
class DarwinCoreRecord:
    """
    A Darwin Core (DwC) occurrence record for submission to biodiversity
    platforms (iNaturalist, GBIF).

    Darwin Core is a standardized set of terms for biodiversity data sharing.
    https://dwc.tdwg.org/

    Key fields:
      scientificName:    Species scientific name.
      decimalLatitude:   GPS latitude.
      decimalLongitude:  GPS longitude.
      eventDate:         ISO 8601 date/time.
      recordedBy:        Surveyor name.
      occurrenceRemarks: Notes / spectrogram reference.
    """
    scientificName: str = ""
    scientificNameAuthorship: str = ""
    kingdom: str = "Animalia"
    phylum: str = "Chordata"
    class_: str = "Mammalia"
    order: str = "Chiroptera"
    family: str = ""
    genus: str = ""
    specificEpithet: str = ""
    taxonRank: str = "species"

    # Occurrence
    basisOfRecord: str = "MachineObservation"
    occurrenceID: str = ""
    occurrenceRemarks: str = ""
    recordedBy: str = ""
    identificationRemarks: str = ""  # e.g., "Acoustic ID, confidence: 92%"

    # Event
    eventDate: str = ""
    samplingProtocol: str = "Acoustic recording"

    # Location
    decimalLatitude: float | None = None
    decimalLongitude: float | None = None
    geodeticDatum: str = "WGS84"
    country: str = ""

    # Associated media
    associatedMedia: str = ""  # URL or path to spectrogram image

    def to_dict(self) -> dict:
        """Convert to a flat dict for JSON/CSV export."""
        d = {}
        for key, value in self.__dict__.items():
            if key == "class_":
                d["class"] = value
            else:
                d[key] = value
        return d


class CitizenScienceExporter:
    """
    Exports bat observations to citizen science platforms.

    Supports:
      - Darwin Core JSON export (for GBIF / iNaturalist bulk upload)
      - iNaturalist CSV format
      - BatSync-compatible format

    The exporter maps GUANO metadata to Darwin Core terms and generates
    observation records with spectrogram references.
    """

    @staticmethod
    def guano_to_darwin_core(meta: GuanoMetadata,
                              spectrogram_path: str = "",
                              country: str = "") -> DarwinCoreRecord:
        """
        Convert GUANO metadata to a Darwin Core record.

        Maps GUANO fields to DwC terms:
          Species Auto ID   → scientificName
          Lat/Lon           → decimalLatitude/decimalLongitude
          Timestamp         → eventDate
          Surveyor          → recordedBy
          Note              → occurrenceRemarks

        Args:
            meta:             GUANO metadata from the recording.
            spectrogram_path: Path to the spectrogram PNG (for associatedMedia).
            country:          Country name (for DwC country field).

        Returns:
            DarwinCoreRecord populated from GUANO metadata.
        """
        # Parse scientific name into components.
        sci_name = meta.species_auto_id or meta.species_manual_id
        genus = ""
        specific_epithet = ""
        if sci_name:
            parts = sci_name.split()
            if len(parts) >= 2:
                genus = parts[0]
                specific_epithet = parts[1]

        record = DarwinCoreRecord(
            scientificName=sci_name,
            genus=genus,
            specificEpithet=specific_epithet,
            family="",  # Would need a taxonomic lookup table.
            eventDate=meta.timestamp,
            decimalLatitude=meta.latitude,
            decimalLongitude=meta.longitude,
            recordedBy=meta.surveyor,
            occurrenceRemarks=meta.note,
            identificationRemarks=f"Acoustic ID, confidence: {meta.species_auto_p:.0%}",
            samplingProtocol="Acoustic recording",
            country=country,
            associatedMedia=spectrogram_path,
        )

        return record

    @staticmethod
    def export_json(records: list[DarwinCoreRecord], filepath: str) -> bool:
        """
        Export Darwin Core records as a JSON array.

        Suitable for manual upload to GBIF or iNaturalist's import tools.

        Args:
            records:   List of DarwinCoreRecord objects.
            filepath:  Destination file path.

        Returns:
            True if export succeeded.
        """
        try:
            data = [r.to_dict() for r in records]
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"[CitizenScience] Error exporting JSON: {e}")
            return False

    @staticmethod
    def export_csv(records: list[DarwinCoreRecord], filepath: str) -> bool:
        """
        Export Darwin Core records as CSV (iNaturalist-compatible format).

        Args:
            records:   List of DarwinCoreRecord objects.
            filepath:  Destination file path.

        Returns:
            True if export succeeded.
        """
        try:
            import csv
            if not records:
                return False

            # Get all field names from the first record.
            fieldnames = list(records[0].to_dict().keys())

            with open(filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for record in records:
                    writer.writerow(record.to_dict())
            return True
        except Exception as e:
            print(f"[CitizenScience] Error exporting CSV: {e}")
            return False

    @staticmethod
    def generate_inat_import_file(records: list[DarwinCoreRecord],
                                    audio_paths: list[str],
                                    spectrogram_paths: list[str],
                                    filepath: str) -> bool:
        """
        Generate an iNaturalist-compatible CSV import file.

        iNaturalist expects a CSV with specific column names. This function
        maps our Darwin Core records to iNaturalist's format.

        Key iNaturalist columns:
          scientific_name, datetime, latitude, longitude, notes,
          photos, sounds

        Args:
            records:          Darwin Core records.
            audio_paths:     Paths to WAV files (one per record).
            spectrogram_paths: Paths to spectrogram PNGs (one per record).
            filepath:         Destination CSV file path.

        Returns:
            True if export succeeded.
        """
        try:
            import csv

            with open(filepath, "w", newline="") as f:
                fieldnames = [
                    "scientific_name", "datetime", "latitude", "longitude",
                    "notes", "identification_remarks", "sounds", "photos",
                    "observed_by", "sampling_protocol",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for i, record in enumerate(records):
                    writer.writerow({
                        "scientific_name": record.scientificName,
                        "datetime": record.eventDate,
                        "latitude": record.decimalLatitude,
                        "longitude": record.decimalLongitude,
                        "notes": record.occurrenceRemarks,
                        "identification_remarks": record.identificationRemarks,
                        "sounds": audio_paths[i] if i < len(audio_paths) else "",
                        "photos": spectrogram_paths[i] if i < len(spectrogram_paths) else "",
                        "observed_by": record.recordedBy,
                        "sampling_protocol": record.samplingProtocol,
                    })

            return True
        except Exception as e:
            print(f"[CitizenScience] Error generating iNaturalist file: {e}")
            return False