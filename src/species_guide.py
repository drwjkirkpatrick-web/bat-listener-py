"""
Bat Species Identification Guide (Module 4).

This module contains a built-in reference database of 12 common bat species
and their echolocation call characteristics. It provides:

  - Species data: name, scientific name, peak frequency, call type, rhythm
  - Peak frequency matching: given a detected peak frequency, find the
    closest species within a ±5 kHz tolerance
  - Click-to-tune support: when a user selects a species, the heterodyne
    oscillator can be tuned to that species' peak frequency

The data is based on published echolocation studies and field guides for
European bat species. Frequencies are given in kHz (kilohertz).
"""

from dataclasses import dataclass


@dataclass
class BatSpecies:
    """
    A single bat species entry in the reference guide.

    Attributes:
        name:           Common name (e.g. "Common Pipistrelle").
        scientific:     Scientific name (genus + species).
        peak_freq_khz:  Peak echolocation frequency in kilohertz.
                        For species with a frequency range, this is the
                        midpoint of that range.
        freq_range:     Optional (min, max) range in kHz if the species
                        uses a band rather than a single peak.
        call_type:      Description of the call sound (e.g. "Wet slappy").
        rhythm:         "regular" or "irregular" — describes call spacing.
        call_category:  "FM" (frequency-modulated) or "CF" (constant-frequency).
                        Horseshoe bats use CF calls at a narrow frequency.
    """
    name: str
    scientific: str
    peak_freq_khz: float
    freq_range: tuple | None
    call_type: str
    rhythm: str
    call_category: str


# ---------------------------------------------------------------------------
# Species Database
#
# 12 species commonly found in Europe, covering the 20–120 kHz echolocation
# range. Organized from lowest to highest peak frequency.
# ---------------------------------------------------------------------------
SPECIES_DATABASE: list[BatSpecies] = [
    BatSpecies("Noctule",              "Nyctalus noctula",          22.5, (20, 25),   "Chip-chop",   "regular",   "FM"),
    BatSpecies("Leisler's Bat",        "Nyctalus leisleri",         25.0, None,       "Chip-chop",   "regular",   "FM"),
    BatSpecies("Serotine",             "Eptesicus serotinus",       27.0, (25, 29),   "Tock",        "irregular", "FM"),
    BatSpecies("Barbastelle",          "Barbastella barbastellus",  32.0, None,       "Call",        "regular",   "FM"),
    BatSpecies("Nathusius' Pipistrelle","Pipistrellus nathusii",    39.0, None,       "Wet slappy",  "regular",   "FM"),
    BatSpecies("Common Pipistrelle",   "Pipistrellus pipistrellus", 45.0, (42, 48),   "Wet slappy",  "irregular", "FM"),
    BatSpecies("Soprano Pipistrelle",  "Pipistrellus pygmaeus",     53.5, (52, 55),   "Wet slappy",  "irregular", "FM"),
    BatSpecies("Daubenton's Bat",      "Myotis daubentonii",        45.0, None,       "Call",        "regular",   "FM"),
    BatSpecies("Natterer's Bat",       "Myotis nattereri",          50.0, None,       "Call",        "regular",   "FM"),
    BatSpecies("Brown Long-eared",     "Plecotus auritus",          47.5, (45, 50),   "Call",        "regular",   "FM"),
    BatSpecies("Greater Horseshoe",    "Rhinolophus ferrumequinum", 80.0, None,       "CF call",     "regular",   "CF"),
    BatSpecies("Lesser Horseshoe",     "Rhinolophus hipposideros", 108.0, None,       "CF call",     "regular",   "CF"),
]


# Tolerance for peak-frequency matching, in kHz.
# If a detected peak is within this distance of a species' peak frequency,
# we consider it a "possible match."
MATCH_TOLERANCE_KHZ = 5.0


def find_species_match(peak_freq_hz: float) -> BatSpecies | None:
    """
    Given a detected peak frequency (in Hz), find the closest matching species.

    The function checks whether the peak falls within ±MATCH_TOLERANCE_KHZ
    of any species' peak frequency. If multiple species match, the closest
    one is returned.

    Args:
        peak_freq_hz: Detected peak frequency in Hertz.

    Returns:
        The matching BatSpecies, or None if no species is within tolerance.
    """
    peak_khz = peak_freq_hz / 1000.0
    best_match: BatSpecies | None = None
    best_distance = float("inf")

    for species in SPECIES_DATABASE:
        distance = abs(peak_khz - species.peak_freq_khz)
        if distance <= MATCH_TOLERANCE_KHZ and distance < best_distance:
            best_match = species
            best_distance = distance

    return best_match


def get_species_by_index(index: int) -> BatSpecies | None:
    """
    Get a species by its index in the database (for click handling).

    Args:
        index: Zero-based index into SPECIES_DATABASE.

    Returns:
        The BatSpecies at that index, or None if out of range.
    """
    if 0 <= index < len(SPECIES_DATABASE):
        return SPECIES_DATABASE[index]
    return None