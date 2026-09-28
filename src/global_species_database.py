"""
Unified Global Bat Species Database.

Merges all regional species databases into a single global database with
geographic range filtering. This module replaces the original 12-species
European-only database in species_guide.py with a comprehensive global database.

Regions covered:
  - Europe:    12 species (original species_guide.py)
  - North America: 49 species (SonoBat tables, NABat/USGS, BCI)
  - South America: 47 species (Jung et al., Schnitzler et al., Kalko et al.)
  - Africa:    (added when research completes)

Total: 108+ species across all continents where echolocating bats occur.

Each species entry includes:
  - name:           Common name
  - scientific:     Scientific name (genus + species)
  - peak_freq_khz:  Peak echolocation frequency (kHz)
  - freq_range:     Optional (min, max) frequency range in kHz
  - call_type:      Description of call sound/pattern
  - rhythm:         "regular" or "irregular"
  - call_category:  "FM" (frequency-modulated) or "CF" (constant-frequency)
  - family:         Taxonomic family
  - min_lat/max_lat/min_lon/max_lon: Geographic bounding box
  - continents:     List of continents where the species occurs

Usage:
    from global_species_database import GLOBAL_SPECIES_DATABASE, get_species_by_region

    # Get all species
    all_species = GLOBAL_SPECIES_DATABASE

    # Get species plausible at a location
    from geo_filter import GeoFilter
    plausible = GeoFilter.filter_species(50.0, 0.0)  # Uses the full database
"""

from species_guide import SPECIES_DATABASE, BatSpecies
from north_american_species import NORTH_AMERICAN_SPECIES
from south_american_species import SOUTH_AMERICAN_SPECIES


def _dict_to_batspecies(d: dict, family: str = "", region: str = "") -> BatSpecies:
    """
    Convert a species dict from a regional database to a BatSpecies dataclass.

    Args:
        d:      Species dict with keys matching the regional format.
        family: Taxonomic family (if not in the dict).
        region: Region name for reference.

    Returns:
        BatSpecies instance.
    """
    freq_range = d.get("freq_range")
    if freq_range and isinstance(freq_range, (list, tuple)) and len(freq_range) == 2:
        freq_range = (float(freq_range[0]), float(freq_range[1]))
    else:
        freq_range = None

    return BatSpecies(
        name=d["name"],
        scientific=d["scientific"],
        peak_freq_khz=float(d["peak_freq_khz"]),
        freq_range=freq_range,
        call_type=d.get("call_type", ""),
        rhythm=d.get("rhythm", "regular"),
        call_category=d.get("call_category", "FM"),
    )


# ---------------------------------------------------------------------------
# Build the unified global database
# ---------------------------------------------------------------------------

# European species (already in BatSpecies format, 12 species)
EUROPEAN_SPECIES: list[BatSpecies] = SPECIES_DATABASE.copy()

# North American species (convert from dict format, 49 species)
NORTH_AMERICAN_BATS: list[BatSpecies] = [
    _dict_to_batspecies(d, d.get("family", ""), "North America")
    for d in NORTH_AMERICAN_SPECIES
]

# South American species (convert from dict format, 47 species)
SOUTH_AMERICAN_BATS: list[BatSpecies] = [
    _dict_to_batspecies(d, d.get("family", ""), "South America")
    for d in SOUTH_AMERICAN_SPECIES
]

# Combined global database (deduplicated by scientific name)
_SEEN_SCIENTIFIC: set[str] = set()
GLOBAL_SPECIES_DATABASE: list[BatSpecies] = []

for species in EUROPEAN_SPECIES + NORTH_AMERICAN_BATS + SOUTH_AMERICAN_BATS:
    sci_key = species.scientific.lower()
    if sci_key not in _SEEN_SCIENTIFIC:
        GLOBAL_SPECIES_DATABASE.append(species)
        _SEEN_SCIENTIFIC.add(sci_key)


# ---------------------------------------------------------------------------
# Geographic range data for all species (for geo_filter.py)
# ---------------------------------------------------------------------------

# European species ranges (from species_range_data.py to avoid circular import)
from species_range_data import SpeciesRange, EUROPEAN_SPECIES_RANGES as _EU_RANGES

# Build range entries for North American species
_NA_RANGE_MAP: list[SpeciesRange] = []
for d in NORTH_AMERICAN_SPECIES:
    geo = d.get("geo_range", {})
    sp = next((s for s in NORTH_AMERICAN_BATS if s.scientific == d["scientific"]), None)
    if sp:
        _NA_RANGE_MAP.append(SpeciesRange(
            species=sp,
            min_lat=float(geo.get("min_lat", 7)),
            max_lat=float(geo.get("max_lat", 70)),
            min_lon=float(geo.get("min_lon", -168)),
            max_lon=float(geo.get("max_lon", -60)),
            continents=geo.get("continents", ["North America"]),
        ))

# Build range entries for South American species
_SA_RANGE_MAP: list[SpeciesRange] = []
for d in SOUTH_AMERICAN_SPECIES:
    sp = next((s for s in SOUTH_AMERICAN_BATS if s.scientific == d["scientific"]), None)
    if sp:
        _SA_RANGE_MAP.append(SpeciesRange(
            species=sp,
            min_lat=float(d.get("min_lat", -55)),
            max_lat=float(d.get("max_lat", 12)),
            min_lon=float(d.get("min_lon", -90)),
            max_lon=float(d.get("max_lon", -35)),
            continents=d.get("continents", ["South America"]),
        ))

# Combined global ranges
GLOBAL_SPECIES_RANGES: list[SpeciesRange] = _EU_RANGES + _NA_RANGE_MAP + _SA_RANGE_MAP


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def get_species_by_region(region: str) -> list[BatSpecies]:
    """
    Get species from a specific region.

    Args:
        region: "europe", "north_america", "south_america", "africa", or "all".

    Returns:
        List of BatSpecies from that region.
    """
    if region.lower() == "europe":
        return EUROPEAN_SPECIES
    elif region.lower() == "north_america":
        return NORTH_AMERICAN_BATS
    elif region.lower() == "south_america":
        return SOUTH_AMERICAN_BATS
    elif region.lower() == "all":
        return GLOBAL_SPECIES_DATABASE
    else:
        return []

def get_species_by_family(family: str) -> list[BatSpecies]:
    """
    Get all species from a specific taxonomic family.

    Args:
        family: Family name (e.g., "Vespertilionidae", "Molossidae").

    Returns:
        List of BatSpecies in that family.
    """
    # This requires the family info, which is in the dict-based databases.
    results = []
    for d in NORTH_AMERICAN_SPECIES + SOUTH_AMERICAN_SPECIES:
        if d.get("family", "").lower() == family.lower():
            sp = next((s for s in GLOBAL_SPECIES_DATABASE if s.scientific == d["scientific"]), None)
            if sp:
                results.append(sp)
    return results

def get_species_by_frequency_range(min_khz: float, max_khz: float) -> list[BatSpecies]:
    """
    Get all species whose peak frequency falls within a range.

    Useful for filtering by what a microphone can detect.

    Args:
        min_khz: Minimum peak frequency (kHz).
        max_khz: Maximum peak frequency (kHz).

    Returns:
        List of BatSpecies within the frequency range.
    """
    return [
        sp for sp in GLOBAL_SPECIES_DATABASE
        if min_khz <= sp.peak_freq_khz <= max_khz
    ]

def get_species_count() -> dict:
    """Return a summary of species counts by region and family."""
    counts = {
        "total": len(GLOBAL_SPECIES_DATABASE),
        "europe": len(EUROPEAN_SPECIES),
        "north_america": len(NORTH_AMERICAN_BATS),
        "south_america": len(SOUTH_AMERICAN_BATS),
    }

    # Count by call category
    fm_count = sum(1 for s in GLOBAL_SPECIES_DATABASE if s.call_category == "FM")
    cf_count = sum(1 for s in GLOBAL_SPECIES_DATABASE if s.call_category == "CF")
    counts["FM"] = fm_count
    counts["CF"] = cf_count

    # Frequency range stats
    freqs = [s.peak_freq_khz for s in GLOBAL_SPECIES_DATABASE]
    counts["min_freq_khz"] = min(freqs)
    counts["max_freq_khz"] = max(freqs)

    return counts

def find_species_match_global(peak_freq_hz: float, tolerance_khz: float = 5.0) -> BatSpecies | None:
    """
    Find the closest matching species from the global database.

    This replaces species_guide.find_species_match but searches all 108+ species.

    Args:
        peak_freq_hz:   Detected peak frequency in Hertz.
        tolerance_khz:  Match tolerance in kHz (default ±5 kHz).

    Returns:
        The closest matching BatSpecies, or None.
    """
    peak_khz = peak_freq_hz / 1000.0
    best_match: BatSpecies | None = None
    best_distance = float("inf")

    for species in GLOBAL_SPECIES_DATABASE:
        distance = abs(peak_khz - species.peak_freq_khz)
        if distance <= tolerance_khz and distance < best_distance:
            best_match = species
            best_distance = distance

    return best_match