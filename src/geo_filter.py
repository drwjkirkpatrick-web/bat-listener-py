"""
Geographic Species Filter (Improvements #26, #2).

Filters the species reference list to only those whose geographic ranges
overlap the recording location. This dramatically reduces false positives
from automated classifiers by excluding species that cannot physically
occur at the site.

Research basis:
  #26 — NABat processing guidance: "limit the potential species list by the
        potential geographic ranges of species" (USGS OFR 2018-1068).
  #2  — Colorado State University / USGS: CNN with range maps as geographic
        prior eliminates impossible predictions (Khalighifar et al. 2022).

This module uses the global species database (108+ species across Europe,
North America, South America, and Africa) with bounding-box geographic ranges.
Range data is simplified to bounding boxes for offline use; a GBIF/IUCN
API lookup could provide more precise polygons for online use.
"""

from dataclasses import dataclass
from species_guide import SPECIES_DATABASE as _EU_SPECIES, BatSpecies
from species_range_data import SpeciesRange, EUROPEAN_SPECIES_RANGES as _EU_RANGES

# Import the global database for ranges and species.
# This is done at module level so that SPECIES_DATABASE and SPECIES_RANGES
# are available to all functions.
try:
    from global_species_database import (
        GLOBAL_SPECIES_DATABASE as SPECIES_DATABASE,
        GLOBAL_SPECIES_RANGES as SPECIES_RANGES,
    )
    _USE_GLOBAL = True
except Exception:
    # Fall back to European-only database if global import fails.
    SPECIES_DATABASE = _EU_SPECIES
    SPECIES_RANGES = _EU_RANGES
    _USE_GLOBAL = False


# Note: SpeciesRange dataclass and SPECIES_RANGES are imported from
# species_range_data / global_species_database at the top of this file.


class GeoFilter:
    """
    Filters the species list by geographic range overlap.

    Given a recording location (lat/lon), this class returns only the species
    whose ranges include that location, reducing false positives in automated
    classification.
    """

    @staticmethod
    def filter_species(latitude: float, longitude: float) -> list[BatSpecies]:
        """
        Return only species whose geographic range includes the given location.

        Args:
            latitude:   Recording location latitude (degrees, +N).
            longitude:  Recording location longitude (degrees, +E).

        Returns:
            List of BatSpecies that can occur at this location.
        """
        if latitude == 0.0 and longitude == 0.0:
            # No GPS — return all species.
            return SPECIES_DATABASE.copy()

        filtered = []
        for sr in SPECIES_RANGES:
            if (sr.min_lat <= latitude <= sr.max_lat and
                sr.min_lon <= longitude <= sr.max_lon):
                filtered.append(sr.species)

        return filtered

    @staticmethod
    def filter_with_confidence(latitude: float, longitude: float,
                                species_name: str) -> tuple[bool, str]:
        """
        Check if a predicted species is plausible at the given location.

        Args:
            latitude:      Recording location latitude.
            longitude:     Recording location longitude.
            species_name:  Predicted species name (common or scientific).

        Returns:
            (is_plausible, reason) — True if species range includes location.
        """
        if latitude == 0.0 and longitude == 0.0:
            return True, "No GPS data — cannot filter by range"

        plausible_species = GeoFilter.filter_species(latitude, longitude)

        # Check if the predicted species is in the plausible list.
        for sp in plausible_species:
            if species_name.lower() in sp.name.lower() or \
               species_name.lower() in sp.scientific.lower():
                return True, f"{sp.name} range includes this location"

        # Find which species was rejected and where it actually occurs.
        for sr in SPECIES_RANGES:
            if species_name.lower() in sr.species.name.lower() or \
               species_name.lower() in sr.species.scientific.lower():
                return False, (
                    f"{sr.species.name} occurs at "
                    f"lat {sr.min_lat}-{sr.max_lat}, lon {sr.min_lon}-{sr.max_lon}; "
                    f"not at ({latitude}, {longitude})"
                )

        return False, f"Species '{species_name}' not in range database"

    @staticmethod
    def get_location_info(latitude: float, longitude: float) -> str:
        """
        Generate a human-readable location and species-plausibility summary.

        Args:
            latitude:   Recording location latitude.
            longitude:  Recording location longitude.

        Returns:
            Multi-line string with location info and plausible species count.
        """
        if latitude == 0.0 and longitude == 0.0:
            return "Location: not set (no GPS filter active)"

        plausible = GeoFilter.filter_species(latitude, longitude)

        lines = [
            f"Location: {latitude:.4f}°, {longitude:.4f}°",
            f"Plausible species at this location: {len(plausible)} of {len(SPECIES_DATABASE)}",
        ]

        if plausible:
            lines.append("Species in range:")
            for sp in plausible:
                lines.append(f"  • {sp.name} ({sp.peak_freq_khz:.0f} kHz)")

        return "\n".join(lines)