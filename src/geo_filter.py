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

This module ships a built-in species-range database for the 12 species
already in the species_guide, plus a simple lat/lon bounding-box check.
Range data is simplified to bounding boxes for offline use; a GBIF/IUCN
API lookup could provide more precise polygons for online use.
"""

from dataclasses import dataclass
from species_guide import SPECIES_DATABASE, BatSpecies


@dataclass
class SpeciesRange:
    """
    Geographic range for a bat species (simplified bounding box).

    For production use, these should be replaced with IUCN range polygons.
    The bounding box is a coarse approximation — some species have patchy
    distributions within their box.

    Attributes:
        species:     The BatSpecies this range applies to.
        min_lat:     Southern boundary (degrees, -90 to 90).
        max_lat:     Northern boundary (degrees, -90 to 90).
        min_lon:     Western boundary (degrees, -180 to 180).
        max_lon:     Eastern boundary (degrees, -180 to 180).
        continents:  List of continents where the species occurs.
    """
    species: BatSpecies
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float
    continents: list[str]


# ---------------------------------------------------------------------------
# Species Range Database
#
# Bounding boxes for the 12 European species in our database.
# Source: IUCN Red List range maps (simplified to bounding boxes).
# These cover the European/Mediterranean range of each species.
# ---------------------------------------------------------------------------
SPECIES_RANGES: list[SpeciesRange] = [
    SpeciesRange(SPECIES_DATABASE[0],   # Noctule
                 min_lat=35.0, max_lat=65.0, min_lon=-10.0, max_lon=60.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[1],   # Leisler's Bat
                 min_lat=25.0, max_lat=60.0, min_lon=-10.0, max_lon=70.0,
                 continents=["Europe", "Asia", "Africa"]),
    SpeciesRange(SPECIES_DATABASE[2],   # Serotine
                 min_lat=35.0, max_lat=58.0, min_lon=-10.0, max_lon=60.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[3],   # Barbastelle
                 min_lat=35.0, max_lat=62.0, min_lon=-10.0, max_lon=50.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[4],   # Nathusius' Pipistrelle
                 min_lat=35.0, max_lat=65.0, min_lon=-10.0, max_lon=70.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[5],   # Common Pipistrelle
                 min_lat=30.0, max_lat=65.0, min_lon=-10.0, max_lon=80.0,
                 continents=["Europe", "Asia", "Africa"]),
    SpeciesRange(SPECIES_DATABASE[6],   # Soprano Pipistrelle
                 min_lat=35.0, max_lat=65.0, min_lon=-10.0, max_lon=60.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[7],   # Daubenton's Bat
                 min_lat=35.0, max_lat=68.0, min_lon=-10.0, max_lon=140.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[8],   # Natterer's Bat
                 min_lat=35.0, max_lat=65.0, min_lon=-10.0, max_lon=60.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[9],   # Brown Long-eared
                 min_lat=35.0, max_lat=68.0, min_lon=-10.0, max_lon=140.0,
                 continents=["Europe", "Asia"]),
    SpeciesRange(SPECIES_DATABASE[10],  # Greater Horseshoe
                 min_lat=30.0, max_lat=53.0, min_lon=-10.0, max_lon=50.0,
                 continents=["Europe", "Asia", "Africa"]),
    SpeciesRange(SPECIES_DATABASE[11],  # Lesser Horseshoe
                 min_lat=30.0, max_lat=55.0, min_lon=-10.0, max_lon=50.0,
                 continents=["Europe", "Asia", "Africa"]),
]


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