"""
Species Range Data — shared dataclass and European species ranges.

This module is extracted from geo_filter.py to break a circular import:
  - global_species_database.py needs SpeciesRange and European ranges
  - geo_filter.py needs the global database

Both import from here without creating a cycle.
"""

from dataclasses import dataclass
from species_guide import BatSpecies, SPECIES_DATABASE


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
    continents: list


# European species ranges (bounding boxes from IUCN, simplified)
EUROPEAN_SPECIES_RANGES: list[SpeciesRange] = [
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