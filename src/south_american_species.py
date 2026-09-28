"""
South American Echolocating Bat Species Database.

Research compilation of echolocation parameters for bat species native to
South America (including Panama, Trinidad, and the Caribbean). Only
echolocating species (Microchiroptera / Yangochiroptera) are included;
fruit bats (Megachiroptera / Pteropodidae) are excluded.

Families covered:
  - Noctilionidae     (Noctilio)
  - Mormoopidae       (Pteronotus, Mormoops)
  - Phyllostomidae    (echolocating leaf-nosed bats)
  - Emballonuridae    (Saccopteryx, Cormura, Rhynchonycteris, Diclidurus, ...)
  - Molossidae        (Tadarida, Molossus, Eumops, Cynomops, Nyctinomops, ...)
  - Vespertilionidae  (Myotis, Lasiurus, Eptesicus, Rhogeessa, Histiotus)

Data sources (primary):
  - Jung et al. 2007, J Zool 272:125-137 (Emballonuridae, Central America)
  - Jung et al. 2014, PLoS ONE 9:e85279 (Molossidae, New World — 18 species)
  - Schnitzler et al. 1994, Behav Ecol Sociobiol 35:327-345 (Noctilio leporinus)
  - Kalko et al. 1998, Behav Ecol Sociobiol 42:305-319 (Noctilio albiventris)
  - Geipel et al. 2021, PNAS (phyllostomid gleaners, 12 species)
  - Surlykke & Kalko 2008, Front Physiol (Trachops, Carollia)
  - Brinkløv et al. 2010, J Exp Biol (Macrophyllum)
  - Mora et al. 2005, J Mammal (Cuban mormoopids)
  - O'Farrell & Miller 1997, J Mammal 78:954-963 (Neotropical bats)
  - Arias-Aguilar et al. 2018, Mammal Res (Brazilian bats)
  - Rodríguez-San Pedro et al. 2023 (Eumops perotis, Chile)
  - Smotherman & Guillén-Servent 2008, JASA 123:4331 (P. personatus DSC)
  - Ibáñez et al. 1999, J Mammal 80:924-928 (P. davyi, Panama)
  - Mora et al. 2013, J Mammal 87:255-265 (P. quadridens variation)
  - Essick et al. 2023, peerj 10591 (vertical stratification, Peru)
  - Obrist 1995, J Exp Biol 201:143-154 (Lasiurus borealis)
  - Barclay 1983, J Comp Physiol 151:515-520 (Emballonuridae, Panama)
  - Velazco et al. 2024, Rev Chil Hist Nat (S. antioquensis)

Frequencies are peak / dominant frequencies in kHz (frequency of maximum
energy, typically in the second harmonic for mormoopids and emballonurids).
freq_range gives (min, max) for species with a documented frequency band
or alternating frequencies; null for species with a narrow single peak.
"""

SOUTH_AMERICAN_SPECIES = [
    # ===================================================================
    # Noctilionidae — Bulldog Bats (2 species)
    # ===================================================================
    {
        "name": "Greater Fishing Bat",
        "scientific": "Noctilio leporinus",
        "peak_freq_khz": 55.0,
        "freq_range": (50.0, 60.0),
        "call_type": "Long CF-FM; CF at 52.8-56.2 kHz with terminal FM sweep ~26 kHz bandwidth; pure CF signals up to 17 ms",
        "rhythm": "regular",
        "call_category": "CF",
        "min_lat": -35.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Lesser Bulldog Bat",
        "scientific": "Noctilio albiventris",
        "peak_freq_khz": 70.0,
        "freq_range": (67.0, 72.0),
        "call_type": "CF-FM; qCF at 67-72 kHz with FM component ~32 kHz bandwidth; also long FM-only signals 15-21 ms",
        "rhythm": "regular",
        "call_category": "CF",
        "min_lat": -35.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },

    # ===================================================================
    # Mormoopidae — Mustached & Ghost-faced Bats (7 species)
    # ===================================================================
    {
        "name": "Parnell's Mustached Bat",
        "scientific": "Pteronotus parnellii",
        "peak_freq_khz": 60.0,
        "freq_range": (58.0, 65.0),
        "call_type": "Long CF-FM; 4-5 harmonics, CF in 2nd harmonic at ~60-65 kHz, long CF component ~20 ms with brief FM sweeps; Doppler-shift compensated",
        "rhythm": "regular",
        "call_category": "CF",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Big Naked-backed Bat",
        "scientific": "Pteronotus gymnonotus",
        "peak_freq_khz": 57.0,
        "freq_range": (53.0, 61.0),
        "call_type": "sCF-FM; short ascending FM then CF at 55-61 kHz in 2nd harmonic, terminal FM to ~45 kHz; 4-5 ms duration",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Wagner's Lesser Mustached Bat",
        "scientific": "Pteronotus personatus",
        "peak_freq_khz": 80.0,
        "freq_range": (65.0, 85.0),
        "call_type": "sCF-FM with dual CF; initial CF2 at ~80 kHz, downward FM sweep, terminal CF2 at ~65 kHz; ~5 ms, Doppler-shift compensated",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 15.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America"],
    },
    {
        "name": "Davy's Naked-backed Bat",
        "scientific": "Pteronotus davyi",
        "peak_freq_khz": 68.0,
        "freq_range": (51.0, 74.0),
        "call_type": "sCF-FM; initial CF at 67-68 kHz, downward FM sweep, terminal QCF at 51-58 kHz; 4.6-6.7 ms, 2nd harmonic dominant",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -10.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -60.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Sooty Mustached Bat",
        "scientific": "Pteronotus quadridens",
        "peak_freq_khz": 82.0,
        "freq_range": (68.0, 84.0),
        "call_type": "QCF-FM; QCF at 81-84 kHz in 2nd harmonic, downward FM sweep with 15-16 kHz bandwidth; 3.9-4.4 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": 17.0, "max_lat": 27.0,
        "min_lon": -85.0, "max_lon": -65.0,
        "continents": ["Caribbean"],
    },
    {
        "name": "Macleay's Mustached Bat",
        "scientific": "Pteronotus macleayii",
        "peak_freq_khz": 70.0,
        "freq_range": (55.0, 71.0),
        "call_type": "sCF-FM; short CF at ~70 kHz in 2nd harmonic followed by downward FM; similar to P. quadridens but lower frequency",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": 19.0, "max_lat": 23.0,
        "min_lon": -85.0, "max_lon": -74.0,
        "continents": ["Caribbean"],
    },
    {
        "name": "Antillean Ghost-faced Bat",
        "scientific": "Mormoops blainvillei",
        "peak_freq_khz": 60.0,
        "freq_range": (52.0, 68.0),
        "call_type": "Steep FM downsweep; 52.5-68.4 kHz, short duration, variable slope; multiharmonic",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": 17.0, "max_lat": 27.0,
        "min_lon": -85.0, "max_lon": -65.0,
        "continents": ["Caribbean"],
    },

    # ===================================================================
    # Phyllostomidae — Echolocating Leaf-nosed Bats (7 species)
    # ===================================================================
    {
        "name": "Seba's Short-tailed Bat",
        "scientific": "Carollia perspicillata",
        "peak_freq_khz": 90.0,
        "freq_range": (80.0, 100.0),
        "call_type": "Short multiharmonic FM sweep; peak at ~90 kHz, 0.5-1 ms duration, low intensity ('whispering'); 2nd/3rd harmonic dominant",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Greater Spear-nosed Bat",
        "scientific": "Phyllostomus hastatus",
        "peak_freq_khz": 45.0,
        "freq_range": (40.0, 50.0),
        "call_type": "Broadband FM sweep from 80 to 40 kHz with most energy below 50 kHz; relatively high intensity for a phyllostomid; >5 ms in open space",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Pale Spear-nosed Bat",
        "scientific": "Phyllostomus discolor",
        "peak_freq_khz": 50.0,
        "freq_range": (45.0, 55.0),
        "call_type": "Short multiharmonic FM sweep; lower peak frequency than P. hastatus, broadband; omnivore guild has lowest min/peak frequencies among phyllostomids",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Fringe-lipped Bat",
        "scientific": "Trachops cirrhosus",
        "peak_freq_khz": 90.0,
        "freq_range": (80.0, 110.0),
        "call_type": "Short multiharmonic FM sweep; main energy in 3rd harmonic at ~90 kHz, <1 ms duration, very low intensity (<70 dB SPL at 10 cm); highly directional beam",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Long-legged Bat",
        "scientific": "Macrophyllum macrophyllum",
        "peak_freq_khz": 87.0,
        "freq_range": (80.0, 95.0),
        "call_type": "Short multiharmonic FM sweep; centroid frequency ~87 kHz in field, 2.5 ms duration, higher intensity than other phyllostomids; unique trawling phyllostomid with search/approach/buzz pattern",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Hairy Big-eared Bat",
        "scientific": "Micronycteris microtis",
        "peak_freq_khz": 98.0,
        "freq_range": (90.0, 105.0),
        "call_type": "Short steep FM sweep; highest peak frequency among measured phyllostomid gleaners at 97.6 kHz, 76 kHz bandwidth, steepest sweep rate (132.6 kHz/ms); active gleaning predator",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Davies' Big-eared Bat",
        "scientific": "Glyphonycteris daviesi",
        "peak_freq_khz": 68.0,
        "freq_range": (65.0, 75.0),
        "call_type": "Short FM sweep; lowest peak frequency among measured phyllostomid gleaners at 68.1 kHz, narrowest bandwidth (32.7 kHz), shallowest sweep rate (47.6 kHz/ms); ~0.55 ms duration",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 12.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America"],
    },

    # ===================================================================
    # Emballonuridae — Sac-winged & Sheath-tailed Bats (6 species)
    # ===================================================================
    {
        "name": "Greater Sac-winged Bat",
        "scientific": "Saccopteryx bilineata",
        "peak_freq_khz": 45.0,
        "freq_range": (43.0, 47.0),
        "call_type": "QCF-FM with frequency alternation; 2nd harmonic at 43/47 kHz (low/high alternation), central narrowband with initial and terminal FM sweeps; ~8-9 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Lesser Sac-winged Bat",
        "scientific": "Saccopteryx leptura",
        "peak_freq_khz": 52.0,
        "freq_range": (47.0, 54.0),
        "call_type": "QCF-FM with frequency alternation; 2nd harmonic at 47.5/54 kHz (low/high alternation), multi-harmonic; forages in edge/gap space",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Antioquian Sac-winged Bat",
        "scientific": "Saccopteryx antioquensis",
        "peak_freq_khz": 56.0,
        "freq_range": (55.0, 59.0),
        "call_type": "QCF-FM with frequency alternation; inverted U-shaped spectrograms, 2nd harmonic at 55.2-57.1 kHz (LF) and 54.5-59.4 kHz (HF); ~3.3-3.8 ms, BW ~2.5 kHz",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": 5.0, "max_lat": 8.0,
        "min_lon": -77.0, "max_lon": -74.0,
        "continents": ["South America"],
    },
    {
        "name": "Chestnut Sac-winged Bat",
        "scientific": "Cormura brevirostris",
        "peak_freq_khz": 27.0,
        "freq_range": (25.0, 32.0),
        "call_type": "QCF-FM ascending triplets ('do-re-mi bat'); 3 alternating frequencies at 25.4, 28.7, 32.1 kHz in 2nd harmonic; long calls ~9-10 ms, open-space forager",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 15.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America"],
    },
    {
        "name": "Proboscis Bat",
        "scientific": "Rhynchonycteris naso",
        "peak_freq_khz": 100.0,
        "freq_range": (67.0, 100.0),
        "call_type": "QCF at ~100 kHz in 2nd harmonic during search/approach, drops to ~67 kHz in terminal buzz; very high frequency for edge-space foraging over water",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Northern Ghost Bat",
        "scientific": "Diclidurus albus",
        "peak_freq_khz": 22.0,
        "freq_range": (20.0, 25.0),
        "call_type": "Low-frequency QCF; ~22 kHz, very long pulse durations and long pulse intervals; open-space forager flying high above ground; near-audible to humans",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },

    # ===================================================================
    # Molossidae — Free-tailed Bats (18 species)
    # Data primarily from Jung et al. 2014, PLoS ONE — 18 species
    # ===================================================================
    {
        "name": "Brazilian Free-tailed Bat",
        "scientific": "Tadarida brasiliensis",
        "peak_freq_khz": 26.0,
        "freq_range": (24.0, 28.0),
        "call_type": "QCF downsweep; SF 27.6 / EF 24.4 kHz, BW 3.2 kHz, ~14 ms; highly plastic, also emits calls at 40-75 kHz in approach; open-space forager at high altitudes",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -55.0, "max_lat": 50.0,
        "min_lon": -125.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Velvety Free-tailed Bat",
        "scientific": "Molossus molossus",
        "peak_freq_khz": 39.0,
        "freq_range": (33.0, 43.0),
        "call_type": "QCF downsweep with frequency alternation; 3 alternating peaks at 35.6/39.1/42.8 kHz, BW ~2 kHz, ~10 ms; very narrow band",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Black Free-tailed Bat",
        "scientific": "Molossus rufus",
        "peak_freq_khz": 27.0,
        "freq_range": (25.0, 28.0),
        "call_type": "QCF downsweep with frequency alternation; 2 peaks at 26.3/27.8 kHz, BW ~2 kHz, ~13 ms; narrow band, open-space",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 15.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America"],
    },
    {
        "name": "Creager's Free-tailed Bat",
        "scientific": "Molossus currentium",
        "peak_freq_khz": 32.0,
        "freq_range": (24.0, 35.0),
        "call_type": "QCF downsweep with frequency alternation; 3 peaks at 29.7/32.9/35.1 kHz, BW ~4 kHz, ~14 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Cinnamon Dog-faced Bat",
        "scientific": "Cynomops planirostris",
        "peak_freq_khz": 29.0,
        "freq_range": (21.0, 33.0),
        "call_type": "QCF downsweep with frequency alternation; 2 peaks at 28.8/32.9 kHz, BW ~8 kHz, ~16 ms; wider bandwidth than Molossus",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Greenhall's Dog-faced Bat",
        "scientific": "Cynomops greenhalli",
        "peak_freq_khz": 27.0,
        "freq_range": (17.0, 29.0),
        "call_type": "QCF downsweep with frequency alternation; 2 peaks at 25.2/29.0 kHz, BW ~8 kHz, ~15 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Broad-eared Bat",
        "scientific": "Nyctinomops laticaudatus",
        "peak_freq_khz": 28.0,
        "freq_range": (24.0, 32.0),
        "call_type": "QCF downsweep with frequency alternation; 3 peaks at 26.7/28.7/32.4 kHz, BW ~5 kHz, ~12 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Big Free-tailed Bat",
        "scientific": "Nyctinomops macrotis",
        "peak_freq_khz": 24.0,
        "freq_range": (17.0, 29.0),
        "call_type": "QCF downsweep; SF 28.8 / EF 16.7 kHz, BW 12 kHz, ~13 ms; largest molossid in Cuba, calls below 20-25 kHz (near-audible); feeds on large moths",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 40.0,
        "min_lon": -120.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Wagner's Bonneted Bat",
        "scientific": "Eumops glaucinus",
        "peak_freq_khz": 28.0,
        "freq_range": (19.0, 29.0),
        "call_type": "FM-QCF downsweep with frequency alternation; 2 peaks at 27.4/29.3 kHz, BW ~8.5 kHz, ~16 ms; low frequency, near-audible range",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America", "North America", "Caribbean"],
    },
    {
        "name": "Black Bonneted Bat",
        "scientific": "Eumops auripendulus",
        "peak_freq_khz": 34.0,
        "freq_range": (18.0, 36.0),
        "call_type": "FM-QCF downsweep with frequency alternation; 2 peaks at 32.4/35.8 kHz, BW ~14 kHz, ~20 ms; broad bandwidth for a molossid",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Dabbene's Mastiff Bat",
        "scientific": "Eumops dabbenei",
        "peak_freq_khz": 23.0,
        "freq_range": (14.0, 25.0),
        "call_type": "FM-QCF downsweep with frequency alternation; 2 peaks at 21.3/24.6 kHz, BW ~8 kHz, ~27 ms; very low frequency, long duration; large molossid",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 12.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America"],
    },
    {
        "name": "Western Mastiff Bat",
        "scientific": "Eumops perotis",
        "peak_freq_khz": 12.7,
        "freq_range": (9.0, 15.0),
        "call_type": "QCF downsweep; SF 14.7 / EF 10.3 kHz, peak 12.7 kHz, ~17 ms; lowest frequency of any New World bat, audible to humans; largest New World molossid",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 40.0,
        "min_lon": -120.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Dwarf Bonneted Bat",
        "scientific": "Eumops nanus",
        "peak_freq_khz": 28.0,
        "freq_range": (25.0, 31.0),
        "call_type": "FM-QCF downsweep with frequency alternation; 2 peaks at 27.9/30.5 kHz, BW ~3 kHz, ~15 ms; narrow band, smallest Eumops",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -10.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Big Crested Mastiff Bat",
        "scientific": "Promops centralis",
        "peak_freq_khz": 28.0,
        "freq_range": (25.0, 36.0),
        "call_type": "FM up-QCF with alternating call types; low type ascending FM to 28 kHz QCF, high type descending FM at 35.7 kHz; ~18 ms; unique upward modulation",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Brown Mastiff Bat",
        "scientific": "Promops nasutus",
        "peak_freq_khz": 33.0,
        "freq_range": (33.0, 47.0),
        "call_type": "FM up-QCF; ascending FM to 34.7 kHz QCF, with rare high descending type at 47 kHz; ~12 ms; upward-modulated call design shared with Promops",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 12.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America"],
    },
    {
        "name": "Mato Grosso Dog-faced Bat",
        "scientific": "Neoplatymops mattogrossensis",
        "peak_freq_khz": 33.0,
        "freq_range": (28.0, 37.0),
        "call_type": "QCF downsweep with frequency alternation; 2 peaks at 32.6/36.9 kHz, BW ~4 kHz, ~12 ms; very short FM component",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -20.0, "max_lat": 5.0,
        "min_lon": -65.0, "max_lon": -35.0,
        "continents": ["South America"],
    },
    {
        "name": "Dwarf Dog-faced Bat",
        "scientific": "Molossops temminckii",
        "peak_freq_khz": 52.0,
        "freq_range": (45.0, 55.0),
        "call_type": "FM up-QCF; ascending FM sweep to ~52-55 kHz QCF; highest calling frequency among molossids, adapted for detecting small beetles; ~8 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 5.0,
        "min_lon": -72.0, "max_lon": -35.0,
        "continents": ["South America"],
    },
    {
        "name": "Apathetic Dog-faced Bat",
        "scientific": "Molossops neglectus",
        "peak_freq_khz": 38.0,
        "freq_range": (32.0, 56.0),
        "call_type": "FM up-QCF with alternating call types; low type ascending to 44 kHz, high type ascending to 47 kHz, rare high-II descending at 56 kHz; ~10 ms",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 5.0,
        "min_lon": -65.0, "max_lon": -35.0,
        "continents": ["South America"],
    },

    # ===================================================================
    # Vespertilionidae — Vesper / Evening Bats (7 species)
    # ===================================================================
    {
        "name": "Black Myotis",
        "scientific": "Myotis nigricans",
        "peak_freq_khz": 55.0,
        "freq_range": (48.0, 65.0),
        "call_type": "FM downsweep; open-space calls peak 54 kHz, terminal 51 kHz, BW ~11 kHz, ~7 ms; edge/gap calls peak 55 kHz, terminal 52 kHz, BW up to 40 kHz broadband; adapts call to habitat",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -25.0, "max_lat": 25.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Hoary Bat",
        "scientific": "Lasiurus cinereus",
        "peak_freq_khz": 18.0,
        "freq_range": (17.0, 20.0),
        "call_type": "Low QCF; search calls 20-17 kHz, essentially constant frequency, single harmonic; long-range detection in open air; very low frequency for large-bodied fast-flying bat",
        "rhythm": "regular",
        "call_category": "FM",
        "min_lat": -55.0, "max_lat": 60.0,
        "min_lon": -130.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Western Red Bat",
        "scientific": "Lasiurus blossevillii",
        "peak_freq_khz": 40.0,
        "freq_range": (30.0, 50.0),
        "call_type": "FM downsweep; search calls sweep 45-30 kHz with peak at ~35-40 kHz, approach calls 65-35 kHz, terminal 70-30 kHz; alternating 'plip-plop' calls in open space",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 50.0,
        "min_lon": -125.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
    {
        "name": "Thomas's Yellow Bat",
        "scientific": "Rhogeessa io",
        "peak_freq_khz": 55.0,
        "freq_range": (50.0, 60.0),
        "call_type": "Short broadband FM; maximum energy at 50-60 kHz, lowest energy at 40-50 kHz; crepuscular, flies low along trails and streams",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -15.0, "max_lat": 15.0,
        "min_lon": -82.0, "max_lon": -50.0,
        "continents": ["South America", "Central America", "Caribbean"],
    },
    {
        "name": "Black-winged Little Yellow Bat",
        "scientific": "Rhogeessa tumida",
        "peak_freq_khz": 55.0,
        "freq_range": (50.0, 60.0),
        "call_type": "Short broadband FM; maximum energy at 50-60 kHz, lowest energy at 40-50 kHz; flies low along streams, crepuscular activity peaks",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": 10.0, "max_lat": 22.0,
        "min_lon": -92.0, "max_lon": -77.0,
        "continents": ["Central America", "North America"],
    },
    {
        "name": "Small Big-eared Brown Bat",
        "scientific": "Histiotus montanus",
        "peak_freq_khz": 25.0,
        "freq_range": (18.0, 30.0),
        "call_type": "FM downsweep; calls dominated by frequencies below 20-30 kHz, large ears adapted for low-frequency hearing; aerial insectivore in temperate and tropical zones",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -55.0, "max_lat": 10.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America"],
    },
    {
        "name": "Brazilian Big Brown Bat",
        "scientific": "Eptesicus brasiliensis",
        "peak_freq_khz": 30.0,
        "freq_range": (25.0, 35.0),
        "call_type": "FM downsweep; typical vespertilionid call, peak ~28-32 kHz; medium-sized aerial insectivore, forages in edge and gap spaces",
        "rhythm": "irregular",
        "call_category": "FM",
        "min_lat": -35.0, "max_lat": 22.0,
        "min_lon": -82.0, "max_lon": -35.0,
        "continents": ["South America", "Central America", "North America"],
    },
]


if __name__ == "__main__":
    print(f"Total South American species: {len(SOUTH_AMERICAN_SPECIES)}")

    # Print summary organized by family
    families = {
        "Noctilionidae": ["Noctilio"],
        "Mormoopidae": ["Pteronotus", "Mormoops"],
        "Phyllostomidae": ["Carollia", "Phyllostomus", "Trachops",
                           "Macrophyllum", "Micronycteris", "Glyphonycteris"],
        "Emballonuridae": ["Saccopteryx", "Cormura", "Rhynchonycteris",
                           "Diclidurus"],
        "Molossidae": ["Tadarida", "Molossus", "Cynomops", "Nyctinomops",
                       "Eumops", "Promops", "Neoplatymops", "Molossops"],
        "Vespertilionidae": ["Myotis", "Lasiurus", "Rhogeessa",
                             "Histiotus", "Eptesicus"],
    }

    for family, genera in families.items():
        count = sum(
            1 for s in SOUTH_AMERICAN_SPECIES
            if any(s["scientific"].startswith(g) for g in genera)
        )
        if count:
            print(f"  {family}: {count} species")

    # Frequency range coverage
    freqs = [s["peak_freq_khz"] for s in SOUTH_AMERICAN_SPECIES]
    print(f"\nFrequency range: {min(freqs):.1f} - {max(freqs):.1f} kHz")
    print(f"CF species: "
          f"{sum(1 for s in SOUTH_AMERICAN_SPECIES if s['call_category'] == 'CF')}")
    print(f"FM species: "
          f"{sum(1 for s in SOUTH_AMERICAN_SPECIES if s['call_category'] == 'FM')}")