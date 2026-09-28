"""
Settings persistence for the Bat Listener application.

This module handles saving and loading user preferences to a JSON file,
acting as the Python equivalent of the original HTML app's localStorage usage.
Settings include: selected frequency-shift method, expansion factor,
oscillator frequency, filter states, and FFT size.

The file lives at ~/.bat_listener_settings.json so it persists across sessions.
"""

import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Default settings — used on first launch or when the settings file is missing.
# These mirror the defaults from the original HTML application.
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    # Which frequency-shift method is active:
    #   "time_expansion"   — slowed playback (preserves harmonics)
    #   "heterodyne"       — mix with local oscillator (real-time)
    #   "frequency_division" — integer down-conversion (simplest)
    "method": "time_expansion",

    # Expansion factor for time expansion (1x–20x).
    # Also used as the divisor for frequency division (rounded to nearest valid integer).
    "method_value": 10,

    # FFT size for the spectrogram. Must be a power of 2.
    # Larger = better frequency resolution, slower update rate.
    "fft_size": 2048,

    # Audio filter configuration.
    # Each filter has an "active" toggle and a numeric parameter.
    "filters": {
        "highpass": {"active": False, "cutoff": 5000},       # Hz
        "lowpass":  {"active": False, "cutoff": 80000},      # Hz
        "bandpass": {"active": False, "center": 45000, "q": 5},  # Hz, dimensionless
        "noise_gate": {"active": False, "threshold": -40},   # dB
        "gain":     {"active": True, "gain": 1.0},           # linear multiplier
    },
}

# Path to the settings file in the user's home directory.
SETTINGS_PATH = Path.home() / ".bat_listener_settings.json"


def load_settings() -> dict:
    """
    Load settings from the JSON file, falling back to defaults if the file
    does not exist or is corrupted.

    Returns:
        A dictionary containing all application settings.
    """
    if not SETTINGS_PATH.exists():
        return DEFAULT_SETTINGS.copy()

    try:
        with open(SETTINGS_PATH, "r") as f:
            saved = json.load(f)
        # Merge with defaults so new keys are present even if the saved file
        # was written by an older version of the app.
        merged = DEFAULT_SETTINGS.copy()
        merged.update(saved)
        # Ensure nested filter dict has all expected keys.
        if "filters" in saved:
            for key in DEFAULT_SETTINGS["filters"]:
                if key in saved["filters"]:
                    merged["filters"][key] = saved["filters"][key]
        return merged
    except (json.JSONDecodeError, IOError):
        # Corrupted or unreadable file — use defaults.
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    """
    Save the current settings dictionary to the JSON file.

    Args:
        settings: The settings dictionary to persist.
    """
    try:
        with open(SETTINGS_PATH, "w") as f:
            json.dump(settings, f, indent=2)
    except IOError:
        # Silently fail — settings are a convenience, not critical.
        pass