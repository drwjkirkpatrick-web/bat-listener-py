"""
Activity Metrics & Visualization (Improvements #23, #27, #28).

  #23 — Automated Feeding Buzz Detection:
    Detects terminal-phase echolocation "feeding buzzes" — rapid sequences of
    short pulses bats emit when closing on prey. Buzzes serve as a foraging
    activity index beyond mere presence/absence.
    Source: Buzzfindr (PMC11335113); BatSpot CNN (biorxiv 2026.03.11).

  #27 — Temporal Pass Plots (TPP):
    Fine-scale bat passes per time interval (e.g. per minute) with
    sunset/sunrise overlays. Comparable between sites/dates.
    Source: Ecological Indicators 2020, doi:10.1016/j.ecolind.2020.106202.

  #28 — Activity Heatmaps & Seasonal Detection Charts:
    Calendar heatmaps (night × species activity intensity) and seasonal trend
    charts modeled on BatAMP (Bat Acoustic Monitoring Portal).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, date


@dataclass
class FeedingBuzz:
    """
    A detected feeding buzz event.

    A feeding buzz is characterized by:
      - Rapid pulse rate (typically >10 pulses/second, up to 200/sec)
      - Short pulse duration (< 2ms in terminal phase)
      - Decreasing frequency and amplitude
    """
    start_time_s: float = 0.0      # When the buzz starts (seconds from recording start)
    end_time_s: float = 0.0        # When the buzz ends
    num_pulses: int = 0            # Number of pulses in the buzz
    pulse_rate_hz: float = 0.0     # Average pulse rate (pulses/second)
    peak_freq_khz: float = 0.0     # Peak frequency during the buzz


@dataclass
class ActivitySession:
    """
    A single recording session's activity data for visualization.

    Used to build Temporal Pass Plots and Activity Heatmaps.
    """
    date: Optional[date] = None
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0
    sunset_time: Optional[datetime] = None
    sunrise_time: Optional[datetime] = None

    # Per-minute pass counts (index = minute from recording start).
    passes_per_minute: list[int] = field(default_factory=list)

    # Per-species detection counts.
    species_counts: dict[str, int] = field(default_factory=dict)

    # Detected feeding buzzes.
    feeding_buzzes: list[FeedingBuzz] = field(default_factory=list)

    @property
    def total_passes(self) -> int:
        return sum(self.passes_per_minute)

    @property
    def total_buzzes(self) -> int:
        return len(self.feeding_buzzes)

    @property
    def total_minutes(self) -> int:
        return len(self.passes_per_minute)


class FeedingBuzzDetector:
    """
    Detects feeding buzzes from call timestamp sequences.

    A feeding buzz is identified by:
      1. A rapid increase in pulse rate (from ~5–10 Hz to >15 Hz)
      2. The pulse rate remains elevated for ≥0.5 seconds
      3. The buzz ends when the pulse rate drops back below the buzz threshold

    The detector works on inter-pulse intervals (IPIs):
      - Normal search phase: IPI ~ 50–200 ms (5–20 Hz)
      - Approach phase: IPI ~ 20–50 ms (20–50 Hz)
      - Terminal/buzz phase: IPI < 20 ms (>50 Hz)

    Source: Buzzfindr (PMC11335113) — pulse rate threshold for buzz detection.
    """

    # IPI threshold for buzz phase (ms). Below this = buzz.
    BUZZ_IPI_THRESHOLD_MS = 20.0

    # Minimum number of consecutive buzz-rate pulses to qualify as a buzz.
    MIN_BUZZ_PULSES = 5

    # Minimum buzz duration (ms).
    MIN_BUZZ_DURATION_MS = 200.0

    def detect(self, call_timestamps_s: list[float]) -> list[FeedingBuzz]:
        """
        Detect feeding buzzes from call onset timestamps.

        Args:
            call_timestamps_s: Sorted list of call onset times (seconds).

        Returns:
            List of detected FeedingBuzz events.
        """
        if len(call_timestamps_s) < self.MIN_BUZZ_PULSES + 1:
            return []

        timestamps = sorted(call_timestamps_s)
        buzzes: list[FeedingBuzz] = []

        # Compute inter-pulse intervals.
        ipis_ms = [
            (timestamps[i + 1] - timestamps[i]) * 1000.0
            for i in range(len(timestamps) - 1)
        ]

        # Scan for buzz sequences.
        i = 0
        while i < len(ipis_ms):
            if ipis_ms[i] <= self.BUZZ_IPI_THRESHOLD_MS:
                # Start of a potential buzz.
                buzz_start_idx = i
                buzz_end_idx = i

                # Extend the buzz while IPIs remain below threshold.
                while (buzz_end_idx < len(ipis_ms) and
                       ipis_ms[buzz_end_idx] <= self.BUZZ_IPI_THRESHOLD_MS):
                    buzz_end_idx += 1

                # Number of pulses in this buzz = (buzz_end_idx - buzz_start_idx) + 1.
                num_pulses = (buzz_end_idx - buzz_start_idx) + 1

                # Duration in ms.
                buzz_start_s = timestamps[buzz_start_idx]
                buzz_end_s = timestamps[buzz_end_idx]
                duration_ms = (buzz_end_s - buzz_start_s) * 1000.0

                # Check minimum criteria.
                if (num_pulses >= self.MIN_BUZZ_PULSES and
                    duration_ms >= self.MIN_BUZZ_DURATION_MS):
                    pulse_rate = num_pulses / (buzz_end_s - buzz_start_s)
                    buzzes.append(FeedingBuzz(
                        start_time_s=buzz_start_s,
                        end_time_s=buzz_end_s,
                        num_pulses=num_pulses,
                        pulse_rate_hz=pulse_rate,
                    ))

                i = buzz_end_idx
            else:
                i += 1

        return buzzes


class TemporalPassPlot:
    """
    Temporal Pass Plot (TPP) data generator.

    A TPP shows bat passes per time interval (typically per minute) over
    the recording period, with sunset/sunrise times overlaid.

    TPPs are directly comparable between sites, dates, and species because
    they use fixed-length time bins.

    Source: Ecological Indicators 2020, doi:10.1016/j.ecolind.2020.106202
    """

    @staticmethod
    def build(pass_timestamps_s: list[float],
              bin_size_min: float = 1.0,
              total_duration_s: float = 0.0) -> list[int]:
        """
        Build per-bin pass counts for a Temporal Pass Plot.

        A "pass" is defined as one or more calls occurring within a 1-second
        window (calls closer than 1 second = same pass).

        Args:
            pass_timestamps_s: Sorted list of call timestamps (seconds).
            bin_size_min:       Bin size in minutes (default 1 minute).
            total_duration_s:   Total recording duration. If 0, uses the last timestamp.

        Returns:
            List of pass counts per bin (index 0 = first bin).
        """
        if not pass_timestamps_s:
            return []

        if total_duration_s <= 0:
            total_duration_s = max(pass_timestamps_s) + 1.0

        bin_size_s = bin_size_min * 60.0
        n_bins = max(1, int(total_duration_s / bin_size_s) + 1)

        # Group calls into passes (calls within 1 second = 1 pass).
        passes = []
        sorted_ts = sorted(pass_timestamps_s)
        current_pass_start = sorted_ts[0]
        for ts in sorted_ts[1:]:
            if ts - current_pass_start > 1.0:
                passes.append(current_pass_start)
                current_pass_start = ts
        passes.append(current_pass_start)

        # Count passes per bin.
        bins = [0] * n_bins
        for pass_time in passes:
            bin_idx = int(pass_time / bin_size_s)
            if 0 <= bin_idx < n_bins:
                bins[bin_idx] += 1

        return bins

    @staticmethod
    def to_display_string(passes_per_min: list[int],
                          sunset_bin: int | None = None,
                          sunrise_bin: int | None = None) -> str:
        """
        Format the TPP as a text bar chart for the canvas.

        Args:
            passes_per_min:  Pass counts per minute (from build()).
            sunset_bin:     Minute index when sunset occurs (optional overlay).
            sunrise_bin:    Minute index when sunrise occurs (optional overlay).

        Returns:
            Multi-line text bar chart.
        """
        if not passes_per_min:
            return "No activity data."

        max_passes = max(passes_per_min) if passes_per_min else 1
        if max_passes == 0:
            max_passes = 1

        lines = ["Temporal Pass Plot (passes/min):"]
        for i, count in enumerate(passes_per_min):
            bar_len = int((count / max_passes) * 40)
            bar = "█" * bar_len
            marker = ""
            if sunset_bin is not None and i == sunset_bin:
                marker += " [SUNSET]"
            if sunrise_bin is not None and i == sunrise_bin:
                marker += " [SUNRISE]"
            lines.append(f"  {i:3d}m |{bar:40s}| {count:3d}{marker}")

        return "\n".join(lines)


class ActivityHeatmap:
    """
    Activity Heatmap data generator for seasonal visualization.

    Creates a calendar-style heatmap where:
      - X-axis = survey dates (weeks or days)
      - Y-axis = species
      - Cell color intensity = number of detections

    Modeled on BatAMP (Bat Acoustic Monitoring Portal), which visualizes
    44 North American species with seasonal trend and location filtering.

    Usage:
        heatmap = ActivityHeatmap()
        heatmap.add_session(session1)
        heatmap.add_session(session2)
        matrix = heatmap.build_matrix()  # [species × dates]
    """

    def __init__(self):
        self._sessions: list[ActivitySession] = []

    def add_session(self, session: ActivitySession) -> None:
        """Add a recording session to the heatmap data."""
        if session.date is not None:
            self._sessions.append(session)

    def build_matrix(self) -> tuple[np.ndarray, list[str], list[date]]:
        """
        Build the heatmap matrix.

        Returns:
            (matrix, species_names, dates) where:
              matrix: 2D int array [n_species × n_dates], detection counts.
              species_names: List of species names (row labels).
              dates: List of dates (column labels).
        """
        if not self._sessions:
            return np.zeros((0, 0)), [], []

        # Collect all species and dates.
        all_species = set()
        all_dates = set()
        for session in self._sessions:
            for sp in session.species_counts:
                all_species.add(sp)
            if session.date:
                all_dates.add(session.date)

        species_list = sorted(all_species)
        date_list = sorted(all_dates)

        # Build matrix.
        n_species = len(species_list)
        n_dates = len(date_list)
        matrix = np.zeros((n_species, n_dates), dtype="int32")

        for session in self._sessions:
            if session.date not in date_list:
                continue
            date_idx = date_list.index(session.date)
            for sp, count in session.species_counts.items():
                if sp in species_list:
                    sp_idx = species_list.index(sp)
                    matrix[sp_idx, date_idx] += count

        return matrix, species_list, date_list

    @staticmethod
    def to_display_string(matrix: np.ndarray, species: list[str],
                          dates: list[date]) -> str:
        """
        Format the heatmap as a text grid for the canvas.

        Args:
            matrix:   Detection matrix [n_species × n_dates].
            species:  Species names (rows).
            dates:    Dates (columns).

        Returns:
            Multi-line text grid.
        """
        if matrix.size == 0:
            return "No heatmap data."

        lines = ["Activity Heatmap:"]
        # Header: date column labels (abbreviated).
        header = "Species"
        for d in dates:
            header += f" {d.strftime('%m/%d'):>5s}"
        lines.append(header)
        lines.append("-" * len(header))

        for i, sp in enumerate(species):
            row = f"{sp[:12]:12s}"
            for j in range(len(dates)):
                count = matrix[i, j]
                if count == 0:
                    row += "    . "
                else:
                    # Color-code by intensity (text representation).
                    if count <= 2:
                        row += "   ▁ "
                    elif count <= 5:
                        row += "   ▃ "
                    elif count <= 10:
                        row += "   ▅ "
                    else:
                        row += "   ▇ "
            lines.append(row)

        return "\n".join(lines)