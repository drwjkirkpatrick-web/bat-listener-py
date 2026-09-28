"""
Survey Protocols & Activity Metrics (Improvements #22, #24).

Implements structured survey modes matching EUROBATS Publication Series No. 5
and BCT Good Practice Guidelines (4th edition), plus duty-cycle-aware
activity metrics for valid inter-site comparisons.

  #22 — Survey Protocol Modes:
    Four standard survey modes with pre-filled metadata and timing rules:
      - Walked transect (fixed path, constant speed, sunset +1–3 hrs)
      - Point count (fixed listening points, 4-min intervals, 100m apart)
      - Vehicle transect (≤25 km/h, annual 4-week window)
      - Static/automated deployment (200m grid spacing, 3+ nights)

    Each mode enforces correct metadata fields (transect ID, point number,
    sunset time, weather conditions) and produces data comparable to
    pan-European/NABat trend analyses.

  #24 — Duty-Cycle-Aware Activity Metrics:
    When recordings use duty-cycled (intermittent) sampling, raw counts are
    biased. This module computes validated correction metrics:
      - Call Rate (CR): calls per unit time
      - Activity Index (AI): proportion of time intervals with ≥1 call
      - Bout-Time Percentage (BTP): proportion of time in active bouts
    With confidence flags for duty-cycled data.

Sources:
  - EUROBATS Publication Series No. 5 (3rd ed.)
  - BCT Good Practice Guidelines (4th ed., 2024)
  - biorxiv 2025.04.15.649046 (duty-cycle correction metrics)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import datetime


class SurveyMode(Enum):
    """
    Standardized survey modes from EUROBATS and BCT guidelines.
    """
    WALKED_TRANSECT = "walked_transect"
    POINT_COUNT = "point_count"
    VEHICLE_TRANSECT = "vehicle_transect"
    STATIC_DEPLOYMENT = "static_deployment"
    AD_HOC = "ad_hoc"  # Unstructured recording (no protocol)


@dataclass
class SurveyConfig:
    """
    Configuration for a bat acoustic survey session.

    Fields vary by mode — only relevant fields need to be filled.
    """
    mode: SurveyMode = SurveyMode.AD_HOC

    # --- Common fields ---
    surveyor_name: str = ""
    survey_id: str = ""
    sunset_time: Optional[datetime.datetime] = None
    weather_notes: str = ""
    temperature_c: float = 0.0
    humidity_pct: float = 0.0

    # --- Transect fields (walked and vehicle) ---
    transect_id: str = ""
    transect_speed_kmh: float = 0.0  # For vehicle transect: ≤25 km/h

    # --- Point count fields ---
    point_id: str = ""
    point_interval_min: float = 4.0  # Standard: 4-minute intervals
    point_spacing_m: float = 100.0   # Standard: 100m apart

    # --- Static deployment fields ---
    grid_spacing_m: float = 200.0   # Standard: 200m grid spacing
    deployment_nights: int = 3       # Standard: minimum 3 nights

    # --- Vehicle transect annual window ---
    survey_window_start: Optional[datetime.datetime] = None
    survey_window_end: Optional[datetime.datetime] = None

    def validate(self) -> list[str]:
        """
        Validate the survey configuration against protocol requirements.

        Returns:
            List of warning messages for protocol violations (empty = compliant).
        """
        warnings = []

        if self.mode == SurveyMode.WALKED_TRANSECT:
            if not self.transect_id:
                warnings.append("Transect ID required for walked transect mode")
            if self.sunset_time is None:
                warnings.append("Sunset time required — walked transects start sunset +1–3 hrs")

        elif self.mode == SurveyMode.VEHICLE_TRANSECT:
            if self.transect_speed_kmh > 25.0:
                warnings.append(f"Vehicle transect speed {self.transect_speed_kmh} km/h exceeds 25 km/h limit")
            if not self.survey_window_start or not self.survey_window_end:
                warnings.append("Annual survey window (4-week) not set for vehicle transect")

        elif self.mode == SurveyMode.POINT_COUNT:
            if not self.point_id:
                warnings.append("Point ID required for point count mode")
            if self.point_interval_min != 4.0:
                warnings.append(f"Point interval {self.point_interval_min} min — standard is 4 min")

        elif self.mode == SurveyMode.STATIC_DEPLOYMENT:
            if self.grid_spacing_m < 200.0:
                warnings.append(f"Grid spacing {self.grid_spacing_m}m — BCT recommends ≥200m")
            if self.deployment_nights < 3:
                warnings.append(f"Only {self.deployment_nights} nights — BCT recommends ≥3 for >90% detection")

        return warnings

    def to_metadata_fields(self) -> dict:
        """
        Convert survey config to GUANO-compatible metadata fields.

        Returns:
            Dict of key-value pairs for GUANO metadata.
        """
        fields = {
            "Survey Mode": self.mode.value,
        }
        if self.surveyor_name:
            fields["Surveyor"] = self.surveyor_name
        if self.survey_id:
            fields["Survey ID"] = self.survey_id
        if self.transect_id:
            fields["Transect ID"] = self.transect_id
        if self.point_id:
            fields["Point ID"] = self.point_id
        if self.sunset_time:
            fields["Sunset Time"] = self.sunset_time.isoformat()
        if self.temperature_c:
            fields["Temperature"] = f"{self.temperature_c:.1f}"
        if self.humidity_pct:
            fields["Humidity"] = f"{self.humidity_pct:.1f}"
        if self.weather_notes:
            fields["Weather"] = self.weather_notes
        return fields


# ---------------------------------------------------------------------------
# Duty-Cycle-Aware Activity Metrics (Improvement #24)
# ---------------------------------------------------------------------------

@dataclass
class DutyCycleConfig:
    """
    Configuration for duty-cycled (intermittent) recording.

    Duty cycling records for a fixed period, then pauses, repeating.
    This saves battery and storage but biases activity metrics.

    Attributes:
        record_duration_s:  How long the recorder is ON per cycle (seconds).
        cycle_duration_s:   Total cycle length (ON + OFF) in seconds.
        listening_ratio:    record_duration_s / cycle_duration_s (auto-computed).
    """
    record_duration_s: float = 60.0   # 1 minute ON
    cycle_duration_s: float = 300.0   # 5 minute cycle (1 min on, 4 min off)
    listening_ratio: float = field(init=False)

    def __post_init__(self):
        if self.cycle_duration_s > 0:
            self.listening_ratio = self.record_duration_s / self.cycle_duration_s
        else:
            self.listening_ratio = 1.0

    @property
    def is_duty_cycled(self) -> bool:
        """True if the recording is duty-cycled (listening ratio < 1.0)."""
        return self.listening_ratio < 1.0


@dataclass
class ActivityMetrics:
    """
    Computed bat activity metrics from a recording session.

    All metrics include confidence flags when duty-cycling is detected.
    """
    # Raw counts.
    total_calls: int = 0
    total_passes: int = 0         # A "pass" = ≥1 call within 1 second
    recording_duration_s: float = 0.0

    # --- Corrected metrics ---
    call_rate_per_min: float = 0.0     # Calls per minute of listening time
    activity_index: float = 0.0        # Proportion of 1-min intervals with ≥1 call
    bout_time_pct: float = 0.0        # Proportion of time in active bouts (≥2 calls in 5s)

    # --- Duty cycle info ---
    duty_cycled: bool = False
    listening_ratio: float = 1.0
    confidence: str = "high"          # "high", "medium", "low"

    # --- Per-species breakdown (species name → count) ---
    species_counts: dict = field(default_factory=dict)

    def to_display_string(self) -> str:
        """Format as readable text for the UI."""
        lines = [
            f"Calls: {self.total_calls}",
            f"Passes: {self.total_passes}",
            f"Duration: {self.recording_duration_s / 60:.1f} min",
            f"Call Rate: {self.call_rate_per_min:.1f}/min",
            f"Activity Index: {self.activity_index:.2f}",
            f"Bout-Time %: {self.bout_time_pct:.1%}",
        ]
        if self.duty_cycled:
            lines.append(f"Duty-cycled (listening ratio: {self.listening_ratio:.2f})")
            lines.append(f"Confidence: {self.confidence}")
        if self.species_counts:
            lines.append("Species breakdown:")
            for sp, count in sorted(self.species_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {sp}: {count}")
        return "\n".join(lines)


class ActivityCalculator:
    """
    Computes bat activity metrics from detected call timestamps.

    Handles duty-cycled recordings by applying correction factors and
    flagging confidence levels.
    """

    def __init__(self, duty_cycle: DutyCycleConfig | None = None):
        """
        Args:
            duty_cycle: Duty cycle configuration (None = continuous recording).
        """
        self._duty_cycle = duty_cycle or DutyCycleConfig(
            record_duration_s=1.0, cycle_duration_s=1.0
        )

    def calculate(self, call_timestamps_s: list[float],
                  recording_duration_s: float,
                  species_labels: list[str] | None = None) -> ActivityMetrics:
        """
        Calculate activity metrics from call timestamps.

        Args:
            call_timestamps_s:  List of call onset times in seconds from recording start.
            recording_duration_s: Total recording duration (seconds).
            species_labels:     Optional list of species names (parallel to timestamps).

        Returns:
            ActivityMetrics with all computed values.
        """
        metrics = ActivityMetrics(recording_duration_s=recording_duration_s)
        metrics.total_calls = len(call_timestamps_s)
        metrics.duty_cycled = self._duty_cycle.is_duty_cycled
        metrics.listening_ratio = self._duty_cycle.listening_ratio

        # Compute listening time (actual time the recorder was ON).
        listening_time_s = recording_duration_s * self._duty_cycle.listening_ratio
        listening_time_min = listening_time_s / 60.0 if listening_time_s > 0 else 1.0

        # --- Call Rate (CR) ---
        metrics.call_rate_per_min = metrics.total_calls / listening_time_min

        # --- Passes (≥1 call within 1 second = 1 pass) ---
        if call_timestamps_s:
            sorted_ts = sorted(call_timestamps_s)
            passes = 1
            for i in range(1, len(sorted_ts)):
                if sorted_ts[i] - sorted_ts[i - 1] > 1.0:
                    passes += 1
            metrics.total_passes = passes

        # --- Activity Index (AI) ---
        # Proportion of 1-minute intervals with ≥1 call.
        n_intervals = max(1, int(listening_time_min))
        intervals_with_calls = set()
        for ts in call_timestamps_s:
            interval = int(ts / 60.0)
            intervals_with_calls.add(interval)
        metrics.activity_index = len(intervals_with_calls) / n_intervals

        # --- Bout-Time Percentage (BTP) ---
        # Proportion of time in "active bouts" (≥2 calls within 5 seconds).
        if call_timestamps_s:
            sorted_ts = sorted(call_timestamps_s)
            bout_time_s = 0.0
            i = 0
            while i < len(sorted_ts):
                # Start of a potential bout.
                bout_start = sorted_ts[i]
                j = i + 1
                while j < len(sorted_ts) and sorted_ts[j] - sorted_ts[j - 1] <= 5.0:
                    j += 1
                if j - i >= 2:
                    bout_time_s += sorted_ts[j - 1] - bout_start
                i = j
            metrics.bout_time_pct = bout_time_s / listening_time_s if listening_time_s > 0 else 0.0

        # --- Species breakdown ---
        if species_labels:
            for label in species_labels:
                metrics.species_counts[label] = metrics.species_counts.get(label, 0) + 1

        # --- Confidence ---
        if metrics.duty_cycled:
            if self._duty_cycle.listening_ratio < 0.3:
                metrics.confidence = "low"
            elif self._duty_cycle.listening_ratio < 0.5:
                metrics.confidence = "medium"
            else:
                metrics.confidence = "high"

        return metrics