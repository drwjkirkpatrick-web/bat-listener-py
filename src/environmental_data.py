"""
Environmental Data Integration (Improvement #25).

Logs and analyzes environmental covariates that affect bat activity:
temperature, humidity, barometric pressure, moon phase, and time-since-sunset.

Research (Gorman et al. 2021, PMC10967300) found:
  - Mean hourly temperature is the top predictor of bat activity
  - Time since sunset is the second predictor (activity peaks ~2 hrs post-sunset)
  - Day-of-year captures seasonal patterns
  - Moon phase effects vary by species and habitat (lunar phobia vs. lunar philia)
  - Migratory species show positive temperature response
  - Cave-hibernating species show negative temperature response in active seasons

This module:
  - Records environmental data alongside acoustic sessions
  - Computes moon phase from date
  - Calculates time-since-sunset from location and date
  - Provides a simple GLM-style activity model correlating covariates with activity
  - Overlays environmental data on activity charts

Sources:
  - Gorman et al. 2021, "Bat activity responses to weather variables"
  - Astronomy algorithms for moon phase (Meeus, Astronomical Algorithms)
"""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


@dataclass
class EnvironmentalData:
    """
    Environmental conditions at the time of recording.

    All data is optional — fill what's available. Missing values default to 0.0
    and are flagged as "not recorded."
    """
    temperature_c: float = 0.0       # Air temperature (°C)
    humidity_pct: float = 0.0          # Relative humidity (%)
    pressure_hpa: float = 0.0          # Barometric pressure (hPa)
    wind_speed_ms: float = 0.0        # Wind speed (m/s)
    moon_phase: float = 0.0           # 0=new moon, 0.5=full moon, 1.0=next new moon
    time_since_sunset_min: float = 0.0  # Minutes after sunset
    day_of_year: int = 0              # 1–366 (seasonal indicator)

    # Metadata
    recorded: bool = False  # Whether environmental data was actually recorded
    notes: str = ""

    def to_display_string(self) -> str:
        """Format as readable text for the UI."""
        if not self.recorded:
            return "Environmental data: not recorded"
        lines = ["Environmental Conditions:"]
        if self.temperature_c != 0:
            lines.append(f"  Temp: {self.temperature_c:.1f}°C")
        if self.humidity_pct != 0:
            lines.append(f"  Humidity: {self.humidity_pct:.0f}%")
        if self.pressure_hpa != 0:
            lines.append(f"  Pressure: {self.pressure_hpa:.1f} hPa")
        if self.wind_speed_ms != 0:
            lines.append(f"  Wind: {self.wind_speed_ms:.1f} m/s")
        if self.moon_phase != 0:
            phase_name = self._moon_phase_name(self.moon_phase)
            lines.append(f"  Moon: {phase_name} ({self.moon_phase:.2f})")
        if self.time_since_sunset_min != 0:
            hours = self.time_since_sunset_min / 60.0
            lines.append(f"  Time since sunset: {hours:.1f} hrs")
        if self.day_of_year != 0:
            lines.append(f"  Day of year: {self.day_of_year}")
        return "\n".join(lines)

    @staticmethod
    def _moon_phase_name(phase: float) -> str:
        """Convert phase value (0–1) to name."""
        if phase < 0.03 or phase > 0.97:
            return "New Moon"
        elif phase < 0.22:
            return "Waxing Crescent"
        elif phase < 0.28:
            return "First Quarter"
        elif phase < 0.47:
            return "Waxing Gibbous"
        elif phase < 0.53:
            return "Full Moon"
        elif phase < 0.72:
            return "Waning Gibbous"
        elif phase < 0.78:
            return "Last Quarter"
        else:
            return "Waning Crescent"


class EnvironmentalCalculator:
    """
    Computes environmental covariates and correlates them with bat activity.
    """

    @staticmethod
    def compute_moon_phase(date: datetime) -> float:
        """
        Compute the moon phase (0–1) for a given date.

        Uses a simplified astronomical algorithm based on the synodic month
        (29.53059 days). The reference new moon is 2000-01-06 18:14 UTC.

        0.0 = new moon, 0.5 = full moon, 1.0 = next new moon.

        Reference: Meeus, "Astronomical Algorithms", 2nd ed.

        Args:
            date: Date and time to compute the moon phase for.

        Returns:
            Moon phase as a float 0–1.
        """
        # Reference new moon: January 6, 2000, 18:14 UTC.
        ref_new_moon = datetime(2000, 1, 6, 18, 14, 0)
        synodic_month = 29.53059  # days

        # Days since reference new moon.
        days_since = (date - ref_new_moon).total_seconds() / 86400.0

        # Phase = fractional position in the synodic cycle.
        phase = (days_since % synodic_month) / synodic_month

        return phase

    @staticmethod
    def compute_time_since_sunset(date: datetime,
                                   latitude: float = 45.0,
                                   longitude: float = -120.0,
                                   timezone_offset_hours: float = -8.0) -> float:
        """
        Approximate time since sunset in minutes.

        Uses a simple solar position algorithm to estimate sunset time for the
        given date and location, then computes the difference from the given
        datetime.

        This is an approximation — for precise values, use a dedicated
        astronomy library like PyEphem or skyfield.

        Args:
            date:                  The recording datetime.
            latitude:              Site latitude (degrees, +N).
            longitude:             Site longitude (degrees, +E).
            timezone_offset_hours: UTC offset (hours, e.g. -8 for PST).

        Returns:
            Minutes since sunset (negative = before sunset).
        """
        # Compute day of year.
        doy = date.timetuple().tm_yday

        # Solar declination (approximation).
        decl = 23.45 * math.sin(math.radians(360 * (284 + doy) / 365))

        # Hour angle at sunset: cos(h) = -tan(lat) * tan(decl)
        lat_rad = math.radians(latitude)
        decl_rad = math.radians(decl)
        cos_h = -math.tan(lat_rad) * math.tan(decl_rad)

        # Clamp for polar conditions.
        cos_h = max(-1.0, min(1.0, cos_h))

        # Sunset hour angle (degrees).
        h_sunset = math.degrees(math.acos(cos_h))

        # Sunset time (UTC): 12 - h/15 + longitude/15
        sunset_utc_hours = 12.0 - h_sunset / 15.0 + longitude / 15.0
        sunset_local_hours = sunset_utc_hours + timezone_offset_hours

        # Current local time in hours (decimal).
        current_local_hours = date.hour + date.minute / 60.0 + date.second / 3600.0

        # Time since sunset in minutes.
        time_diff_hours = current_local_hours - sunset_local_hours
        return time_diff_hours * 60.0

    @staticmethod
    def build_activity_correlation(temperatures: list[float],
                                    call_rates: list[float],
                                    moon_phases: list[float] | None = None,
                                    times_since_sunset: list[float] | None = None) -> dict:
        """
        Build a simple correlation model between environmental variables and bat activity.

        Computes Pearson correlation coefficients between temperature, moon phase,
        time-since-sunset, and call rate. This isn't a full GLM, but it gives
        researchers a quick view of which environmental variables matter most.

        Args:
            temperatures:        List of temperatures (°C) per session.
            call_rates:           List of call rates (calls/min) per session.
            moon_phases:          Optional list of moon phases (0–1).
            times_since_sunset:   Optional list of times since sunset (min).

        Returns:
            Dict with correlation coefficients and significance indicators.
        """
        result = {}

        if len(temperatures) < 3 or len(temperatures) != len(call_rates):
            return {"error": "Need ≥3 paired observations"}

        # Temperature vs. call rate.
        temp_corr = EnvironmentalCalculator._pearson(temperatures, call_rates)
        result["temperature_correlation"] = temp_corr

        if moon_phases and len(moon_phases) == len(call_rates):
            moon_corr = EnvironmentalCalculator._pearson(moon_phases, call_rates)
            result["moon_phase_correlation"] = moon_corr

        if times_since_sunset and len(times_since_sunset) == len(call_rates):
            sunset_corr = EnvironmentalCalculator._pearson(times_since_sunset, call_rates)
            result["time_since_sunset_correlation"] = sunset_corr

        # Interpretation.
        result["interpretation"] = EnvironmentalCalculator._interpret_correlations(result)

        return result

    @staticmethod
    def _pearson(x: list[float], y: list[float]) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        x_arr = np.array(x, dtype="float64")
        y_arr = np.array(y, dtype="float64")
        if np.std(x_arr) == 0 or np.std(y_arr) == 0:
            return 0.0
        return float(np.corrcoef(x_arr, y_arr)[0, 1])

    @staticmethod
    def _interpret_correlations(corrs: dict) -> str:
        """Generate a human-readable interpretation of correlation results."""
        lines = []

        temp_r = corrs.get("temperature_correlation", 0)
        if abs(temp_r) > 0.1:
            direction = "positive" if temp_r > 0 else "negative"
            strength = "weak" if abs(temp_r) < 0.3 else "moderate" if abs(temp_r) < 0.6 else "strong"
            lines.append(f"Temperature: {strength} {direction} correlation (r={temp_r:.2f})")

        moon_r = corrs.get("moon_phase_correlation", 0)
        if abs(moon_r) > 0.1:
            direction = "positive" if moon_r > 0 else "negative"
            strength = "weak" if abs(moon_r) < 0.3 else "moderate" if abs(moon_r) < 0.6 else "strong"
            lines.append(f"Moon phase: {strength} {direction} correlation (r={moon_r:.2f})")

        sunset_r = corrs.get("time_since_sunset_correlation", 0)
        if abs(sunset_r) > 0.1:
            direction = "positive" if sunset_r > 0 else "negative"
            strength = "weak" if abs(sunset_r) < 0.3 else "moderate" if abs(sunset_r) < 0.6 else "strong"
            lines.append(f"Time since sunset: {strength} {direction} correlation (r={sunset_r:.2f})")

        if not lines:
            lines.append("No significant correlations found (need more data)")

        return "; ".join(lines)


def create_environmental_snapshot(latitude: float = 0.0, longitude: float = 0.0,
                                  temp_c: float = 0.0, humidity: float = 0.0,
                                  pressure: float = 0.0, wind: float = 0.0,
                                  tz_offset: float = -8.0) -> EnvironmentalData:
    """
    Create an EnvironmentalData snapshot with computed moon phase and sunset time.

    Args:
        latitude:  Site latitude.
        longitude: Site longitude.
        temp_c:    Temperature (°C).
        humidity:  Relative humidity (%).
        pressure:  Barometric pressure (hPa).
        wind:      Wind speed (m/s).
        tz_offset: Timezone offset from UTC (hours).

    Returns:
        Populated EnvironmentalData with computed fields.
    """
    now = datetime.now()
    moon_phase = EnvironmentalCalculator.compute_moon_phase(now)
    time_since_sunset = EnvironmentalCalculator.compute_time_since_sunset(
        now, latitude, longitude, tz_offset
    ) if latitude != 0 and longitude != 0 else 0.0

    return EnvironmentalData(
        temperature_c=temp_c,
        humidity_pct=humidity,
        pressure_hpa=pressure,
        wind_speed_ms=wind,
        moon_phase=moon_phase,
        time_since_sunset_min=time_since_sunset,
        day_of_year=now.timetuple().tm_yday,
        recorded=True,
    )