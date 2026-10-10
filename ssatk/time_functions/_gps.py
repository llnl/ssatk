"""Shared GPS-second conversion helpers."""

try:
    from astropy.time import Time as _Time
except ImportError:  # pragma: no cover - astropy is a core dependency in normal installs
    _Time = None


def _to_gps_seconds(t):
    """Accept GPS seconds or ``astropy.time.Time`` and return GPS seconds."""
    if _Time is not None and isinstance(t, _Time):
        return float(t.gps)
    return float(t)
