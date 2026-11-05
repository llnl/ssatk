"""Minor Planet Center (MPC) orbit catalogs: download, parse and convert to states.

The MPC publishes osculating heliocentric elements for every known minor
planet in its fixed-width export format, documented at
https://minorplanetcenter.net/iau/info/MPOrbitFormat.html. Angles are
referred to the ecliptic and mean equinox of J2000.0, epochs are ``.0 TT``
and semimajor axes are in au. This module downloads those catalogs into the
SSATK output root, parses them into a :class:`pandas.DataFrame`, and turns
the elements into heliocentric Cartesian states in metres and metres per
second.

States are two-body (Sun only) conics. They reproduce the MPC solution at its
epoch; away from the epoch planetary perturbations accumulate, so hand the
epoch state to an N-body integrator such as REBOUND for long arcs or close
planetary approaches.
"""

from __future__ import annotations

import gzip
import os
import re
import tempfile
import time as _time
import warnings
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from ..constants import au_to_m, day_to_second
from .datapath import datapath

__all__ = [
    "MPC_GM_SUN",
    "MPC_ORBIT_CATALOGS",
    "fetch_mpc_orbits",
    "mpc_epoch_to_time",
    "mpc_heliocentric_states",
    "parse_mpcorb",
    "select_mpc_object",
    "unpack_mpc_date",
    "unpack_mpc_designation",
]

# MPC mean daily motions follow n = k a**-1.5 with the Gaussian gravitational
# constant k, so k**2 au**3/day**2 is the solar GM consistent with the
# published (a, n) pairs.
_GAUSSIAN_K = 0.01720209895
MPC_GM_SUN = _GAUSSIAN_K**2 * au_to_m**3 / day_to_second**2

MPC_ORBIT_CATALOGS = {
    "nea": "https://minorplanetcenter.net/iau/MPCORB/NEA.txt",
    "pha": "https://minorplanetcenter.net/iau/MPCORB/PHA.txt",
    "mpcorb": "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz",
}

_USER_AGENT = "ssatk-mpc/1.0"
_PACKED_DATE = re.compile(r"^[IJK]\d{2}[1-9A-C][1-9A-V]\d*$")
_ORBIT_TYPES = {
    1: "Atira",
    2: "Aten",
    3: "Apollo",
    4: "Amor",
    5: "q < 1.665 au",
    6: "Hungaria",
    8: "Hilda",
    9: "Jupiter Trojan",
    10: "Distant object",
}
_FLAG_NEO = 2048
_FLAG_KM_NEO = 4096
_FLAG_PHA = 32768


def _base62(char: str) -> int:
    if char.isdigit():
        return int(char)
    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10
    if "a" <= char <= "z":
        return ord(char) - ord("a") + 36
    raise ValueError(f"invalid MPC packed character {char!r}")


def unpack_mpc_date(packed: str) -> tuple[int, int, float]:
    """Unpack an MPC packed date such as ``"K01AM"`` into ``(year, month, day)``.

    Digits after the fifth character are the decimal fraction of the day
    (``"J981I73"`` is 1998 January 18.73).
    """
    packed = packed.strip()
    if not _PACKED_DATE.match(packed):
        raise ValueError(f"not an MPC packed date: {packed!r}")
    year = (_base62(packed[0]) * 100) + int(packed[1:3])
    month = _base62(packed[3])
    day = float(_base62(packed[4]))
    if len(packed) > 5:
        day += float("0." + packed[5:])
    return year, month, day


def mpc_epoch_to_time(packed: str):
    """Return the :class:`astropy.time.Time` (TT) for a packed MPC epoch."""
    from astropy.time import Time

    year, month, day = unpack_mpc_date(packed)
    whole = int(day)
    base = Time(f"{year:04d}-{month:02d}-{whole:02d}T00:00:00", scale="tt")
    return Time(base.jd1, base.jd2 + (day - whole), format="jd", scale="tt")


def _epoch_jd_tt(packed: str) -> float:
    year, month, day = unpack_mpc_date(packed)
    # Fliegel and Van Flandern (1968) Gregorian day number, then shift to the
    # 0h Julian date and add the day fraction.
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    whole = int(day)
    jdn = whole + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return jdn - 0.5 + (day - whole)


def unpack_mpc_designation(packed: str) -> str:
    """Unpack a 5-character permanent or 7-character provisional MPC designation.

    Permanent numbers return the number as a string (``"A0345"`` -> ``"100345"``);
    provisional designations return the readable form (``"J95X01L"`` ->
    ``"1995 XL1"``); survey designations return e.g. ``"2040 P-L"``. Packed
    forms this function does not recognize are returned stripped and unchanged.
    """
    text = packed.strip()
    if len(text) == 5:
        if text.isdigit():
            return str(int(text))
        if text[0] == "~" and all(c.isalnum() for c in text[1:]):
            value = 0
            for char in text[1:]:
                value = value * 62 + _base62(char)
            return str(620000 + value)
        if text[0].isalpha() and text[1:].isdigit():
            return str(_base62(text[0]) * 10000 + int(text[1:]))
    if len(text) == 7:
        survey = {"PLS": "P-L", "T1S": "T-1", "T2S": "T-2", "T3S": "T-3"}
        if text[:3] in survey and text[3:].isdigit():
            return f"{int(text[3:])} {survey[text[:3]]}"
        if text[0] in "IJK" and text[1:3].isdigit() and text[3].isupper() and text[6].isalpha():
            year = _base62(text[0]) * 100 + int(text[1:3])
            cycle = _base62(text[4]) * 10 + int(text[5])
            suffix = str(cycle) if cycle else ""
            return f"{year} {text[3]}{text[6]}{suffix}"
    return text


def _float(field: str) -> float:
    field = field.strip()
    return float(field) if field else np.nan


def _int(field: str):
    field = field.strip()
    return int(field) if field.isdigit() else pd.NA


def _parse_line(line: str) -> dict | None:
    """Parse one MPC export-format record; return ``None`` for non-records."""
    if len(line) < 103 or not _PACKED_DATE.match(line[20:25]):
        return None
    try:
        elements = {
            "H": _float(line[8:13]),
            "G": _float(line[14:19]),
            "mean_anomaly_deg": float(line[26:35]),
            "arg_perihelion_deg": float(line[37:46]),
            "ascending_node_deg": float(line[48:57]),
            "inclination_deg": float(line[59:68]),
            "eccentricity": float(line[70:79]),
            "mean_motion_deg_per_day": float(line[80:91]),
            "semimajor_axis_au": float(line[92:103]),
        }
    except ValueError:
        return None

    packed = line[0:7].strip()
    readable = line[166:194].strip()
    number = pd.NA
    name = ""
    match = re.match(r"^\((\d+)\)\s*(.*)$", readable)
    if match:
        number = int(match.group(1))
        name = match.group(2).strip()
    if not readable:
        readable = unpack_mpc_designation(packed)
    flags_text = line[161:165].strip()
    flags = int(flags_text, 16) if flags_text else 0

    record = {
        "designation": readable,
        "number": number,
        "name": name,
        "packed_designation": packed,
        "epoch_packed": line[20:25],
        "epoch_jd_tt": _epoch_jd_tt(line[20:25]),
        **elements,
        "uncertainty": line[105:106].strip(),
        "reference": line[107:116].strip(),
        "n_observations": _int(line[117:122]),
        "n_oppositions": _int(line[123:126]),
        "arc": line[127:136].strip(),
        "rms_arcsec": _float(line[137:141]),
        "perturbers_coarse": line[142:145].strip(),
        "perturbers_precise": line[146:149].strip(),
        "computer": line[150:160].strip(),
        "flags": flags,
        "orbit_type": _ORBIT_TYPES.get(flags & 0x3F, ""),
        "is_neo": bool(flags & _FLAG_NEO),
        "is_km_neo": bool(flags & _FLAG_KM_NEO),
        "is_pha": bool(flags & _FLAG_PHA),
        "last_observation": line[194:202].strip(),
    }
    record["perihelion_au"] = record["semimajor_axis_au"] * (1.0 - record["eccentricity"])
    return record


def _read_lines(source):
    if isinstance(source, (list, tuple)):
        return list(source)
    if isinstance(source, Path) or (isinstance(source, str) and "\n" not in source and os.path.isfile(source)):
        path = Path(source)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="ascii", errors="replace") as handle:
            return handle.read().splitlines()
    if isinstance(source, str):
        return source.splitlines()
    return list(source)


def parse_mpcorb(source) -> pd.DataFrame:
    """Parse MPC export-format orbits (``MPCORB.DAT``, ``NEA.txt``, ``PHA.txt``).

    Parameters
    ----------
    source : str, pathlib.Path or iterable of str
        A file path (``.gz`` is decompressed), the file contents, or lines.

    Returns
    -------
    pandas.DataFrame
        One row per orbit. Angles are degrees referred to the ecliptic and
        mean equinox of J2000.0, ``semimajor_axis_au`` and ``perihelion_au``
        are au, and ``epoch_jd_tt`` is the TT Julian date. ``orbit_type``,
        ``is_neo``, ``is_km_neo`` and ``is_pha`` decode the MPC hexadecimal
        flags. Non-blank lines that are not orbit records (the ``MPCORB.DAT``
        header, for example) are counted in ``df.attrs["skipped_lines"]``.
    """
    records = []
    skipped = 0
    for line in _read_lines(source):
        record = _parse_line(line.rstrip("\r\n"))
        if record is None:
            skipped += bool(line.strip())
            continue
        records.append(record)
    frame = pd.DataFrame.from_records(records)
    if not frame.empty:
        frame["number"] = frame["number"].astype("Int64")
        frame["n_observations"] = frame["n_observations"].astype("Int64")
        frame["n_oppositions"] = frame["n_oppositions"].astype("Int64")
    frame.attrs["skipped_lines"] = skipped
    return frame


def _download(url: str, target: Path, timeout: float) -> None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(request, timeout=timeout) as response, tempfile.NamedTemporaryFile(
        "wb", delete=False, dir=str(target.parent), suffix=".part"
    ) as handle:
        tmp_path = Path(handle.name)
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    if tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        raise OSError(f"empty response from {url}")
    os.replace(tmp_path, target)


def fetch_mpc_orbits(
    catalog: str = "nea",
    *,
    max_age_hours: float = 24.0,
    refresh: bool = False,
    timeout: float = 300.0,
    path=None,
) -> pd.DataFrame:
    """Download (or reuse a cached copy of) an MPC orbit catalog and parse it.

    Parameters
    ----------
    catalog : {"nea", "pha", "mpcorb"}
        Near-Earth asteroids (about 9 MB), potentially hazardous asteroids
        (about 0.5 MB) or every minor planet (``MPCORB.DAT.gz``, about 94 MB
        compressed). The URLs are in :data:`MPC_ORBIT_CATALOGS`.
    max_age_hours : float
        Reuse the cached file when it is younger than this. The MPC
        regenerates these files daily.
    refresh : bool
        Download even when the cache is fresh.
    timeout : float
        Network timeout in seconds.
    path : str or pathlib.Path, optional
        Cache file location. Defaults to ``mpc/<file name>`` under the SSATK
        output root (``~/ssatk_output`` or ``$SSATK_OUTPUT_DIR``).

    Returns
    -------
    pandas.DataFrame
        See :func:`parse_mpcorb`. ``attrs`` also records ``source_url``,
        ``cache_path`` and ``retrieved_utc`` (the cache file's modification
        time).

    Notes
    -----
    A failed download falls back to an existing stale cache with a warning;
    without one the network error is raised.
    """
    key = catalog.lower()
    if key not in MPC_ORBIT_CATALOGS:
        raise ValueError(f"catalog must be one of {sorted(MPC_ORBIT_CATALOGS)}, got {catalog!r}")
    url = MPC_ORBIT_CATALOGS[key]
    target = Path(path).expanduser() if path is not None else Path(datapath(f"mpc/{url.rsplit('/', 1)[-1]}"))

    fresh = target.exists() and (_time.time() - target.stat().st_mtime) < max_age_hours * 3600.0
    if refresh or not fresh:
        try:
            _download(url, target, timeout)
        except OSError as exc:
            if not target.exists():
                raise
            warnings.warn(
                f"MPC download from {url} failed ({exc}); using cached {target}",
                RuntimeWarning,
                stacklevel=2,
            )

    frame = parse_mpcorb(target)
    frame.attrs["source_url"] = url
    frame.attrs["cache_path"] = str(target)
    frame.attrs["retrieved_utc"] = pd.Timestamp(target.stat().st_mtime, unit="s", tz="UTC").isoformat()
    return frame


def select_mpc_object(orbits: pd.DataFrame, query) -> pd.Series:
    """Return the single catalog row matching a number, name or designation.

    ``query`` may be a minor-planet number (``433`` or ``"433"``), a name
    (``"Eros"``, case-insensitive), the readable designation
    (``"(433) Eros"`` or ``"2024 YR4"``) or the packed designation.
    """
    text = str(query).strip()
    mask = np.zeros(len(orbits), dtype=bool)
    if text.isdigit():
        mask |= (orbits["number"] == int(text)).fillna(False).to_numpy(dtype=bool)
    lowered = text.lower()
    for column in ("name", "designation", "packed_designation"):
        mask |= orbits[column].str.lower().eq(lowered).to_numpy()
    matches = orbits[mask]
    if matches.empty:
        raise KeyError(f"no MPC orbit matches {query!r}")
    if len(matches) > 1:
        raise ValueError(f"{len(matches)} MPC orbits match {query!r}: {list(matches['designation'])}")
    return matches.iloc[0]


def _column(orbits, name):
    if isinstance(orbits, pd.Series):
        return np.atleast_1d(np.asarray(orbits[name], dtype=float))
    if isinstance(orbits, pd.DataFrame):
        return orbits[name].to_numpy(dtype=float)
    return np.atleast_1d(np.asarray(orbits[name], dtype=float))


def mpc_heliocentric_states(orbits, time=None, *, frame: str = "ecliptic", mu: float = MPC_GM_SUN):
    """Heliocentric two-body states from MPC elements.

    Parameters
    ----------
    orbits : pandas.DataFrame, pandas.Series or mapping
        Output of :func:`parse_mpcorb` (or one row of it), or any mapping with
        ``semimajor_axis_au``, ``eccentricity``, ``inclination_deg``,
        ``ascending_node_deg``, ``arg_perihelion_deg``, ``mean_anomaly_deg``
        and ``epoch_jd_tt``.
    time : astropy.time.Time, optional
        Scalar evaluation time. ``None`` evaluates each orbit at its own epoch.
        Otherwise the mean anomaly is advanced by the two-body mean motion
        ``sqrt(mu / a**3)``; planetary perturbations are not modeled.
    frame : {"ecliptic", "icrf"}
        ``"ecliptic"`` returns ecliptic and mean equinox of J2000.0 axes (the
        MPC frame); ``"icrf"`` rotates by the J2000 obliquity to equatorial
        ICRF-aligned axes, the axes used by GCRF.
    mu : float
        Solar gravitational parameter in m^3/s^2. The default matches the
        Gaussian constant that MPC mean motions use.

    Returns
    -------
    r, v : numpy.ndarray
        Heliocentric position (m) and velocity (m/s), shape ``(N, 3)``, or
        ``(3,)`` for a single orbit given as a Series or scalar mapping. Rows
        with non-elliptic or missing elements are NaN.
    """
    # Imported here because ssatk.plots imports ssatk.io while it initializes,
    # and orbital_mechanics imports ssatk.plots; a module-level import would
    # close that cycle.
    from ..coordinates.equatorial_ecliptic import ecliptic_xyz_to_equatorial_xyz
    from ..orbital_mechanics.keplerian import kepler_to_state, true_anomaly

    frame = frame.lower()
    if frame not in ("ecliptic", "icrf"):
        raise ValueError(f"frame must be 'ecliptic' or 'icrf', got {frame!r}")
    single = isinstance(orbits, pd.Series) or (
        not isinstance(orbits, pd.DataFrame) and np.ndim(orbits["semimajor_axis_au"]) == 0
    )

    a = _column(orbits, "semimajor_axis_au") * au_to_m
    e = _column(orbits, "eccentricity")
    inc = np.radians(_column(orbits, "inclination_deg"))
    node = np.radians(_column(orbits, "ascending_node_deg"))
    peri = np.radians(_column(orbits, "arg_perihelion_deg"))
    mean = np.radians(_column(orbits, "mean_anomaly_deg"))
    epoch = _column(orbits, "epoch_jd_tt")

    valid = np.isfinite(a) & np.isfinite(e) & (a > 0) & (e >= 0) & (e < 1)
    valid &= np.isfinite(inc) & np.isfinite(node) & np.isfinite(peri) & np.isfinite(mean) & np.isfinite(epoch)

    r = np.full((a.size, 3), np.nan)
    v = np.full((a.size, 3), np.nan)
    if np.any(valid):
        if time is not None:
            tt = time.tt
            if not tt.isscalar:
                raise ValueError("time must be a scalar astropy Time")
            # Subtract in two parts so a ~2.46e6 day Julian date keeps
            # sub-millisecond resolution.
            dt_days = (tt.jd1 - epoch[valid]) + tt.jd2
            n = np.sqrt(mu / a[valid] ** 3)
            mean_v = mean[valid] + n * dt_days * day_to_second
        else:
            mean_v = mean[valid]
        nu = true_anomaly(eccentricity=e[valid], mean_anomaly=mean_v)
        r_v, v_v = kepler_to_state(
            a=a[valid], e=e[valid], i=inc[valid], pa=peri[valid], raan=node[valid], nu=nu, mu=mu
        )
        r[valid] = np.reshape(r_v, (-1, 3))
        v[valid] = np.reshape(v_v, (-1, 3))
    if frame == "icrf":
        r = np.column_stack(ecliptic_xyz_to_equatorial_xyz(r[:, 0], r[:, 1], r[:, 2]))
        v = np.column_stack(ecliptic_xyz_to_equatorial_xyz(v[:, 0], v[:, 1], v[:, 2]))
    if single:
        return r[0], v[0]
    return r, v
