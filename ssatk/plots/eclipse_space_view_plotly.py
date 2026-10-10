"""
eclipse_space_view_plotly.py — Consolidated eclipse module (search + matplotlib panel + interactive 3D space view)
======================================================================================================================
This used to be two files (eclipse_demo.py for the search + 2D matplotlib
panel, this file for the interactive 3D Plotly "view from space"). Merged
into one so there's a single "main" eclipse module — same functions, same
behavior, just one import instead of two. eclipse_demo.py's content lives
in the first half of this file (search, moon_color/brightness helpers, the
matplotlib figure); the original Plotly space-view code follows it.

Usage
-----
    from eclipse_space_view_plotly import find_and_plot_eclipse, plot_space_view_plotly, plot_space_view_animated

    fig, stats = find_and_plot_eclipse(mode="lunar", event="2014-04-15", save_path="lunar.png")
    plot_space_view_plotly(mode="lunar", event="2014-04-15", save_path="lunar_space.html")
    plot_space_view_animated(mode="solar", save_path="solar_space_animated.html")
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ssatk.constants import EARTH_MEAN_RADIUS_KM, EARTH_MU_KM3_S2, MOON_MEAN_RADIUS_KM

try:
    from .globe_orbit_daynight_plotly import _earth_mesh, _sun_sphere_traces, _earth_atmosphere_trace, RE_KM
    from .moon_render import moon_mesh_plotly
    from .eclipse_brightness_plot import propagate_eci, sun_direction_eci, illumination_fraction, R_SUN_KM, AU_KM
    from .eclipse_appearance_strip import render_lunar_panel, render_solar_panel
    from .scene_primitives import earth_rotation_deg_from_time
except ImportError:
    from globe_orbit_daynight_plotly import _earth_mesh, _earth_atmosphere_trace, RE_KM
    from moon_render import moon_mesh_plotly
    from eclipse_brightness_plot import propagate_eci, sun_direction_eci, illumination_fraction, R_SUN_KM, AU_KM
    from eclipse_appearance_strip import render_lunar_panel, render_solar_panel
    from scene_primitives import earth_rotation_deg_from_time

D_MOON_A_KM = 384_748.0
D_MOON_E = 0.0549
D_MOON_INC_DEG = 5.145
R_MOON_KM = MOON_MEAN_RADIUS_KM


REAL_LUNAR_ECLIPSE_2014 = {
    "key": "2014-04-15-total-lunar-wisconsin",
    "label": "April 15, 2014 total lunar eclipse",
    "peak_utc": "2014-04-15T07:46:48",
    "observer_name": "Madison, Wisconsin",
    "observer_lat_deg": 43.0731,
    "observer_lon_deg": -89.4012,
    "observer_height_m": 267.0,
    "source": "NASA eclipse catalog greatest-eclipse time; Astropy built-in ephemeris for demo geometry",
}

_LUNAR_EVENT_ALIASES = {
    "2014",
    "2014-lunar",
    "2014-total-lunar",
    "2014-april",
    "2014-april-lunar",
    "2014-04-15",
    "2014-04-15-lunar",
    "2014-04-15-total-lunar",
    "2014-04-15-total-lunar-wisconsin",
    "april-2014-lunar",
    "wisconsin-2014-lunar",
}


_SOLAR_ECLIPSE_CATALOG_21ST_CENTURY = """2001-06-21 12:04:46 T
2001-12-14 20:53:01 A
2002-06-10 23:45:22 A
2002-12-04 07:32:16 T
2003-05-31 04:09:22 An
2003-11-23 22:50:22 T
2004-04-19 13:35:05 P
2004-10-14 03:00:23 P
2005-04-08 20:36:51 H
2005-10-03 10:32:47 A
2006-03-29 10:12:23 T
2006-09-22 11:41:16 A
2007-03-19 02:32:57 P
2007-09-11 12:32:24 P
2008-02-07 03:56:10 A
2008-08-01 10:22:12 T
2009-01-26 07:59:45 A
2009-07-22 02:36:25 T
2010-01-15 07:07:39 A
2010-07-11 19:34:38 T
2011-01-04 08:51:42 P
2011-06-01 21:17:18 P
2011-07-01 08:39:30 Pb
2011-11-25 06:21:24 P
2012-05-20 23:53:54 A
2012-11-13 22:12:55 T
2013-05-10 00:26:20 A
2013-11-03 12:47:36 H3
2014-04-29 06:04:33 A-
2014-10-23 21:45:39 P
2015-03-20 09:46:47 T
2015-09-13 06:55:19 P
2016-03-09 01:58:19 T
2016-09-01 09:08:02 A
2017-02-26 14:54:33 A
2017-08-21 18:26:40 T
2018-02-15 20:52:33 P
2018-07-13 03:02:16 P
2018-08-11 09:47:28 P
2019-01-06 01:42:38 P
2019-07-02 19:24:07 T
2019-12-26 05:18:53 A
2020-06-21 06:41:15 Am
2020-12-14 16:14:39 T
2021-06-10 10:43:07 A
2021-12-04 07:34:38 T
2022-04-30 20:42:36 P
2022-10-25 11:01:20 P
2023-04-20 04:17:56 H
2023-10-14 18:00:41 A
2024-04-08 18:18:29 T
2024-10-02 18:46:13 A
2025-03-29 10:48:36 P
2025-09-21 19:43:04 P
2026-02-17 12:13:06 A
2026-08-12 17:47:06 T
2027-02-06 16:00:48 A
2027-08-02 10:07:50 T
2028-01-26 15:08:59 A
2028-07-22 02:56:40 T
2029-01-14 17:13:48 P
2029-06-12 04:06:13 P
2029-07-11 15:37:19 P
2029-12-05 15:03:58 P
2030-06-01 06:29:13 A
2030-11-25 06:51:37 T
2031-05-21 07:16:04 A
2031-11-14 21:07:31 H
2032-05-09 13:26:42 A
2032-11-03 05:34:13 P
2033-03-30 18:02:36 T
2033-09-23 13:54:31 P
2034-03-20 10:18:45 T
2034-09-12 16:19:28 A
2035-03-09 23:05:54 A
2035-09-02 01:56:46 T
2036-02-27 04:46:49 P
2036-07-23 10:32:06 P
2036-08-21 17:25:45 P
2037-01-16 09:48:55 P
2037-07-13 02:40:36 T
2038-01-05 13:47:11 A
2038-07-02 13:32:55 A
2038-12-26 01:00:10 T
2039-06-21 17:12:54 A
2039-12-15 16:23:46 T
2040-05-11 03:43:02 P
2040-11-04 19:09:02 P
2041-04-30 11:52:21 T
2041-10-25 01:36:22 A
2042-04-20 02:17:30 T
2042-10-14 02:00:42 A
2043-04-09 18:57:49 T+
2043-10-03 03:01:49 A-
2044-02-28 20:24:39 As
2044-08-23 01:17:02 T
2045-02-16 23:56:07 A
2045-08-12 17:42:39 T
2046-02-05 23:06:26 A
2046-08-02 10:21:13 T
2047-01-26 01:33:18 P
2047-06-23 10:52:31 P
2047-07-22 22:36:17 P
2047-12-16 23:50:12 P
2048-06-11 12:58:53 A
2048-12-05 15:35:27 T
2049-05-31 13:59:59 A
2049-11-25 05:33:48 H
2050-05-20 20:42:50 H
2050-11-14 13:30:53 P
2051-04-11 02:10:39 P
2051-10-04 21:02:14 P
2052-03-30 18:31:53 T
2052-09-22 23:39:10 A
2053-03-20 07:08:19 A
2053-09-12 09:34:09 T
2054-03-09 12:33:40 P
2054-08-03 18:04:02 Pe
2054-09-02 01:09:34 P
2055-01-27 17:54:05 P
2055-07-24 09:57:50 T
2056-01-16 22:16:45 A
2056-07-12 20:21:59 A
2057-01-05 09:47:52 T
2057-07-01 23:40:15 A
2057-12-26 01:14:35 T
2058-05-22 10:39:25 P
2058-06-21 00:19:35 Pb
2058-11-16 03:23:07 P
2059-05-11 19:22:16 T
2059-11-05 09:18:15 A
2060-04-30 10:10:00 T
2060-10-24 09:24:10 A
2061-04-20 02:56:49 T
2061-10-13 10:32:10 A
2062-03-11 04:26:16 P
2062-09-03 08:54:27 P
2063-02-28 07:43:30 A
2063-08-24 01:22:11 T
2064-02-17 07:00:23 A
2064-08-12 17:46:06 T
2065-02-05 09:52:26 P
2065-07-03 17:33:52 P
2065-08-02 05:34:17 P
2065-12-27 08:39:56 P
2066-06-22 19:25:48 A
2066-12-17 00:23:40 T
2067-06-11 20:42:26 A
2067-12-06 14:03:43 H
2068-05-31 03:56:39 T
2068-11-24 21:32:30 P
2069-04-21 10:11:09 P
2069-05-20 17:53:18 Pb
2069-10-15 04:19:56 P
2070-04-11 02:36:09 T
2070-10-04 07:08:57 A
2071-03-31 15:01:06 A
2071-09-23 17:20:28 T
2072-03-19 20:10:31 P
2072-09-12 08:59:20 T
2073-02-07 01:55:59 P
2073-08-03 17:15:23 T
2074-01-27 06:44:15 A
2074-07-24 03:10:32 A
2075-01-16 18:36:04 T
2075-07-13 06:05:44 A
2076-01-06 10:07:27 T
2076-06-01 17:31:22 P
2076-07-01 06:50:43 P
2076-11-26 11:43:01 P
2077-05-22 02:46:05 T
2077-11-15 17:07:56 A
2078-05-11 17:56:55 T
2078-11-04 16:55:44 A
2079-05-01 10:50:13 T
2079-10-24 18:11:21 A
2080-03-21 12:20:15 P
2080-09-13 16:38:09 P
2081-03-10 15:23:31 A
2081-09-03 09:07:31 T
2082-02-27 14:47:00 A
2082-08-24 01:16:21 T
2083-02-16 18:06:36 P
2083-07-15 00:14:23 Pe
2083-08-13 12:34:41 P
2084-01-07 17:30:24 P
2084-07-03 01:50:26 A
2084-12-27 09:13:48 T
2085-06-22 03:21:16 A
2085-12-16 22:37:48 A
2086-06-11 11:07:14 T
2086-12-06 05:38:55 P
2087-05-02 18:04:42 P
2087-06-01 01:27:14 P
2087-10-26 11:46:57 P
2088-04-21 10:31:49 T
2088-10-14 14:48:05 A
2089-04-10 22:44:42 A
2089-10-04 01:15:23 T
2090-03-31 03:38:08 P
2090-09-23 16:56:36 T
2091-02-18 09:54:40 P
2091-08-15 00:34:43 T
2092-02-07 15:10:20 A
2092-08-03 09:59:33 A
2093-01-27 03:22:16 T
2093-07-23 12:32:04 A
2094-01-16 18:59:03 T
2094-06-13 00:22:11 P
2094-07-12 13:24:35 P
2094-12-07 20:05:56 P
2095-06-02 10:07:40 T
2095-11-27 01:02:57 A
2096-05-22 01:37:14 T
2096-11-15 00:36:15 A
2097-05-11 18:34:31 T
2097-11-04 02:01:25 A
2098-04-01 20:02:31 P
2098-09-25 00:31:16 P
2098-10-24 10:36:11 Pb
2099-03-21 22:54:32 A
2099-09-14 16:57:53 T
"""

_LUNAR_ECLIPSE_CATALOG_21ST_CENTURY = """2001-01-09 20:21:40 T
2001-07-05 14:56:23 P
2001-12-30 10:30:22 N
2002-05-26 12:04:26 N
2002-06-24 21:28:13 N
2002-11-20 01:47:40 N
2003-05-16 03:41:13 T
2003-11-09 01:19:38 T
2004-05-04 20:31:17 T
2004-10-28 03:05:11 T
2005-04-24 09:55:55 N
2005-10-17 12:04:27 P
2006-03-14 23:48:34 Nx
2006-09-07 18:52:25 P
2007-03-03 23:21:59 T
2007-08-28 10:38:27 T-
2008-02-21 03:27:09 T
2008-08-16 21:11:12 P
2009-02-09 14:39:22 N
2009-07-07 09:39:43 N
2009-08-06 00:40:18 N
2009-12-31 19:23:46 P
2010-06-26 11:39:34 P
2010-12-21 08:18:04 T
2011-06-15 20:13:43 T+
2011-12-10 14:32:56 T
2012-06-04 11:04:20 P
2012-11-28 14:34:07 N
2013-04-25 20:08:38 P
2013-05-25 04:11:06 Nb
2013-10-18 23:51:25 N
2014-04-15 07:46:48 T
2014-10-08 10:55:44 T
2015-04-04 12:01:24 T
2015-09-28 02:48:17 T
2016-03-23 11:48:21 N
2016-09-16 18:55:27 N
2017-02-11 00:45:03 N
2017-08-07 18:21:38 P
2018-01-31 13:31:00 T
2018-07-27 20:22:54 T+
2019-01-21 05:13:27 T
2019-07-16 21:31:55 P
2020-01-10 19:11:11 N
2020-06-05 19:26:14 N
2020-07-05 04:31:12 N
2020-11-30 09:44:01 N
2021-05-26 11:19:53 T
2021-11-19 09:04:06 P
2022-05-16 04:12:42 T-
2022-11-08 11:00:22 T+
2023-05-05 17:24:05 N
2023-10-28 20:15:18 P
2024-03-25 07:13:59 N
2024-09-18 02:45:25 P
2025-03-14 06:59:56 T
2025-09-07 18:12:58 T
2026-03-03 11:34:52 T
2026-08-28 04:14:04 P
2027-02-20 23:14:06 N
2027-07-18 16:04:09 Ne
2027-08-17 07:14:59 N
2028-01-12 04:14:13 P
2028-07-06 18:20:57 P
2028-12-31 16:53:15 T
2029-06-26 03:23:22 T+
2029-12-20 22:43:12 T
2030-06-15 18:34:34 P
2030-12-09 22:28:51 N
2031-05-07 03:52:02 N
2031-06-05 11:45:17 N
2031-10-30 07:46:45 N
2032-04-25 15:14:51 T
2032-10-18 19:03:40 T
2033-04-14 19:13:51 T
2033-10-08 10:56:23 T
2034-04-03 19:06:59 N
2034-09-28 02:47:37 P
2035-02-22 09:06:12 N
2035-08-19 01:12:15 P
2036-02-11 22:13:06 T
2036-08-07 02:52:32 T+
2037-01-31 14:01:38 T
2037-07-27 04:09:53 P
2038-01-21 03:49:52 N
2038-06-17 02:45:02 N
2038-07-16 11:35:56 N
2038-12-11 17:45:00 N
2039-06-06 18:54:25 P
2039-11-30 16:56:28 P
2040-05-26 11:46:22 T-
2040-11-18 19:04:40 T+
2041-05-16 00:43:03 P
2041-11-08 04:35:05 P
2042-04-05 14:30:11 N
2042-09-29 10:45:47 N
2043-03-25 14:32:04 T
2043-09-19 01:51:50 T
2044-03-13 19:38:33 T
2044-09-07 11:20:44 T
2045-03-03 07:43:26 N
2045-08-27 13:54:50 N
2046-01-22 13:02:37 P
2046-07-18 01:06:05 P
2047-01-12 01:26:14 T
2047-07-07 10:35:45 T-
2048-01-01 06:53:55 T
2048-06-26 02:02:28 P
2048-12-20 06:27:48 N
2049-05-17 11:26:39 N
2049-06-15 19:14:12 N
2049-11-09 15:52:11 N
2050-05-06 22:32:02 T
2050-10-30 03:21:47 T
2051-04-26 02:16:28 T
2051-10-19 19:11:50 T-
2052-04-14 02:18:06 N
2052-10-08 10:45:58 P
2053-03-04 17:22:10 N
2053-08-29 08:05:50 Nx
2054-02-22 06:51:27 T
2054-08-18 09:26:30 T
2055-02-11 22:46:17 T
2055-08-07 10:53:18 P
2056-02-01 12:26:06 N
2056-06-27 10:03:09 N
2056-07-26 18:43:24 N
2056-12-22 01:48:56 N
2057-06-17 02:26:20 P
2057-12-11 00:53:38 P
2058-06-06 19:15:48 T-
2058-11-30 03:16:18 T+
2059-05-27 07:55:35 P
2059-11-19 13:01:36 P
2060-04-15 21:37:04 N
2060-10-09 18:53:32 N
2060-11-08 04:04:15 N
2061-04-04 21:54:05 T
2061-09-29 09:38:13 T
2062-03-25 03:33:50 T
2062-09-18 18:34:02 T
2063-03-14 16:05:49 P
2063-09-07 20:41:12 N
2064-02-02 21:48:57 P
2064-07-28 07:52:48 P
2065-01-22 09:58:58 T
2065-07-17 17:48:40 T-
2066-01-11 15:04:47 T
2066-07-07 09:30:29 P
2066-12-31 14:30:10 N
2067-05-28 18:56:08 N
2067-06-27 02:41:06 N
2067-11-21 00:04:42 N
2068-05-17 05:42:17 P
2068-11-09 11:47:00 T
2069-05-06 09:09:57 T+
2069-10-30 03:35:06 T-
2070-04-25 09:21:24 Nx
2070-10-19 18:51:12 P
2071-03-16 01:31:09 N
2071-09-09 15:05:41 N
2072-03-04 15:23:07 T
2072-08-28 16:05:42 T
2073-02-22 07:24:53 T
2073-08-17 17:42:41 T
2074-02-11 20:55:58 N
2074-07-08 17:21:38 N
2074-08-07 01:56:03 N
2075-01-02 09:55:03 N
2075-06-28 09:55:35 P
2075-12-22 08:55:55 P
2076-06-17 02:39:47 T-
2076-12-10 11:34:51 T+
2077-06-06 14:59:52 P
2077-11-29 21:35:53 P
2078-04-27 04:35:44 N
2078-10-21 03:08:03 N
2078-11-19 12:40:04 N
2079-04-16 05:10:45 P
2079-10-10 17:30:30 T
2080-04-04 11:23:38 T
2080-09-29 01:52:42 T
2081-03-25 00:22:01 P
2081-09-18 03:35:26 N
2082-02-13 06:29:19 P
2082-08-08 14:46:42 Nx
2083-02-02 18:26:46 T
2083-07-29 01:05:34 T-
2084-01-22 23:13:00 T
2084-07-17 16:58:51 P
2085-01-10 22:32:29 N
2085-06-08 02:17:36 N
2085-07-07 10:04:40 N
2085-12-01 08:25:35 N
2086-05-28 12:43:47 P
2086-11-20 20:19:42 P
2087-05-17 15:55:20 T+
2087-11-10 12:05:33 T-
2088-05-05 16:16:50 P
2088-10-30 03:03:20 P
2089-03-26 09:34:14 N
2089-09-19 22:11:17 N
2090-03-15 23:48:31 T
2090-09-08 22:52:29 T
2091-03-05 15:58:22 T
2091-08-29 00:38:25 T
2092-02-23 05:20:59 N
2092-07-19 00:41:58 Ne
2092-08-17 09:13:59 N
2093-01-12 18:00:03 N
2093-07-08 17:24:18 P
2094-01-01 17:00:06 P
2094-06-28 10:01:57 T+
2094-12-21 19:56:32 T+
2095-06-17 22:00:11 P
2095-12-11 06:15:02 P
2096-05-07 11:24:42 N
2096-06-06 02:43:41 Nb
2096-10-31 11:30:23 N
2096-11-29 21:22:22 N
2097-04-26 12:18:17 P
2097-10-21 01:30:55 T
2098-04-15 19:04:48 T-
2098-10-10 09:19:58 T
2099-04-05 08:30:56 P
2099-09-29 10:36:38 Nx
"""

_SOLAR_ECLIPSE_TYPE_LABELS = {
    "P": "Partial solar eclipse",
    "A": "Annular solar eclipse",
    "T": "Total solar eclipse",
    "H": "Hybrid solar eclipse",
}

_LUNAR_ECLIPSE_TYPE_LABELS = {
    "N": "Penumbral lunar eclipse",
    "P": "Partial lunar eclipse",
    "T": "Total lunar eclipse",
}


def _eclipse_type_label(mode, type_code):
    lead = str(type_code).strip()[:1].upper()
    if mode == "solar":
        return _SOLAR_ECLIPSE_TYPE_LABELS.get(lead, f"Solar eclipse ({type_code})")
    return _LUNAR_ECLIPSE_TYPE_LABELS.get(lead, f"Lunar eclipse ({type_code})")


def _parse_eclipse_catalog(mode):
    table = (_SOLAR_ECLIPSE_CATALOG_21ST_CENTURY if mode == "solar"
             else _LUNAR_ECLIPSE_CATALOG_21ST_CENTURY)
    entries = []
    for line in table.splitlines():
        line = line.strip()
        if not line:
            continue
        date, time_utc, type_code = line.split()[:3]
        type_label = _eclipse_type_label(mode, type_code)
        entries.append({
            "mode": mode,
            "key": f"{mode}-{date}",
            "event": date,
            "date": date,
            "time_utc": time_utc,
            "type_code": type_code,
            "type_label": type_label,
            "label": f"{date} {time_utc} UTC — {type_label}",
            "source": "NASA/GSFC Five Millennium Canon eclipse catalog, 2001-2100 page",
        })
    return entries


def eclipse_catalog_21st_century(mode=None):
    """Return compact NASA-derived 2001-2100 solar/lunar eclipse metadata."""
    if mode is None:
        return _parse_eclipse_catalog("lunar") + _parse_eclipse_catalog("solar")
    mode = str(mode).strip().lower()
    if mode not in {"lunar", "solar"}:
        raise ValueError("mode must be 'lunar', 'solar', or None")
    return _parse_eclipse_catalog(mode)


def _catalog_event_by_date(mode, date):
    for entry in eclipse_catalog_21st_century(mode):
        if entry["date"] == date or entry["key"] == date:
            return entry
    return None


def _render_call_for_catalog_entry(entry):
    if entry["mode"] == "lunar":
        return ("from ssatk.plots.eclipse_space_view_plotly import plot_space_view_animated\n"
                f"plot_space_view_animated(mode='lunar', event='{entry['date']}', "
                "save_path='lunar_eclipse.html')")
    if entry["date"] == "2024-04-08":
        return ("from ssatk.plots.eclipse_space_view_plotly import plot_2024_solar_eclipse_animated\n"
                "plot_2024_solar_eclipse_animated(save_path='solar_eclipse.html')")
    return ("from ssatk.plots.eclipse_space_view_plotly import plot_2024_solar_eclipse_animated\n"
            f"# Solar event '{entry['date']}' is in the selector catalog; use the date/time "
            "as the center for a custom real-ephemeris solar render.")


def _default_catalog_key(mode, default_event=None):
    if default_event:
        key = str(default_event).strip().lower()
        if key in {"2014", "2014-04-15", "2014-04-15-total-lunar-wisconsin"}:
            return "lunar-2014-04-15"
        if key in {"2024", "2024-04-08", "2024-04-08-total-solar"}:
            return "solar-2024-04-08"
        if key.startswith("lunar-") or key.startswith("solar-"):
            return key
        if len(key) >= 10 and key[4] == "-" and key[7] == "-":
            return f"{mode}-{key[:10]}"
    return "lunar-2014-04-15" if mode == "lunar" else "solar-2024-04-08"


def _eclipse_catalog_dropdown_payload(default_mode, default_event=None):
    default_key = _default_catalog_key(default_mode, default_event)
    entries = eclipse_catalog_21st_century()
    for entry in entries:
        entry["render_call"] = _render_call_for_catalog_entry(entry)
        if entry["key"] == "lunar-2014-04-15":
            entry["default_note"] = "Default lunar scene selected by Travis: Wisconsin-visible total lunar eclipse."
        elif entry["key"] == "solar-2024-04-08":
            entry["default_note"] = "Default solar scene selected by Travis: central Texas total solar eclipse."
        else:
            entry["default_note"] = "Catalog selection; regenerate the scene with the shown Python call."
    return {
        "default_mode": default_mode,
        "default_key": default_key,
        "rendered_note": (
            "This static Plotly scene is rendered for the default event. The selector lists every "
            "NASA/GSFC cataloged 21st-century eclipse and shows the event metadata plus a Python "
            "call to generate a dedicated scene."
        ),
        "entries": entries,
    }


def _inject_eclipse_catalog_dropdown(html_path, *, default_mode, default_event=None):
    """Add a self-contained 21st-century eclipse selector to a Plotly HTML file."""
    path = str(html_path)
    with open(path, "r", encoding="utf-8") as handle:
        html = handle.read()
    marker_start = "<!-- SSATK_ECLIPSE_CATALOG_SELECTOR_START -->"
    marker_end = "<!-- SSATK_ECLIPSE_CATALOG_SELECTOR_END -->"
    if marker_start in html and marker_end in html:
        before = html.split(marker_start, 1)[0]
        after = html.split(marker_end, 1)[1]
        html = before + after

    payload_json = json.dumps(_eclipse_catalog_dropdown_payload(default_mode, default_event))
    selector = f"""
{marker_start}
<style>
#ssatk-eclipse-selector {{
  position: fixed; left: 16px; top: 16px; z-index: 10000; width: min(430px, calc(100vw - 32px));
  color: #eef3ff; background: rgba(5, 8, 16, 0.82); border: 1px solid rgba(180, 205, 255, 0.32);
  border-radius: 12px; padding: 12px 14px; font: 13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  box-shadow: 0 8px 28px rgba(0,0,0,0.42); backdrop-filter: blur(5px);
}}
#ssatk-eclipse-selector h2 {{ margin: 0 0 8px; font-size: 15px; color: #ffffff; }}
#ssatk-eclipse-selector label {{ display: block; margin-top: 8px; margin-bottom: 3px; color: #b9c8e8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }}
#ssatk-eclipse-selector select {{ width: 100%; background: #111827; color: #f8fbff; border: 1px solid #42506a; border-radius: 7px; padding: 6px 7px; }}
#ssatk-eclipse-details {{ margin-top: 9px; color: #d9e4ff; }}
#ssatk-eclipse-details code {{ display: block; white-space: pre-wrap; margin-top: 6px; padding: 7px; background: rgba(255,255,255,0.08); border-radius: 7px; color: #d9fff0; font-size: 11px; }}
#ssatk-eclipse-selector .note {{ margin-top: 7px; color: #aab7d4; font-size: 11px; }}
#ssatk-eclipse-selector .close {{ float: right; margin-left: 8px; background: transparent; color: #d8e3ff; border: 0; font-size: 18px; cursor: pointer; }}
</style>
<div id="ssatk-eclipse-selector" role="group" aria-label="21st-century eclipse selector">
  <button class="close" type="button" title="Hide selector" onclick="document.getElementById('ssatk-eclipse-selector').style.display='none'">×</button>
  <h2>21st-Century Eclipse Selector</h2>
  <label for="ssatk-eclipse-mode">Eclipse Family</label>
  <select id="ssatk-eclipse-mode">
    <option value="lunar">Lunar eclipses</option>
    <option value="solar">Solar eclipses</option>
  </select>
  <label for="ssatk-eclipse-event">Catalog Event</label>
  <select id="ssatk-eclipse-event"></select>
  <div id="ssatk-eclipse-details"></div>
  <div class="note" id="ssatk-eclipse-rendered-note"></div>
</div>
<script>
(function() {{
  const payload = {payload_json};
  const modeSelect = document.getElementById('ssatk-eclipse-mode');
  const eventSelect = document.getElementById('ssatk-eclipse-event');
  const details = document.getElementById('ssatk-eclipse-details');
  const renderedNote = document.getElementById('ssatk-eclipse-rendered-note');
  renderedNote.textContent = payload.rendered_note;
  function entriesForMode(mode) {{ return payload.entries.filter(e => e.mode === mode); }}
  function escapeHtml(text) {{ return String(text).replace(/[&<>]/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[ch])); }}
  function populateEvents() {{
    const mode = modeSelect.value;
    const entries = entriesForMode(mode);
    eventSelect.innerHTML = '';
    for (const entry of entries) {{
      const opt = document.createElement('option');
      opt.value = entry.key;
      opt.textContent = entry.label;
      eventSelect.appendChild(opt);
    }}
    const preferred = entries.find(e => e.key === payload.default_key) || entries[0];
    if (preferred) eventSelect.value = preferred.key;
    updateDetails();
  }}
  function updateDetails() {{
    const entry = payload.entries.find(e => e.key === eventSelect.value);
    if (!entry) return;
    details.innerHTML = '<strong>' + escapeHtml(entry.type_label) + '</strong><br>' +
      escapeHtml(entry.date + ' ' + entry.time_utc + ' UTC') + '<br>' +
      'NASA type code: ' + escapeHtml(entry.type_code) + '<br>' +
      '<span style="color:#aab7d4">' + escapeHtml(entry.default_note) + '</span>' +
      '<code>' + escapeHtml(entry.render_call) + '</code>';
  }}
  modeSelect.value = payload.default_mode;
  modeSelect.addEventListener('change', populateEvents);
  eventSelect.addEventListener('change', updateDetails);
  populateEvents();
}})();
</script>
{marker_end}
"""
    if "</body>" in html:
        html = html.replace("</body>", selector + "\n</body>", 1)
    else:
        html += selector
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)


def _event_date_from_key(mode, event):
    key = str(event).strip().lower().replace("_", "-").replace(" ", "-")
    prefix = f"{mode}-"
    if key.startswith(prefix):
        key = key[len(prefix):]
    return key[:10] if len(key) >= 10 and key[4] == "-" and key[7] == "-" else key


def _normalize_lunar_event_key(event):
    if event in (None, False, ""):
        return None
    key = str(event).strip().lower().replace("_", "-").replace(" ", "-")
    if key in _LUNAR_EVENT_ALIASES:
        return REAL_LUNAR_ECLIPSE_2014["key"]
    date = _event_date_from_key("lunar", event)
    if _catalog_event_by_date("lunar", date) is not None:
        return f"lunar-{date}"
    raise ValueError(
        "Unsupported lunar eclipse event {!r}; use a date from "
        "eclipse_catalog_21st_century('lunar'), for example '2014-04-15'.".format(event)
    )


def _lunar_event_metadata(event):
    key = _normalize_lunar_event_key(event)
    if key is None:
        return None
    if key == REAL_LUNAR_ECLIPSE_2014["key"]:
        meta = dict(REAL_LUNAR_ECLIPSE_2014)
        meta["date"] = "2014-04-15"
        meta["type_label"] = "Total lunar eclipse"
        return meta
    date = _event_date_from_key("lunar", key)
    entry = _catalog_event_by_date("lunar", date)
    if entry is None:
        raise ValueError(f"No 21st-century lunar eclipse catalog entry for {event!r}")
    return {
        "key": entry["key"],
        "label": entry["type_label"] + " — " + entry["date"],
        "date": entry["date"],
        "type_label": entry["type_label"],
        "peak_utc": f"{entry['date']}T{entry['time_utc']}",
        "source": entry["source"],
    }


def _real_solar_system_vectors(times):
    """Return real geocentric Moon vector and Sun direction for Astropy times.

    The vectors use Astropy's built-in solar-system ephemeris so the demo stays
    offline-capable while still using the actual 2014 epoch instead of the older
    synthetic fixed-node Moon orbit.
    """
    import astropy.units as u
    from astropy.coordinates import get_body_barycentric_posvel, solar_system_ephemeris

    with solar_system_ephemeris.set("builtin"):
        earth_pos, _ = get_body_barycentric_posvel("earth", times)
        moon_pos, _ = get_body_barycentric_posvel("moon", times)
        sun_pos, _ = get_body_barycentric_posvel("sun", times)

    r_moon_km = (moon_pos.xyz - earth_pos.xyz).to_value(u.km).T
    r_sun_km = (sun_pos.xyz - earth_pos.xyz).to_value(u.km).T
    sun_dist_km = np.linalg.norm(r_sun_km, axis=1)
    sun_hat = r_sun_km / sun_dist_km[:, None]
    return r_moon_km, sun_hat, sun_dist_km


def _wisconsin_visibility_at_peak(peak_time):
    """Moon/Sun altitude check for the 2014 event from Madison, Wisconsin."""
    import astropy.units as u
    from astropy.coordinates import AltAz, EarthLocation, get_body, solar_system_ephemeris

    meta = REAL_LUNAR_ECLIPSE_2014
    location = EarthLocation(
        lat=meta["observer_lat_deg"] * u.deg,
        lon=meta["observer_lon_deg"] * u.deg,
        height=meta["observer_height_m"] * u.m,
    )
    frame = AltAz(obstime=peak_time, location=location)
    with solar_system_ephemeris.set("builtin"):
        moon_alt = get_body("moon", peak_time, location).transform_to(frame).alt.deg
        sun_alt = get_body("sun", peak_time, location).transform_to(frame).alt.deg
    return float(moon_alt), float(sun_alt)


def _real_lunar_event_window(event, *, n_steps=4000, half_window_hr=6.0):
    key = _normalize_lunar_event_key(event)
    if key is None:
        return None

    import astropy.units as u
    from astropy.time import Time

    meta = _lunar_event_metadata(event)
    peak_time = Time(meta["peak_utc"], scale="utc")
    offsets_s = np.linspace(-half_window_hr * 3600.0, half_window_hr * 3600.0, int(n_steps))
    times = peak_time + offsets_s * u.s
    r_moon, sun_hat, sun_dist = _real_solar_system_vectors(times)
    illum = illumination_fraction(
        r_moon,
        sun_hat,
        R_body_km=RE_KM,
        R_sun_km=R_SUN_KM,
        D_km=sun_dist,
    )
    peak_idx = int(np.argmin(illum))

    out = {
        "mode": "lunar",
        "event_key": key,
        "event_label": meta["label"],
        "event_source": meta["source"],
        "peak_utc": meta["peak_utc"],
        "peak_time": peak_time,
        "t_s": offsets_s,
        "times": times,
        "r_moon": r_moon,
        "sun_hat": sun_hat,
        "sun_dist_km": sun_dist,
        "illum": illum,
        "peak_idx": peak_idx,
        "epoch_jd": float(peak_time.jd),
    }
    if meta.get("observer_name"):
        moon_alt_deg, sun_alt_deg = _wisconsin_visibility_at_peak(peak_time)
        out.update({
            "observer_name": meta["observer_name"],
            "observer_lat_deg": meta["observer_lat_deg"],
            "observer_lon_deg": meta["observer_lon_deg"],
            "observer_moon_alt_deg": moon_alt_deg,
            "observer_sun_alt_deg": sun_alt_deg,
        })
    return out


def _synthetic_eclipse_window(mode, search_days=None, *, n_steps=4000, verbose=True):
    if search_days is None:
        search_days = 365.0 if mode == "lunar" else 365.25 * 6
    lunar_period_days = 27.32
    n_orbits_year = search_days / lunar_period_days
    coarse_density = 60 if mode == "lunar" else 1500
    t_s, r_moon, _ = propagate_eci(
        a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
        raan_deg=0.0, argp_deg=0.0, nu0_deg=0.0,
        n_orbits=n_orbits_year, n_steps=int(n_orbits_year * coarse_density),
    )
    sun_hat = sun_direction_eci(t_s)
    r_eval = r_moon if mode == "lunar" else -r_moon
    R_occ = RE_KM if mode == "lunar" else R_MOON_KM
    illum_coarse = illumination_fraction(r_eval, sun_hat, R_body_km=R_occ,
                                          R_sun_km=R_SUN_KM, D_km=AU_KM)
    best_idx = int(np.argmin(illum_coarse))
    best_t = t_s[best_idx]
    if verbose:
        print(f"[{mode}] Coarse search over {search_days:.0f} days: "
              f"deepest illumination minimum = {illum_coarse[best_idx]:.4f} "
              f"at t={best_t/86400:.1f} days")

    MU_EARTH_KM3S2 = EARTH_MU_KM3_S2
    window_days = 2.0
    n_rad_s = np.sqrt(MU_EARTH_KM3S2 / D_MOON_A_KM**3)
    t_window_start = best_t - window_days * 86400
    M_start = (n_rad_s * t_window_start) % (2*np.pi)
    E_start = float(M_start)
    for _ in range(60):
        dE = (M_start - E_start + D_MOON_E*np.sin(E_start)) / (1 - D_MOON_E*np.cos(E_start))
        E_start += dE
    nu_start = 2*np.arctan2(np.sqrt(1+D_MOON_E)*np.sin(E_start/2), np.sqrt(1-D_MOON_E)*np.cos(E_start/2))
    n_orbits_window = (2 * window_days) / lunar_period_days

    t_fine, r_fine, _ = propagate_eci(
        a_km=D_MOON_A_KM, e=D_MOON_E, inc_deg=D_MOON_INC_DEG,
        raan_deg=0.0, argp_deg=0.0, nu0_deg=np.degrees(nu_start),
        n_orbits=n_orbits_window, n_steps=int(n_steps),
    )
    t_fine = t_fine + t_window_start
    sun_fine = sun_direction_eci(t_fine)
    r_eval_fine = r_fine if mode == "lunar" else -r_fine
    illum_fine = illumination_fraction(r_eval_fine, sun_fine, R_body_km=R_occ,
                                        R_sun_km=R_SUN_KM, D_km=AU_KM)
    fine_mask = np.abs(t_fine - best_t) < window_days * 86400
    t_win = t_fine[fine_mask]
    r_win = r_fine[fine_mask]
    sun_win = sun_fine[fine_mask]
    illum_win = illum_fine[fine_mask]
    peak_idx = int(np.argmin(illum_win))
    return {
        "mode": mode,
        "event_key": None,
        "event_label": None,
        "t_s": t_win,
        "times": None,
        "r_moon": r_win,
        "sun_hat": sun_win,
        "sun_dist_km": np.full_like(illum_win, AU_KM, dtype=float),
        "illum": illum_win,
        "peak_idx": peak_idx,
        "epoch_jd": 2_460_500.0,
    }


def _eclipse_window(mode, search_days=None, *, event=None, n_steps=4000, verbose=True):
    assert mode in ("lunar", "solar")
    if event is not None:
        if mode != "lunar":
            raise ValueError("Named real-event support is currently implemented for lunar eclipses only.")
        return _real_lunar_event_window(event, n_steps=n_steps)
    return _synthetic_eclipse_window(mode, search_days=search_days, n_steps=n_steps, verbose=verbose)


def _earth_rotation_for_window_sample(window, idx):
    times = window.get("times")
    if times is not None:
        return earth_rotation_deg_from_time(times[idx])
    return earth_rotation_deg_from_time(
        epoch_jd=window.get("epoch_jd", 2_460_500.0),
        relative_seconds=float(window["t_s"][idx]),
    )


def _window_title_prefix(mode, window):
    return window.get("event_label") or f"{mode.capitalize()} eclipse"




def moon_color(illum_frac):
    """Lunar-eclipse 'Blood Moon' colour ramp — grey dims toward deep red.
    Used for the flat 2D appearance-strip circles (no texture to preserve
    there, so a colour blend is fine)."""
    grey = np.array([0.82, 0.82, 0.85])
    red_totality = np.array([0.45, 0.10, 0.06])
    red_mix = np.clip((0.35 - illum_frac) / 0.35, 0, 1) ** 1.5
    base = grey * np.clip(0.15 + 0.85*illum_frac, 0.15, 1.0)
    return base * (1 - red_mix) + red_totality * red_mix


def moon_brightness(illum_frac):
    """
    Real 'darker due to lack of sunlight' model for the TEXTURED 3D Moon —
    brightness only (floor 0.12, never literally black), so craters/mare
    stay clearly visible even in deep shadow. This is what the previous
    colour-replace version (moon_color, above) got wrong when applied to
    a textured sphere: multiplying every point toward the same dark red
    made the crater contrast nearly disappear.
    """
    floor = 0.12
    return np.clip(floor + (1 - floor) * illum_frac, floor, 1.0)


def moon_red_bias(illum_frac):
    """Subtle warm/red multiplicative tint during deep shadow (same real
    cause as before — Earth-atmosphere-scattered sunlight reaching the
    Moon even in geometric shadow) — weak enough to not wash out texture,
    since it multiplies the already-textured, already-darkened base."""
    red_mix = np.clip((0.35 - illum_frac) / 0.35, 0, 1) ** 1.5
    warm = np.array([1.15, 0.55, 0.42])
    return (1 - red_mix) + warm * red_mix


def _sun_sphere_mpl(ax, center, radius, seed=11):
    n = 36
    su, sv = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2)
    SU, SV = np.meshgrid(su, sv)
    nx, ny, nz = np.cos(SU)*np.sin(SV), np.sin(SU)*np.sin(SV), np.cos(SV)
    sx, sy, sz = center[0]+radius*nx, center[1]+radius*ny, center[2]+radius*nz

    rng = np.random.default_rng(seed)
    granulation = np.zeros_like(SU)
    for _ in range(40):
        c = rng.normal(size=3); c /= np.linalg.norm(c)
        dot = nx*c[0] + ny*c[1] + nz*c[2]
        spread = rng.uniform(0.85, 0.97)
        granulation += np.clip((dot - spread)/(1-spread), 0, 1) * rng.uniform(-0.25, 0.25)

    ref = np.array([0.4, 0.4, 0.82]); ref /= np.linalg.norm(ref)
    limb = np.clip(nx*ref[0]+ny*ref[1]+nz*ref[2], 0, 1) ** 0.35
    brightness = np.clip(0.55 + 0.45*limb + granulation, 0.15, 1.0)

    stops = np.array([0.15, 0.5, 0.75, 1.0])
    rgb_stops = np.array([[0.478,0.180,0.0],[0.851,0.333,0.039],[1.0,0.647,0.0],[1.0,0.984,0.918]])
    colors = np.empty(brightness.shape + (3,))
    for ch in range(3):
        colors[..., ch] = np.interp(brightness, stops, rgb_stops[:, ch])

    ax.plot_surface(sx, sy, sz, facecolors=colors, linewidth=0, shade=False, zorder=8)
    for gs, ga in [(1.4, 0.15), (1.9, 0.06)]:
        gx, gy, gz = center[0]+radius*gs*nx, center[1]+radius*gs*ny, center[2]+radius*gs*nz
        ax.plot_surface(gx, gy, gz, color="#FFD700", alpha=ga, linewidth=0, shade=False, zorder=7)


def _earth_sphere_mpl(ax, center, radius, sun_hat, seed=7):
    n_lat, n_lon = 40, 80
    lat = np.linspace(90, -90, n_lat)
    lon = np.linspace(-180, 180, n_lon)
    Lon, Lat = np.meshgrid(lon, lat)
    latr, lonr = np.radians(Lat), np.radians(Lon)
    nx, ny, nz = np.cos(latr)*np.cos(lonr), np.cos(latr)*np.sin(lonr), np.sin(latr)
    ex, ey, ez = center[0]+radius*nx, center[1]+radius*ny, center[2]+radius*nz

    try:
        from global_land_mask import globe
        lon_q = np.where(Lon >= 180, Lon-360, Lon)
        land = globe.is_land(np.clip(Lat, -89.999, 89.999), lon_q)
    except Exception:
        rng = np.random.default_rng(seed)
        field = np.zeros_like(Lat)
        for _ in range(14):
            clat, clon = rng.uniform(-60,60), rng.uniform(-180,180)
            spread = rng.uniform(15, 35)
            d = np.sqrt((Lat-clat)**2 + ((Lon-clon+180)%360-180)**2)
            field += np.exp(-(d**2)/(2*spread**2))
        land = field > 0.35

    ocean, landc = np.array([0.08,0.22,0.50]), np.array([0.20,0.50,0.18])
    rgb = np.where(land[...,None], landc, ocean)
    dot = nx*sun_hat[0] + ny*sun_hat[1] + nz*sun_hat[2]
    lit = np.clip(dot, 0, 1) ** 0.6
    night_tint = np.array([0.03,0.05,0.10])
    rgb_shaded = np.clip(rgb*lit[...,None] + night_tint*(1-lit[...,None]), 0, 1)
    ax.plot_surface(ex, ey, ez, facecolors=rgb_shaded, linewidth=0, shade=False, zorder=8)


def _moon_sphere_mpl(ax, center, radius, tint_rgb, seed=3):
    n = 30
    mu, mv = np.linspace(0, 2*np.pi, n), np.linspace(0, np.pi, n//2)
    muu, mvv = np.meshgrid(mu, mv)
    mx, my, mz = center[0]+radius*np.cos(muu)*np.sin(mvv), \
                 center[1]+radius*np.sin(muu)*np.sin(mvv), \
                 center[2]+radius*np.cos(mvv)
    rng = np.random.default_rng(seed)
    mlat, mlon = 90-np.degrees(mvv), np.degrees(muu)
    albedo = np.full_like(mlat, 0.85)
    for _ in range(6):
        clat, clon = rng.uniform(-40,55), rng.uniform(-70,70)
        spread = rng.uniform(12, 28)
        d = np.sqrt((mlat-clat)**2 + ((mlon-clon+180)%360-180)**2)
        albedo -= 0.30*np.exp(-(d**2)/(2*spread**2))
    for _ in range(60):
        clat, clon = rng.uniform(-85,85), rng.uniform(-180,180)
        radius_c = rng.uniform(2, 8)
        d = np.sqrt((mlat-clat)**2 + ((mlon-clon+180)%360-180)**2)
        albedo -= 0.35*np.clip(1 - d/radius_c, 0, 1)**2
    albedo = np.clip(albedo, 0.35, 1.0)
    rgb = np.clip(np.array(tint_rgb)[None,None,:] * albedo[...,None], 0, 1)
    ax.plot_surface(mx, my, mz, facecolors=rgb, linewidth=0, shade=False, zorder=9)


def find_and_plot_eclipse(mode="lunar", save_path=None, search_days=None, verbose=True, event=None):
    assert mode in ("lunar", "solar")
    window = _eclipse_window(mode, search_days=search_days, event=event, n_steps=4000, verbose=verbose)
    t_win = window["t_s"]
    r_win = window["r_moon"]
    sun_win = window["sun_hat"]
    illum_win = window["illum"]
    best_t = t_win[window["peak_idx"]]

    t_hr = (t_win - best_t) / 3600.0
    mid = int(np.argmin(illum_win))

    # Actual angular separation at peak — the real geometric quantity that
    # determines whether an eclipse happens at all and how deep it is
    # (this is what we were computing ad-hoc during debugging earlier;
    # now it's a first-class, displayed result instead of a side check).
    r_eval_mid = r_win[mid] if mode == "lunar" else -r_win[mid]
    occ_hat_mid = r_eval_mid / np.linalg.norm(r_eval_mid)
    cos_sep_mid = np.dot(-occ_hat_mid, sun_win[mid])
    sep_deg_mid = np.degrees(np.arccos(np.clip(cos_sep_mid, -1, 1)))

    frac_umbra = np.mean(illum_win < 0.02) * 100
    dt_mean = np.mean(np.diff(t_win)) if len(t_win) > 1 else 0
    dur_umbra_hr = np.sum(illum_win < 0.02) * dt_mean / 3600.0
    dur_total_hr = np.sum(illum_win < 0.999) * dt_mean / 3600.0

    if mode == "lunar":
        ecl_type = "Total" if illum_win.min() < 0.02 else "Partial" if illum_win.min() < 0.999 else "None"
    else:
        # Solar eclipses distinguish total/annular by relative angular
        # size (Moon vs Sun as seen from Earth) at the deepest point, not
        # just how dark it gets — an annular eclipse never reaches
        # illum=0 (a bright ring always remains) even at its deepest.
        r_mid = np.linalg.norm(r_win[mid])
        moon_ang = R_MOON_KM / r_mid
        sun_ang = R_SUN_KM / AU_KM
        if illum_win.min() < 0.001:
            ecl_type = "Total"
        elif moon_ang < sun_ang:
            ecl_type = "Annular"
        else:
            ecl_type = "Partial"

    event_label = _window_title_prefix(mode, window)
    event_note = ""
    if window.get("event_key") and window.get("observer_name"):
        event_note = (
            f"; peak {window['peak_utc']} UTC, "
            f"Moon altitude {window['observer_moon_alt_deg']:.1f}° from {window['observer_name']}"
        )
    elif window.get("event_key"):
        event_note = f"; peak {window['peak_utc']} UTC"

    if verbose:
        print(f"[{mode}] Refined minimum illumination: {illum_win.min():.4f} ({ecl_type})")
        print(f"[{mode}] Angular separation at peak: {sep_deg_mid:.3f} deg")
        print(f"[{mode}] Peak phase duration: {dur_umbra_hr*60:.1f} min, "
              f"total event duration: {dur_total_hr:.2f} hr{event_note}")

    fig = plt.figure(figsize=(13.5, 7.2), dpi=115)
    grid = fig.add_gridspec(2, 1, height_ratios=(3.3, 1.15), hspace=0.38)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])

    label = "Blood Moon (umbra)" if mode == "lunar" else "Deepest coverage"
    ax1.plot(t_hr, illum_win, color="#222222", linewidth=1.3)
    ax1.fill_between(t_hr, 0, illum_win, where=(illum_win < 0.999),
                     color="#553322", alpha=0.3, label="Partial phase")
    ax1.fill_between(t_hr, 0, illum_win, where=(illum_win < 0.02),
                     color="#3a0f08", alpha=0.75, label=label)
    ax1.set_xlabel("Time relative to deepest point [hours]")
    ax1.set_ylabel("Illumination fraction" if mode == "lunar" else "Sun visible fraction")
    title_suffix = ""
    if window.get("event_key") and window.get("observer_name"):
        title_suffix = (
            f"\n{window['peak_utc']} UTC; visible from {window['observer_name']} "
            f"(Moon alt {window['observer_moon_alt_deg']:.1f}°)"
        )
    elif window.get("event_key"):
        title_suffix = f"\n{window['peak_utc']} UTC"
    ax1.set_title(f"{ecl_type} {event_label}\n"
                  f"min={illum_win.min():.3f}, angle at peak={sep_deg_mid:.3f}°, "
                  f"peak phase {dur_umbra_hr*60:.0f} min, event {dur_total_hr:.1f} hr"
                  f"{title_suffix}")
    ax1.set_ylim(-0.02, 1.05)
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.25)

    # ── Panel 2: appearance strip ────────────────────────────────────────────
    # Frame times are chosen non-linearly, scaled to THIS event's own real
    # duration (dur_total_hr) rather than spread evenly across the full
    # +/-48h search window — evenly-spaced-in-time frames spent almost all
    # of their budget on flat, unchanging full-moon/full-sun frames (the
    # event itself is only ~4 hours out of a 96-hour window) and gave only
    # one real transition frame. Denser sampling near the peak (matching
    # both reference photos' look: several visible partial-phase steps,
    # not just one) with a couple of untouched frames at each end for
    # context.
    n_frames = 9
    n_side = n_frames // 2
    bound_hr = max(dur_total_hr * 3.0, 6.0)
    ingress = np.where((t_hr < 0) & (t_hr > -bound_hr))[0]
    egress = np.where((t_hr > 0) & (t_hr < bound_hr))[0]
    illum_min = illum_win.min()

    # Target specific illumination LEVELS (not time offsets) for each
    # step, bunched non-linearly toward the deepest point — this
    # guarantees each frame shows a materially different phase (a real
    # crescent step) instead of time-based sampling, which could jump
    # straight from "full" to "totality" if the actual transition happens
    # faster than the time-grid spacing.
    #
    # The innermost target is capped at illum_min + 8% of the range
    # rather than illum_min itself — totality is genuinely FLAT at
    # illum_min for its whole duration (a real physical fact, not a
    # sampling gap), so a target of exactly illum_min ties with dozens of
    # indices and, combined with the already-shown centre frame, produced
    # duplicate all-red panels instead of a fourth distinct crescent step.
    frac_lin = np.linspace(0.05, 0.92, n_side)
    frac = frac_lin ** 0.6
    illum_targets = 1.0 - frac * (1.0 - illum_min)

    idxs = np.zeros(n_frames, dtype=int)

    # The centre panel should be the MOST-aligned instant, not merely the
    # first instant illum reaches its floor — illum is exactly flat for
    # totality's whole duration (a real physical fact), so argmin(illum)
    # alone ties across every point in that window and can land anywhere
    # in it, including a point still measurably off-centre within the
    # umbra (enough to render a faint sliver instead of a clean solid
    # disc / corona). Re-searching by actual geometric separation within
    # the flat zone picks the true best-aligned instant instead.
    flat_zone = np.where(illum_win <= illum_min + 1e-6)[0]
    if mode == "lunar":
        def _sep(i):
            dist_i = np.linalg.norm(r_win[i])
            occ_hat = r_win[i] / dist_i
            return np.arccos(np.clip(np.dot(-occ_hat, sun_win[i]), -1, 1))
    else:
        def _sep(i):
            r_i = np.linalg.norm(r_win[i])
            moon_hat = r_win[i] / r_i
            return np.arccos(np.clip(np.dot(moon_hat, sun_win[i]), -1, 1))
    seps = np.array([_sep(i) for i in flat_zone])
    center_idx = flat_zone[np.argmin(seps)]
    idxs[n_side] = center_idx

    for step, target in enumerate(illum_targets):
        k_in = step
        k_out = n_frames - 1 - step
        idxs[k_in] = ingress[np.argmin(np.abs(illum_win[ingress] - target))]
        idxs[k_out] = egress[np.argmin(np.abs(illum_win[egress] - target))]

    ax2.set_xlim(0, n_frames); ax2.set_ylim(0, 1); ax2.set_aspect('equal')
    ax2.set_facecolor("black")
    for k, idx in enumerate(idxs):
        # arccos() below always returns a positive magnitude — with no
        # sign, both renderers were showing the shadow/Moon approach from
        # one side and recede back to that SAME side, instead of sweeping
        # continuously through in one direction the way a real eclipse
        # does. Real motion during the brief eclipse window is close
        # enough to a straight-line pass that the sign of t_hr (before
        # vs after the deepest point) is a fine stand-in for the actual
        # direction of travel, without needing a full 2D vector
        # projection just for this cosmetic detail.
        side = -1.0 if t_hr[idx] < 0 else (1.0 if t_hr[idx] > 0 else 0.0)
        if mode == "lunar":
            # Real per-pixel crescent render instead of a flat colour
            # disc: convert this instant's REAL angular separation
            # between the Moon and Earth's shadow axis (the same
            # geometry already driving illum_win, not a synthetic
            # re-sweep) into the "shadow-offset in Moon-radii" units
            # render_lunar_panel expects, then rasterize an actual
            # crescent-shaped shadow at that offset.
            dist_i = np.linalg.norm(r_win[idx])
            occ_hat = r_win[idx] / dist_i          # Earth -> Moon direction
            moon_ang_r = np.arcsin(np.clip(R_MOON_KM / dist_i, -1, 1))
            sep = np.arccos(np.clip(np.dot(-occ_hat, sun_win[idx]), -1, 1))
            shadow_offset = side * sep / moon_ang_r     # in Moon-radii units, signed
            panel_img = render_lunar_panel(shadow_offset)
        else:
            # Dedicated solar renderer (real Sun/Moon disc overlap,
            # rasterized with an actual corona at totality/annularity) —
            # its own function, matching how lunar has its own, rather
            # than a shared inline matplotlib-patch fallback.
            r_i = np.linalg.norm(r_win[idx])
            moon_hat = r_win[idx] / r_i
            sun_ang_r = np.arcsin(np.clip(R_SUN_KM / AU_KM, -1, 1))
            sep = np.arccos(np.clip(np.dot(moon_hat, sun_win[idx]), -1, 1))
            moon_offset = side * sep / sun_ang_r        # in Sun-radii units, signed
            is_total_ish = illum_win[idx] < (illum_win.min() + 0.03)
            panel_img = render_solar_panel(moon_offset, corona=is_total_ish and illum_win.min() < 0.85)
        # RGBA with real transparency outside the disk (see
        # eclipse_appearance_strip.py) — panels sit cleanly against the
        # black axes background with a visible gap between them instead
        # of touching opaque black squares.
        ax2.imshow(np.asarray(panel_img), extent=(k+0.02, k+0.98, 0.02, 0.98), zorder=2)
        ax2.text(k+0.5, 0.06, f"{t_hr[idx]:+.1f}h", color="white", ha="center", fontsize=8)
        ax2.text(k+0.5, 0.94, f"{illum_win[idx]:.2f}", color="white", ha="center", fontsize=8)
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_title("Moon appearance through the event" if mode == "lunar"
                 else "Sun appearance through the event", color="black")

    # 3D "view from space" now lives only in eclipse_space_view_plotly.py
    # as interactive HTML, not duplicated here as a static matplotlib panel.

    fig.suptitle(event_label, fontsize=13, y=0.985)

    if mode == "lunar":
        if window.get("event_key") and window.get("observer_name"):
            caption = (
                f"What this is: the real {window['event_label']}, using the greatest-eclipse time "
                f"{window['peak_utc']} UTC. The Moon was above {window['observer_name']} "
                f"at {window['observer_moon_alt_deg']:.1f}° altitude while the Sun was "
                f"{window['observer_sun_alt_deg']:.1f}° below the horizon. How it's made: Astropy's "
                "offline built-in solar-system ephemeris supplies geocentric Moon and Sun vectors at the "
                "actual event epoch, then SSATK's two-circle Sun-disk-overlap illumination model renders "
                "the umbra/penumbra transition at each sampled instant."
            )
        elif window.get("event_key"):
            caption = (
                f"What this is: the real {window['event_label']}, using the greatest-eclipse time "
                f"{window['peak_utc']} UTC from the packaged 21st-century NASA/GSFC eclipse catalog. "
                "How it's made: Astropy's offline built-in solar-system ephemeris supplies geocentric "
                "Moon and Sun vectors at the actual event epoch, then SSATK's two-circle "
                "Sun-disk-overlap illumination model renders the umbra/penumbra transition at each "
                "sampled instant."
            )
        else:
            caption = ("What this is: a real total/partial lunar eclipse, found by searching actual Moon-Earth-Sun geometry over "
                      "time rather than staging one. How it's made: two-body Keplerian propagation of the Moon's real orbit "
                      "(true a/e), real Sun-direction vectors, and the same two-circle Sun-disk-overlap illumination physics "
                      "used for shadow calculations throughout this toolkit. Each Moon panel is rasterized from that same real "
                      "geometry at that instant (a real umbra/penumbra boundary crossing an actual cratered surface), not a "
                      "flat colour swap.")
    else:
        caption = ("What this is: a real total/partial/annular solar eclipse, found by searching actual Moon-Earth-Sun "
                  "geometry over time rather than staging one. How it's made: the same real orbit propagation and "
                  "illumination physics as the lunar case, with occluder and occluded body swapped (Moon casts the "
                  "shadow, Earth's the reference point). Each Sun panel is rasterized from the real Sun/Moon angular "
                  "overlap at that instant, including an actual corona render once coverage is deep enough.")
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9.5, wrap=True,
             transform=fig.transFigure)

    fig.tight_layout(rect=(0.0, 0.11, 1.0, 0.95))
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    stats = dict(mode=mode, eclipse_type=ecl_type, min_illum=float(illum_win.min()),
                 angle_at_peak_deg=float(sep_deg_mid),
                 dur_peak_hr=float(dur_umbra_hr), dur_total_hr=float(dur_total_hr))
    for key in (
        "event_key", "event_label", "event_source", "peak_utc", "observer_name",
        "observer_lat_deg", "observer_lon_deg", "observer_moon_alt_deg", "observer_sun_alt_deg",
    ):
        if key in window and window[key] is not None:
            stats[key] = window[key]
    return fig, stats


def _plot_space_view_unified(ax, moon_r_km, sun_hat, illum, mode, sep_deg=None):
    """Sun, Earth, Moon, and the correct shadow cone direction for whichever
    mode is active (Earth's shadow toward the Moon for lunar; the Moon's
    shadow toward Earth for solar) — same rendering quality throughout."""
    Re = RE_KM
    size_boost = 16.0   # NOT 40 -- at 40x, Earth+Moon's boosted radii
    # summed to ~89% of their real center-to-center distance at eclipse
    # alignment, leaving only a ~11% visual gap: the two bodies looked
    # almost touching even though the Moon is genuinely about 60 Earth-
    # radii away. 16x still boosts both bodies enough to show real
    # surface detail, while leaving a real, honest ~64% gap between them.

    if mode == "lunar":
        earth_pos = np.array([0.0, 0.0, 0.0])
        moon_pos = moon_r_km
        cone_origin, cone_dir, cone_base_r = earth_pos, -sun_hat, Re*size_boost
        cone_len = min(Re / np.tan(np.arcsin((R_SUN_KM-Re)/AU_KM)), Re*80)
    else:
        # moon_r_km here is still "Moon relative to Earth" (r_win), so
        # Earth relative to Moon is its negation — matches the sign
        # convention used in the search above.
        moon_pos = np.array([0.0, 0.0, 0.0])
        earth_pos = -moon_r_km
        cone_origin, cone_dir, cone_base_r = moon_pos, -sun_hat, R_MOON_KM*size_boost
        cone_len = min(R_MOON_KM / np.tan(np.arcsin((R_SUN_KM-R_MOON_KM)/AU_KM)), R_MOON_KM*300)

    _earth_sphere_mpl(ax, earth_pos, Re*size_boost, sun_hat)
    tint = moon_color(illum) if mode == "lunar" else [0.75, 0.75, 0.78]
    _moon_sphere_mpl(ax, moon_pos, R_MOON_KM*size_boost, tint)

    # Shadow cone (tapered outline, top-down-collapsed wedge as before)
    perp = np.array([-cone_dir[1], cone_dir[0], 0.0])
    if np.linalg.norm(perp) < 1e-6:
        perp = np.array([1.0, 0.0, 0.0])
    for sign in (-1, 1):
        edge = np.array([cone_origin + perp*sign*cone_base_r,
                         cone_origin + cone_dir*cone_len])
        ax.plot(edge[:, 0], edge[:, 1], edge[:, 2] if edge.shape[1] > 2 else [0, 0],
               color="#aa4444", alpha=0.7, linewidth=1.5)

    ref_r = np.linalg.norm(moon_r_km)
    sun_len = ref_r * 4.0   # further out than before (was 1.15x) — still
    # nowhere near the real ~390x ratio (1 AU vs Earth-Moon distance),
    # which would put the Sun far outside any usable frame, but enough to
    # read as "clearly further away" rather than sitting right next to
    # the Earth/Moon system.
    sun_center = sun_hat * sun_len + (earth_pos if mode == "solar" else 0)
    sun_radius = ref_r * 0.09
    _sun_sphere_mpl(ax, sun_center, sun_radius)
    ax.text(sun_center[0], sun_center[1], sun_center[2]+sun_radius*1.8,
            "Sun", color="#FFD700", fontsize=9, ha="center", zorder=9)

    lim = sun_len * 1.2
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=18, azim=-50)
    ax.set_xlabel("X [km]"); ax.set_ylabel("Y [km]"); ax.set_zlabel("Z [km]")
    _angle_str = f", angle={sep_deg:.3f}°" if sep_deg is not None else ""
    ax.set_title(f"View from space — illum={illum:.3f}{_angle_str}", fontsize=11)


if __name__ == "__main__":
    from ssatk.plots.figpath import figpath

    fig, stats = find_and_plot_eclipse(mode="lunar", save_path=figpath("eclipse_lunar.png"))
    print(stats)
    fig, stats = find_and_plot_eclipse(mode="solar", save_path=figpath("eclipse_solar.png"))
    print(stats)




def _light_ray_traces(sun_pos, earth_pos, earth_r, moon_pos, moon_r, n_rays=None):
    """
    ONE direct light path along the actual eclipse axis (Sun -> Moon centre
    -> Earth), not a scattered fan of rays in random directions across
    Earth's disk — that fan was confusing rather than illustrative, and
    didn't clearly show the single alignment that causes the eclipse.

    The ray is drawn in two segments: solid yellow from the Sun to where
    it's blocked by the Moon, then a small tapered dark cone from that
    blocking point onward — this is literally "where the light would be
    blocked off by the Moon", not an abstract shadow shape.
    """
    to_earth = earth_pos - sun_pos
    dist = np.linalg.norm(to_earth)
    fwd = to_earth / dist

    # Exact ray-sphere intersection with the Moon along the direct
    # Sun->Earth line (this line passes very close to the Moon's centre
    # by construction, since that's the whole reason an eclipse is
    # happening at this instant).
    oc = sun_pos - moon_pos
    b = np.dot(oc, fwd)
    c = np.dot(oc, oc) - moon_r**2
    disc = b**2 - c
    if disc > 0:
        t_hit = -b - np.sqrt(disc)
        block_point = sun_pos + fwd * max(t_hit, 0)
    else:
        block_point = earth_pos  # (shouldn't happen during a real eclipse)

    traces = [go.Scatter3d(
        x=[sun_pos[0], block_point[0]], y=[sun_pos[1], block_point[1]], z=[sun_pos[2], block_point[2]],
        mode="lines", line=dict(color="#ffe066", width=4),
        hoverinfo="skip", showlegend=False, name="Sunlight",
    )]

    # Small tapered cone from the blocking point onward, showing exactly
    # where the light is cut off — reuses the same real cone-surface
    # builder as the Earth/Moon shadow cones elsewhere in this scene.
    cone_len = min(dist - np.linalg.norm(block_point - sun_pos), moon_r * 6)
    traces.append(_shadow_cone_trace(block_point, fwd, moon_r * 0.5, max(cone_len, moon_r),
                                     color="#552222", opacity=0.35))
    return traces


def _moon_mesh_plotly_REMOVED_use_moon_render_instead():
    """Superseded by moon_render.moon_mesh_plotly (real texture + real
    per-vertex diffuse lighting from bump-mapped normals instead of a
    flat-lit painted albedo). Kept as a stub only so any external code
    still importing the old name gets a clear pointer instead of a
    silent behavior change."""
    raise NotImplementedError("Use moon_render.moon_mesh_plotly instead.")


def _shadow_cone_trace(origin, direction, base_radius, length, color="#aa4444", opacity=0.18,
                       end_radius=0.0):
    """Real tapered 3D cone surface (not a flattened 2D wedge) pointing
    along `direction` from `origin`. Linear radius interpolation from
    `base_radius` at the origin to `end_radius` at `length` away —
    `end_radius=0` (the old default) gives the umbra's converging shape;
    passing an `end_radius` larger than `base_radius` gives the
    penumbra's diverging shape instead, using the exact same builder."""
    zc = np.linspace(0, length, 20)
    thetac = np.linspace(0, 2*np.pi, 24)
    Zc, Tc = np.meshgrid(zc, thetac)
    Rc = base_radius + (end_radius - base_radius) * (Zc/length)
    Xc, Yc = Rc*np.cos(Tc), Rc*np.sin(Tc)

    z_ax = np.array([0.0, 0.0, 1.0])
    d = direction / np.linalg.norm(direction)
    v = np.cross(z_ax, d)
    s = np.linalg.norm(v)
    c = np.dot(z_ax, d)
    if s > 1e-8:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1-c)/(s**2))
    else:
        R = np.eye(3) if c > 0 else -np.eye(3)
    pts = np.stack([Xc.ravel(), Yc.ravel(), Zc.ravel()])
    rot = R @ pts
    X = origin[0] + rot[0].reshape(Xc.shape)
    Y = origin[1] + rot[1].reshape(Xc.shape)
    Z = origin[2] + rot[2].reshape(Xc.shape)

    return go.Surface(x=X, y=Y, z=Z, colorscale=[[0, color], [1, color]],
                      showscale=False, opacity=opacity, hoverinfo="skip",
                      lighting=dict(ambient=1.0, diffuse=0.0), name="Shadow")


def _umbra_penumbra_geometry(R_occ_km, size_boost, ref_dist_km=None, penumbra_ratio=1.35):
    """
    Real umbra + penumbra cone geometry for a shadow-casting body of
    radius R_occ_km, from the actual similar-triangles construction
    (Sun's real angular size determines how fast the shadow narrows) —
    not a calibrated guess. The real physical umbra length is:

        L_umbra = R_occ_km * AU_km / (R_sun_km - R_occ_km)

    which gives ~1.385 million km for Earth's shadow (this is the real,
    textbook number — and it comfortably exceeds the Earth-Moon distance,
    which is exactly why lunar eclipses can go total) and ~374,600 km for
    the Moon's shadow (which is *shorter* than the Earth-Moon distance
    most of the time — the real reason annular eclipses happen at all:
    the Moon's umbra apex falls short of Earth's surface, so only the
    surrounding antumbra reaches the ground).

    Using this real (unboosted) length together with the body's boosted
    display radius as the cone's base automatically reproduces the
    correct real-world ratio at the real Earth-Moon distance — no
    separate calibration step needed, and critically, no risk of
    accidentally applying one direction's ratio to the other direction's
    cone (a real bug in an earlier version of this function: it used a
    single 2.6x-Moon-radius target for both the Earth-shadow-on-Moon
    and Moon-shadow-on-Earth cases, even though those are two physically
    different cones with different real apex distances).

    `ref_dist_km` is accepted but unused — kept only so existing callers
    don't need updating.
    """
    L_umbra_real = R_occ_km * AU_KM / (R_SUN_KM - R_occ_km)
    base_r = R_occ_km * size_boost
    slope = base_r / L_umbra_real
    umbra_len = L_umbra_real
    pen_base_r = base_r
    pen_len = umbra_len
    pen_end_r = base_r + slope * penumbra_ratio * pen_len
    return base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope





def _shadow_ground_point(moon_pos, sun_hat, earth_pos, earth_r_real=None):
    """
    Where the real Sun->Moon shadow axis hits Earth's surface — the
    actual sub-shadow point that defines the path of totality/
    annularity, not just "the two bodies are roughly aligned". Same
    ray-sphere intersection math as _light_ray_traces' Sun->Moon check,
    aimed at Earth instead: ray origin = Moon center, direction = away
    from the Sun (-sun_hat), sphere = Earth at its REAL radius (the path
    is a physical ground location, computed before any display-scale
    boost is applied).

    Returns the real (unboosted) hit point in km, or None if the shadow
    axis misses Earth entirely at this instant (normal outside totality
    — most of a partial eclipse's duration, the umbra hasn't reached
    Earth's surface at all).
    """
    try:
        from .globe_orbit_daynight_plotly import RE_KM as _RE_KM
    except ImportError:
        from globe_orbit_daynight_plotly import RE_KM as _RE_KM
    R = earth_r_real if earth_r_real is not None else _RE_KM
    d = -sun_hat / np.linalg.norm(sun_hat)
    oc = moon_pos - earth_pos
    b = np.dot(oc, d)
    c = np.dot(oc, oc) - R**2
    disc = b**2 - c
    if disc < 0:
        return None
    t_hit = -b - np.sqrt(disc)
    if t_hit < 0:
        return None
    return moon_pos + d * t_hit


def _starfield_trace(radius, n_stars=500, seed=7):
    """
    A static background starfield. Not recomputed per frame — real stars
    are so far away that their apparent positions genuinely don't shift
    over a multi-hour eclipse window, so keeping this fixed across every
    frame is the physically correct choice, not a shortcut. Random
    brightness/size per star for a natural, non-uniform look, placed just
    inside the scene's own axis range so they're actually visible instead
    of being clipped by the axis boundary.
    """
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n_stars, 3))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pts = vecs * radius * 0.97
    brightness = rng.uniform(0.35, 1.0, n_stars)
    sizes = rng.uniform(1.0, 2.6, n_stars)
    colors = [f"rgba(255,255,255,{b:.2f})" for b in brightness]
    return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                        marker=dict(size=sizes, color=colors), hoverinfo="skip",
                        showlegend=False, name="Stars")


def _lunar_or_solar_camera_eye(mode, sun_hat):
    """
    Real bug, not a style choice: lunar and solar eclipses need DIFFERENT
    camera directions, and using the same sun-relative camera for both
    was hiding the Moon entirely in lunar mode.

    Solar mode: the umbra/antumbra always lands on the sun-facing side of
    Earth, so the camera should look from roughly the Sun's direction —
    otherwise the shadow and ground-track path render correctly but on
    the far side of the globe from wherever the camera happens to point.

    Lunar mode: the Moon sits in OPPOSITION to the Sun (verified
    directly: 179.84 deg between Moon and Sun direction at real peak
    alignment, not 0). Aiming the camera toward the Sun's direction here
    puts the Moon almost exactly behind Earth from the camera's point of
    view — not just small, actually hidden. A side-on view, roughly
    perpendicular to the Sun direction, shows Earth and the Moon next to
    each other instead of one hiding behind the other.
    """
    if mode == "lunar":
        up_ref = np.array([0.0, 0.0, 1.0]) if abs(sun_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side_hat = np.cross(sun_hat, up_ref)
        side_hat = side_hat / np.linalg.norm(side_hat)
        return dict(x=side_hat[0]*0.6, y=side_hat[1]*0.6, z=side_hat[2]*0.6 + 0.25)
    else:
        return dict(x=sun_hat[0]*0.55, y=sun_hat[1]*0.55+0.15, z=sun_hat[2]*0.55+0.2)


def _shadow_footprint_traces(earth_pos, hit_point_display, sun_hat, footprint_r_umbra, footprint_r_penumbra):
    """
    A real dark patch drawn directly on Earth's surface at the current
    ground-track point — this is the actual visible "there is a shadow
    on the Earth" cue that a single thin light-ray line can't provide.
    Built as a small flat disc in the local tangent plane at the hit
    point (a fine approximation at these footprint scales, which are
    small compared to Earth's radius) then nudged just above the
    surface so it doesn't z-fight with the Earth mesh underneath.

    Two nested discs: a wider, lighter penumbra footprint (partial
    shadow — where a solar eclipse would be seen as partial), and a
    smaller, darker umbra/antumbra footprint inside it (where totality
    or an annular eclipse would actually be visible) — same "layers"
    concept requested for the lunar-eclipse cones, applied on the
    ground instead of in space.
    """
    n_hat = hit_point_display - earth_pos
    n_hat = n_hat / np.linalg.norm(n_hat)
    ref = np.array([0.0, 0.0, 1.0]) if abs(n_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, n_hat); u /= np.linalg.norm(u)
    v = np.cross(n_hat, u)
    theta = np.linspace(0, 2*np.pi, 40)

    def _disc(radius, color, opacity):
        pts = (hit_point_display[None, :] + n_hat[None, :] * (radius * 0.002)
              + radius * (np.cos(theta)[:, None]*u[None, :] + np.sin(theta)[:, None]*v[None, :]))
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        n = len(theta)
        xs = np.concatenate([[hit_point_display[0]], x])
        ys = np.concatenate([[hit_point_display[1]], y])
        zs = np.concatenate([[hit_point_display[2]], z])
        i = [0] * (n - 1)
        j = list(range(1, n))
        k = list(range(2, n)) + [1]
        return go.Mesh3d(x=xs, y=ys, z=zs, i=i, j=j, k=k,
                         color=color, opacity=opacity, hoverinfo="skip", showlegend=False,
                         lighting=dict(ambient=1.0, diffuse=0.0), name="Shadow footprint")

    return [_disc(footprint_r_penumbra, "#3a3a55", 0.30),
           _disc(footprint_r_umbra, "#05050a", 0.55)]


def _sun_direction_arrow(origin, sun_hat, length, label="Sun direction"):
    """
    A directional vector pointing toward the Sun, replacing a literal
    small Sun sphere placed nearby. At real solar-system distances the
    Sun cannot be rendered to any consistent scale next to Earth/Moon —
    either it's a speck (uninformative) or, as before, an arbitrarily
    up-scaled sphere close enough to read as a size/distance, which
    visually (and misleadingly) reads as "a second, smaller moon" rather
    than "a star 150 million km away". A labeled arrow says exactly what
    it means — a direction, not a to-scale object — without implying a
    false distance or size.
    """
    shaft_end = origin + sun_hat * length * 0.88
    tip = origin + sun_hat * length
    shaft = go.Scatter3d(x=[origin[0], shaft_end[0]], y=[origin[1], shaft_end[1]], z=[origin[2], shaft_end[2]],
                         mode="lines", line=dict(color="#ffd34d", width=6),
                         hoverinfo="skip", showlegend=False, name=label)
    head = _shadow_cone_trace(shaft_end, sun_hat, length*0.035, length*0.12,
                              color="#ffd34d", opacity=0.9)
    text = go.Scatter3d(x=[tip[0]], y=[tip[1]], z=[tip[2]], mode="text",
                        text=[label], textfont=dict(color="#ffd34d", size=18),
                        hoverinfo="skip", showlegend=False)
    return [shaft, head, text]

def _time_grid_utc(start, stop, n_frames):
    """Return an Astropy Time array spanning two UTC-like endpoints."""
    from astropy.time import Time
    start_t = Time(start, scale="utc") if not hasattr(start, "gps") else start
    stop_t = Time(stop, scale="utc") if not hasattr(stop, "gps") else stop
    gps = np.linspace(float(start_t.gps), float(stop_t.gps), int(n_frames))
    return Time(gps, format="gps", scale="utc")


def _body_xyz_km(name, times, ephemeris="builtin"):
    """Return barycentric body coordinates as an ``(N, 3)`` km array."""
    from astropy import units as u
    from astropy.coordinates import get_body_barycentric, solar_system_ephemeris
    with solar_system_ephemeris.set(ephemeris):
        xyz = get_body_barycentric(name, times).xyz.to_value(u.km)
    arr = np.asarray(xyz, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, 3)
    return np.moveaxis(arr, 0, -1).reshape(-1, 3)


def _real_solar_eclipse_geometry(times, ephemeris="builtin"):
    """Return real Earth-relative Moon positions and Earth-to-Sun directions."""
    earth = _body_xyz_km("earth", times, ephemeris=ephemeris)
    moon = _body_xyz_km("moon", times, ephemeris=ephemeris)
    sun = _body_xyz_km("sun", times, ephemeris=ephemeris)
    moon_pos = moon - earth
    sun_vec = sun - earth
    sun_hat = sun_vec / np.linalg.norm(sun_vec, axis=1, keepdims=True)
    return moon_pos, sun_hat


def _wgs84_radius_along(u_itrf):
    """Distance (km) from Earth's centre to the WGS84 surface along unit ITRF direction(s)."""
    from ssatk.constants import WGS84_A_KM, WGS84_B_KM

    a_km = WGS84_A_KM
    b2_over_a2 = (WGS84_B_KM / WGS84_A_KM) ** 2
    u = np.asarray(u_itrf, dtype=float)
    return a_km / np.sqrt(u[..., 0] ** 2 + u[..., 1] ** 2 + u[..., 2] ** 2 / b2_over_a2)


def _eci_surface_to_latlon(point_km, time):
    """WGS84 geodetic (lat, lon) in degrees of the Earth-surface point along a GCRF direction.

    The GCRF direction is rotated to ITRF with the chain SSAPy's groundTrack
    uses, projected onto the WGS84 ellipsoid, and converted to geodetic
    coordinates. Previously the rotation was GAST about GCRF +z (1242 arcsec,
    38.4 km at the equator, off in 2026) and the latitude was geocentric
    (up to 0.19 deg, 21 km, off).
    """
    import erfa
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    u = eci_to_ecf_matrix(_gps_seconds_at(time)) @ np.asarray(point_km, dtype=float)
    u = u / np.linalg.norm(u)
    surface_m = u * _wgs84_radius_along(u) * 1000.0
    lon, lat, _height = erfa.gc2gd(1, surface_m)
    return float(np.degrees(lat)), float(np.degrees(lon))


def _latlon_to_eci_surface(lat_deg, lon_deg, time, radius_scale=1.0, radius_km=RE_KM):
    """Place a fixed Earth geodetic lat/lon marker in the inertial (GCRF) scene frame.

    The marker sits on the scene's spherical Earth (``radius_km * radius_scale``)
    in the GCRF direction of the WGS84 surface point, so it is the exact
    inverse of :func:`_eci_surface_to_latlon`.
    """
    import erfa
    from ssatk.coordinates.frames import eci_to_ecf_matrix

    itrf_m = erfa.gd2gc(1, np.radians(float(lon_deg)), np.radians(float(lat_deg)), 0.0)
    gcrf = eci_to_ecf_matrix(_gps_seconds_at(time)).T @ itrf_m
    return float(radius_km) * float(radius_scale) * gcrf / np.linalg.norm(gcrf)


def _gps_seconds_at(time):
    gps = getattr(time, "gps", None)
    if gps is not None:
        return float(np.asarray(gps, dtype=float).reshape(-1)[0])
    return float(np.asarray(time, dtype=float).reshape(-1)[0])


def _ground_latlon_path_to_eci(ground_latlon, time, radius_scale=1.0, radius_km=RE_KM):
    """Project a ground-fixed lat/lon path onto Earth's currently rotated mesh."""
    points = [
        _latlon_to_eci_surface(lat, lon, time, radius_scale=radius_scale, radius_km=radius_km)
        for lat, lon in ground_latlon
        if lat is not None and lon is not None
    ]
    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def _great_circle_distance_km(lat1, lon1, lat2, lon2, radius_km=EARTH_MEAN_RADIUS_KM):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return float(radius_km * 2.0 * np.arcsin(np.sqrt(a)))


def plot_2024_solar_eclipse_animated(
    save_path=None,
    *,
    start="2024-04-08T17:30:00",
    stop="2024-04-08T20:00:00",
    n_frames=30,
    n_lat=144,
    n_lon=288,
    ephemeris="builtin",
    target_lat=30.3630,
    target_lon=-97.9790,
    target_label="NW Travis County, TX",
    show_stars=True,
    verbose=True,
):
    """Animate the April 8, 2024 total solar eclipse over central Texas.

    This uses Astropy's real Sun/Earth/Moon ephemerides rather than the generic
    fixed-node eclipse search used by the teaching lunar demo.  The default
    target marker is placed near northwest Travis County/Lakeway, where the
    totality path clipped the Austin/Travis County area.
    """
    times = _time_grid_utc(start, stop, n_frames)
    moon_pos, sun_hat = _real_solar_eclipse_geometry(times, ephemeris=ephemeris)

    size_boost = 16.0
    base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope = _umbra_penumbra_geometry(
        R_MOON_KM,
        size_boost,
        D_MOON_A_KM,
    )

    hits_real = []
    ground_latlon = []
    footprint_radii = []
    target_distances = []
    for time, moon_i, sun_i in zip(times, moon_pos, sun_hat):
        hit = _shadow_ground_point(moon_i, sun_i, np.zeros(3), earth_r_real=RE_KM)
        hits_real.append(hit)
        if hit is None:
            ground_latlon.append(None)
            footprint_radii.append((0.0, 0.0))
            target_distances.append(np.inf)
            continue
        lat, lon = _eci_surface_to_latlon(hit, time)
        dist_from_moon = np.linalg.norm(hit - moon_i)
        r_umbra = abs(base_r - slope * dist_from_moon)
        r_pen = pen_base_r + slope * 1.35 * dist_from_moon
        ground_latlon.append((lat, lon))
        footprint_radii.append((r_umbra, r_pen))
        target_distances.append(_great_circle_distance_km(lat, lon, target_lat, target_lon))

    finite_dist = np.asarray(target_distances, dtype=float)
    closest_idx = int(np.nanargmin(finite_dist)) if np.isfinite(finite_dist).any() else 0
    mid_idx = closest_idx
    moon_mid = moon_pos[mid_idx]
    sun_mid = sun_hat[mid_idx]
    ref_r = float(np.linalg.norm(moon_mid))
    lim = (ref_r + max(RE_KM, R_MOON_KM) * size_boost) * 1.35
    static_traces = [_starfield_trace(lim)] if show_stars else []

    def _empty_shadow_footprint():
        return [go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False),
                go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False)]

    def _path_traces(k):
        all_latlon = [latlon for latlon in ground_latlon if latlon is not None]
        elapsed_latlon = [latlon for latlon in ground_latlon[:k + 1] if latlon is not None]
        all_pts = _ground_latlon_path_to_eci(all_latlon, times[k], radius_scale=size_boost * 1.006)
        elapsed_pts = _ground_latlon_path_to_eci(elapsed_latlon, times[k], radius_scale=size_boost * 1.009)
        if len(all_pts):
            full_path_trace = go.Scatter3d(
                x=all_pts[:, 0], y=all_pts[:, 1], z=all_pts[:, 2], mode="lines",
                line=dict(color="#ff6b5a", width=4), opacity=0.45,
                name="2024 full totality path", hoverinfo="skip", showlegend=False,
            )
        else:
            full_path_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False)

        if len(elapsed_pts):
            elapsed_path_trace = go.Scatter3d(
                x=elapsed_pts[:, 0], y=elapsed_pts[:, 1], z=elapsed_pts[:, 2], mode="lines",
                line=dict(color="#ff3b1f", width=8),
                name="2024 elapsed totality path", hoverinfo="skip", showlegend=False,
            )
        else:
            elapsed_path_trace = go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False)

        if ground_latlon[k] is not None:
            point = _latlon_to_eci_surface(*ground_latlon[k], times[k], radius_scale=size_boost * 1.014)
            marker_trace = go.Scatter3d(
                x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
                marker=dict(color="#ffcc00", size=6, line=dict(color="#ff3b1f", width=1)),
                name="Current shadow center", hoverinfo="skip", showlegend=False,
            )
            r_umbra, r_pen = footprint_radii[k]
            footprint = _shadow_footprint_traces(np.zeros(3), point, sun_hat[k], r_umbra, r_pen)
        else:
            marker_trace = go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip", showlegend=False)
            footprint = _empty_shadow_footprint()
        return [full_path_trace, elapsed_path_trace, marker_trace] + footprint

    def _target_trace(k):
        point = _latlon_to_eci_surface(target_lat, target_lon, times[k], radius_scale=size_boost * 1.002)
        return go.Scatter3d(
            x=[point[0]], y=[point[1]], z=[point[2]], mode="markers",
            marker=dict(color="#7CFC00", size=5, symbol="diamond", line=dict(color="white", width=1)),
            name=target_label,
            hovertemplate=(
                f"{target_label}<br>lat={target_lat:.3f}° lon={target_lon:.3f}°"
                "<br>shadow-center distance=%{customdata:.0f} km<extra></extra>"
            ),
            customdata=[target_distances[k]],
            showlegend=True,
        )

    def _frame_traces(k):
        moon_i = moon_pos[k]
        sun_i = sun_hat[k]
        earth_pos = np.zeros(3)
        cone_origin = moon_i + (-sun_i) * (R_MOON_KM * size_boost * 0.03)
        rotation_deg = earth_rotation_deg_from_time(times[k])
        traces = [
            _earth_mesh(sun_i, n_lat=n_lat, n_lon=n_lon, radius_scale=size_boost,
                        center=tuple(earth_pos), rotation_deg=rotation_deg,
                        shadow_body_center_km=moon_i, shadow_body_radius_km=R_MOON_KM),
            _earth_atmosphere_trace(center=tuple(earth_pos), radius_scale=size_boost),
            moon_mesh_plotly(moon_i, R_MOON_KM * size_boost, sun_hat=sun_i,
                             real_center_km=np.zeros(3), mode="solar",
                             n_lat=max(100, n_lat), n_lon=max(200, n_lon)),
            _shadow_cone_trace(cone_origin, -sun_i, base_r, umbra_len,
                               color="#1a0a0a", opacity=0.12, end_radius=0.0),
            _shadow_cone_trace(cone_origin, -sun_i, pen_base_r, pen_len,
                               color="#4a5a7a", opacity=0.045, end_radius=pen_end_r),
            *_sun_direction_arrow(earth_pos + sun_i * RE_KM * size_boost * 1.15, sun_i, ref_r * 0.55),
            *_path_traces(k),
            _target_trace(k),
        ]
        return traces

    initial_traces = _frame_traces(0)
    dynamic_indices = list(range(len(static_traces), len(static_traces) + len(initial_traces)))
    frames = []
    for k, time in enumerate(times):
        latlon = ground_latlon[k]
        if latlon is None:
            subtitle = "shadow axis off Earth"
        else:
            subtitle = (f"shadow center {latlon[0]:.2f}°, {latlon[1]:.2f}°; "
                        f"{target_distances[k]:.0f} km from {target_label}")
        frames.append(go.Frame(
            data=_frame_traces(k),
            traces=dynamic_indices,
            name=str(k),
            layout=go.Layout(title=dict(text=f"2024 total solar eclipse — {time.utc.iso}<br><sub>{subtitle}</sub>")),
        ))

    fig = go.Figure(data=static_traces + initial_traces, frames=frames)
    camera_eye = _lunar_or_solar_camera_eye("solar", sun_mid)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title=dict(text="X [km]", font=dict(size=18, color="white")),
                       tickfont=dict(size=14, color="white"), backgroundcolor="black", gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title=dict(text="Y [km]", font=dict(size=18, color="white")),
                       tickfont=dict(size=14, color="white"), backgroundcolor="black", gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title=dict(text="Z [km]", font=dict(size=18, color="white")),
                       tickfont=dict(size=14, color="white"), backgroundcolor="black", gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=camera_eye),
        ),
        paper_bgcolor="black",
        font=dict(color="white", size=18),
        title=dict(
            text=(f"2024 total solar eclipse — {times[0].utc.iso}<br>"
                  f"<sub>green marker: {target_label}; red/yellow: Moon shadow center</sub>"),
            x=0.5,
            font=dict(color="white", size=24),
        ),
        margin=dict(l=0, r=0, t=92, b=0),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.45)", font=dict(color="white", size=12)),
        updatemenus=[dict(
            type="buttons", showactive=False, y=0.02, x=0.5, xanchor="center",
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True), fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            steps=[dict(method="animate", args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))],
                        label=times[k].utc.iso[11:16]) for k in range(len(times))],
            transition=dict(duration=0), x=0.05, len=0.9, y=0.0,
        )],
    )

    stats = dict(
        event="2024-04-08 total solar eclipse",
        start_utc=times[0].utc.iso,
        stop_utc=times[-1].utc.iso,
        closest_time_utc=times[closest_idx].utc.iso,
        closest_shadow_latlon=ground_latlon[closest_idx],
        closest_target_distance_km=float(target_distances[closest_idx]),
        target_lat=float(target_lat),
        target_lon=float(target_lon),
        target_label=target_label,
        ephemeris=ephemeris,
    )
    if verbose:
        latlon = stats["closest_shadow_latlon"]
        print(f"[2024 solar] closest shadow center to {target_label}: "
              f"{stats['closest_target_distance_km']:.1f} km at {stats['closest_time_utc']} "
              f"(lat/lon={latlon[0]:.3f}, {latlon[1]:.3f})")
    if save_path:
        if str(save_path).endswith(".html"):
            fig.write_html(save_path)
            _inject_eclipse_catalog_dropdown(save_path, default_mode="solar", default_event="2024-04-08")
        else:
            fig.write_image(save_path)
        print(f"Saved -> {save_path}")
    return fig, stats


def plot_space_view_animated(mode="lunar", search_days=None, n_frames=26,
                             save_path=None, n_lat=60, n_lon=120, verbose=True,
                             show_eclipse_path=True, event=None):
    """
    Animated version of plot_space_view_plotly: steps through the real
    eclipse time window (the same fine-resolution window eclipse_demo.py
    searches and refines) instead of showing one frozen instant. Two real
    motions play simultaneously, exactly as they would in reality:

      1. Earth's own sidereal rotation (fast — ~15 deg/hour), animated by
         rolling _earth_mesh's body-fixed fields via its `rotation_deg`
         parameter rather than recomputing the land mask / relief from
         scratch every frame (recomputing at n_frames >~ 20 would be the
         slow part otherwise).
      2. The Moon's real orbital motion through r_win (already the exact
         same array eclipse_demo.py's refined search produces) — this is
         what actually sweeps the eclipse shadow across the disk.

    The Sun's direction (sun_win) also technically changes over the
    window, but negligibly over a few hours, so its motion is included
    for correctness at essentially no visual cost.

    Mesh resolution is reduced by default (90x180 / Moon proportionally)
    since this rebuilds full Earth+Moon meshes per frame — n_frames=28 at
    full 180x360 resolution would be quite slow to generate.
    """
    assert mode in ("lunar", "solar")
    window = _eclipse_window(mode, search_days=search_days, event=event, n_steps=4000, verbose=verbose)
    t_fine = window["t_s"]
    r_fine = window["r_moon"]
    sun_fine = window["sun_hat"]
    illum_fine = window["illum"]
    mid = window["peak_idx"]

    # Window of interest around the peak — a bit wider than the peak
    # phase itself so the animation shows the approach and departure, not
    # just the deepest instant.
    half_window_hr = 3.0 if mode == "lunar" else 1.5
    in_window = np.abs(t_fine - t_fine[mid]) < half_window_hr * 3600
    idxs_all = np.where(in_window)[0]
    frame_idxs = idxs_all[np.linspace(0, len(idxs_all)-1, n_frames).astype(int)]

    if verbose:
        print(f"[{mode}] Animating {n_frames} frames over "
              f"+/-{half_window_hr:.1f} hr around the peak "
              f"(illum min={illum_fine[mid]:.4f})")

    # 16x, not 40x -- at 40x the boosted Earth+Moon radii summed to ~89%
    # of their real center-to-center distance at eclipse alignment,
    # leaving only ~11% of real space between them (they looked almost
    # touching even though the Moon is genuinely ~60 Earth-radii away).
    # 16x still shows real surface detail while leaving an honest ~64% gap.
    size_boost = 16.0
    ref_dist_km = D_MOON_A_KM  # shared calibration distance for both
    # directions of shadow (Earth's shadow reaching the Moon, or the
    # Moon's shadow reaching Earth) — this is the real Earth-Moon
    # semi-major axis, the physically relevant "other body" distance in
    # both cases.

    def _frame_traces(idx):
        moon_r_km = r_fine[idx]
        sun_hat = sun_fine[idx]
        illum = illum_fine[idx]
        rotation_deg = _earth_rotation_for_window_sample(window, idx)

        # Earth-fixed frame for BOTH modes — Earth stays at the origin,
        # the Moon orbits around it. The previous solar-mode version used
        # a Moon-fixed frame (Earth orbiting the Moon) — physically
        # equivalent, but visually the Moon just sat still at the origin
        # every frame while Earth drifted, which looks wrong in an
        # animation: the Moon is the thing actually sweeping between Sun
        # and Earth in reality, and should visibly do so here too.
        earth_pos = np.array([0.0, 0.0, 0.0])
        moon_pos = moon_r_km
        occ_r_km = RE_KM if mode == "lunar" else R_MOON_KM
        cone_origin = earth_pos if mode == "lunar" else moon_pos
        cone_dir = -sun_hat
        # Nudge the cone's start point a couple percent further along its
        # own direction, burying its near cross-section just inside the
        # opaque body instead of exactly on its surface — at exact
        # coincidence, the cone's semi-transparent near-face and the
        # sphere's silhouette edge overlap almost exactly from most
        # camera angles, rendering as a visible dark ring right at the
        # Moon's edge that isn't a real shadow feature.
        cone_origin = cone_origin + cone_dir * (occ_r_km * size_boost * 0.03)
        shadow_kwargs = ({} if mode == "lunar"
                         else dict(shadow_body_center_km=moon_pos, shadow_body_radius_km=R_MOON_KM))

        traces = [
            _earth_mesh(sun_hat, n_lat=n_lat, n_lon=n_lon, radius_scale=size_boost,
                       center=tuple(earth_pos), rotation_deg=rotation_deg, **shadow_kwargs),
            _earth_atmosphere_trace(center=tuple(earth_pos), radius_scale=size_boost),
        ]
        real_moon_center = moon_r_km if mode == "lunar" else np.array([0.0, 0.0, 0.0])
        traces.append(moon_mesh_plotly(moon_pos, R_MOON_KM*size_boost, sun_hat=sun_hat,
                                       real_center_km=real_moon_center, mode=mode,
                                       n_lat=max(140, n_lat), n_lon=max(280, n_lon)))

        # Real umbra (converging, dark) + penumbra (diverging, lighter)
        # cone pair, calibrated so their cross-section at the real
        # Earth-Moon distance matches the actual physical umbra/penumbra
        # size there — this is what was missing entirely for lunar mode
        # ("no layers for total or partial") and what was clipping
        # straight through the Moon for solar mode (the old single cone's
        # base radius was sized from the boosted OCCLUDER radius with no
        # calibration against the other body's boosted size, so the two
        # boosted objects ended up at incompatible scales at the distance
        # that actually matters).
        base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope = _umbra_penumbra_geometry(
            occ_r_km, size_boost, ref_dist_km)
        traces.append(_shadow_cone_trace(cone_origin, cone_dir, base_r, umbra_len,
                                         color="#1a0a0a", opacity=0.12, end_radius=0.0))
        traces.append(_shadow_cone_trace(cone_origin, cone_dir, pen_base_r, pen_len,
                                         color="#4a5a7a", opacity=0.045, end_radius=pen_end_r))

        # Directional Sun vector instead of a small sphere sitting nearby
        # (see _sun_direction_arrow docstring) — anchored just outside
        # Earth/Moon's boosted extent so it doesn't overlap either body.
        ref_r = np.linalg.norm(r_fine[mid])
        arrow_len = ref_r * 0.55  # short enough not to force the whole
        # scene to zoom out to fit it — a vector only needs to be long
        # enough to clearly indicate a direction, not to reach any
        # particular distance
        arrow_origin = earth_pos + sun_hat * (RE_KM*size_boost*1.15 if mode == "lunar"
                                              else np.linalg.norm(moon_pos) + R_MOON_KM*size_boost*1.15)
        traces.extend(_sun_direction_arrow(arrow_origin, sun_hat, arrow_len))

        # Axis range is set from Earth/Moon's own extent (see `lim` below),
        # NOT from the Sun arrow — the arrow used to be long enough that
        # fitting it in frame forced a ~3-million-km-wide view, at which
        # scale the Moon's real ~10,000 km shift over the animation window
        # was under 1% of the frame and effectively invisible, even though
        # it was being computed correctly the whole time.
        sun_len = ref_r  # kept only for the return signature below
        return traces, illum, sun_len, occ_r_km, base_r, umbra_len, slope

    # Real ground-track of the shadow axis hitting Earth's surface — the
    # actual path of totality/annularity, computed once per frame index
    # up front (cheap — just a ray-sphere intersection) so the animation
    # loop below can draw the accumulated path so far at each frame
    # rather than only ever showing the current instant.
    ground_track_display = []
    ground_footprint_r = []  # (umbra_r, penumbra_r) at the hit point, display units
    if mode == "solar" and show_eclipse_path:
        base_r0, umbra_len0, pen_base_r0, pen_len0, pen_end_r0, slope0 = _umbra_penumbra_geometry(
            R_MOON_KM, size_boost, ref_dist_km)

        # First pass: real, uncompressed hit points (or None).
        raw_hits = []
        for idx in frame_idxs:
            moon_pos_i = r_fine[idx]
            sun_hat_i = sun_fine[idx]
            earth_pos_i = np.array([0.0, 0.0, 0.0])
            # A raw ray-sphere intersection test alone isn't enough here:
            # Earth is a large target (6378 km radius) relative to how
            # tightly the umbra/antumbra actually has to align for a real
            # eclipse, so the geometric shadow AXIS can keep "hitting"
            # Earth's sphere well outside the real event window — found
            # this directly: every single frame showed a hit, including
            # ones at illum=1.000 (no eclipse effect at all, Sun fully
            # visible from Earth's own center). Gate on the real
            # illumination value too, so the shadow only appears when a
            # real eclipse effect is actually present.
            no_real_eclipse = illum_fine[idx] > 0.999
            hit = None if no_real_eclipse else _shadow_ground_point(
                moon_pos_i, sun_hat_i, earth_pos_i, earth_r_real=RE_KM)
            raw_hits.append(hit)

        for k, idx in enumerate(frame_idxs):
            hit = raw_hits[k]
            earth_pos_i = np.array([0.0, 0.0, 0.0])
            moon_pos_i = r_fine[idx]
            if hit is None:
                ground_track_display.append(None)
                ground_footprint_r.append((0.0, 0.0))
            else:
                unit = (hit - earth_pos_i)
                unit = unit / np.linalg.norm(unit)
                # NO compression here (an earlier version of this tried
                # compressing the angular deviation toward the peak
                # direction, first with a linear blend, then with proper
                # slerp) — that whole approach was based on a flawed
                # comparison. The shadow's real angular sweep across
                # Earth's surface (tens of degrees over the active
                # window — checked directly: ~33 deg just 40 samples from
                # peak, so wider across the full window) and the Moon's
                # real 3D position shift (small relative to its vast
                # distance from Earth) are two different physical
                # quantities that were never going to match by
                # comparing their raw magnitudes. The shadow SHOULD sweep
                # a visibly large fraction of Earth's disk — that's
                # genuinely how a real eclipse's ground track looks over
                # a couple of hours — while the Moon's own position
                # barely shifts on screen simply because the Earth-Moon
                # distance dwarfs how far it moves in that time. Forcing
                # them to match was the actual mistake.
                ground_track_display.append(earth_pos_i + unit * RE_KM * size_boost)
                dist_from_moon = np.linalg.norm(hit - moon_pos_i)
                # Signed radius from the linear taper — once dist_from_moon
                # exceeds the umbra's real apex distance, this goes
                # negative and represents the antumbra re-widening past
                # the apex (the real shape during an annular eclipse,
                # where the umbra falls short of Earth and only the
                # surrounding antumbra reaches the ground) rather than
                # "no shadow at all".
                r_umbra = abs(base_r0 - slope0*dist_from_moon)
                r_pen = pen_base_r0 + slope0*1.35*dist_from_moon
                ground_footprint_r.append((r_umbra, r_pen))
    else:
        ground_track_display = [None] * len(frame_idxs)
        ground_footprint_r = [(0.0, 0.0)] * len(frame_idxs)

    def _path_traces(k):
        """Growing red path line (every hit point up to and including
        frame k) plus a bright marker and a real shadow footprint disc
        at the current shadow location — always exactly 4 traces, even
        when empty, so every frame has the same trace count/order
        (Plotly assumes this when scrubbing the slider out of order)."""
        pts = [p for p in ground_track_display[:k+1] if p is not None]
        if pts:
            arr = np.array(pts)
            line_tr = go.Scatter3d(x=arr[:, 0], y=arr[:, 1], z=arr[:, 2], mode="lines",
                                   line=dict(color="#ff3b1f", width=7),
                                   name="Path of eclipse", hoverinfo="skip", showlegend=False)
        else:
            line_tr = go.Scatter3d(x=[], y=[], z=[], mode="lines", hoverinfo="skip", showlegend=False)

        # The marker specifically must reflect THIS frame's real hit, not
        # just the last point anywhere in the accumulated path — using
        # arr[-1] unconditionally meant the marker kept sitting at the
        # last known position indefinitely even after the real eclipse
        # ended (ground_track_display[k] is None there), which visually
        # implied an active shadow that wasn't actually there anymore.
        if ground_track_display[k] is not None:
            p = ground_track_display[k]
            marker_tr = go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
                                     marker=dict(color="#ffcc00", size=6, line=dict(color="#ff3b1f", width=1)),
                                     name="Current shadow center", hoverinfo="skip", showlegend=False)
        else:
            marker_tr = go.Scatter3d(x=[], y=[], z=[], mode="markers", hoverinfo="skip", showlegend=False)

        if ground_track_display[k] is not None:
            r_umbra, r_pen = ground_footprint_r[k]
            footprint = _shadow_footprint_traces(np.array([0.0, 0.0, 0.0]), ground_track_display[k],
                                                 sun_fine[frame_idxs[k]], r_umbra, r_pen)
        else:
            footprint = [go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False),
                        go.Mesh3d(x=[], y=[], z=[], i=[], j=[], k=[], hoverinfo="skip", showlegend=False)]
        return [line_tr, marker_tr] + footprint

    ref_r_lim = np.linalg.norm(r_fine[mid])
    lim = (ref_r_lim + max(RE_KM, R_MOON_KM)*size_boost) * 1.35
    static_traces = [_starfield_trace(lim)]

    first_traces, first_illum, sun_len, *_ = _frame_traces(frame_idxs[0])
    if mode == "solar" and show_eclipse_path:
        first_traces = first_traces + _path_traces(0)
    dynamic_indices = list(range(len(static_traces), len(static_traces) + len(first_traces)))
    fig = go.Figure(data=static_traces + first_traces)

    frames = []
    for k, idx in enumerate(frame_idxs):
        traces, illum, *_ = _frame_traces(idx)
        if mode == "solar" and show_eclipse_path:
            traces = traces + _path_traces(k)
        t_rel_hr = (t_fine[idx] - t_fine[mid]) / 3600.0
        frames.append(go.Frame(
            data=traces, name=str(k),
            traces=dynamic_indices,
            layout=go.Layout(title=dict(
                text=f"{_window_title_prefix(mode, window)} — animated<br>"
                    f"<sub>t={t_rel_hr:+.2f} hr, illum={illum:.3f}</sub>",
                x=0.5, font=dict(color="white", size=15))),
        ))
    fig.frames = frames

    sun_hat_mid = sun_fine[mid]
    # Solar mode: the umbra/antumbra always lands on the sun-facing side
    # of Earth, by definition — a generic fixed camera angle has no
    # reason to be looking at that hemisphere, which is exactly why the
    # shadow dimming and the ground-track path looked absent: they were
    # being computed correctly but rendered on the far side of the globe
    # from wherever the camera happened to be pointed. Same fix the
    # static plot_space_view_plotly already used for solar mode, applied
    # here too.
    #
    # Lunar mode needs a DIFFERENT camera direction, not the same one —
    # this was a real bug: in a lunar eclipse the Moon sits in OPPOSITION
    # to the Sun (verified directly: 179.84 deg between Moon and Sun
    # direction at peak, not 0), so aiming the camera toward the Sun's
    # direction puts the Moon almost exactly behind Earth from the
    # camera's point of view — hidden, not just small. A side-on view,
    # roughly perpendicular to the Sun direction, shows Earth and the
    # Moon next to each other instead of one hiding behind the other.
    camera_eye = _lunar_or_solar_camera_eye(mode, sun_hat_mid)
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title=dict(text="X [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title=dict(text="Y [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title=dict(text="Z [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=camera_eye),
        ),
        paper_bgcolor="black",
        font=dict(color="white", size=18),
        title=dict(text=f"{_window_title_prefix(mode, window)} — animated<br>"
                        f"<sub>t={(t_fine[frame_idxs[0]]-t_fine[mid])/3600:+.2f} hr, illum={first_illum:.3f}</sub>",
                  x=0.5, font=dict(color="white", size=24)),
        margin=dict(l=0, r=0, t=90, b=0),
        showlegend=False,
        updatemenus=[dict(
            type="buttons", showactive=False, y=0, x=0.05, xanchor="left", yanchor="top",
            pad=dict(t=0, r=10),
            # Play/Pause text defaults to small black lettering, which was
            # completely invisible against this scene's black background
            # — not just small, actually unreadable regardless of vision.
            # Explicit white font plus a visible button background fixes
            # both the contrast and the size at once.
            font=dict(color="white", size=16),
            bgcolor="#333344", bordercolor="#888", borderwidth=1,
            buttons=[
                dict(label="▶ Play", method="animate",
                    args=[None, dict(frame=dict(duration=180, redraw=True),
                                     fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause", method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False),
                                       mode="immediate", transition=dict(duration=0))]),
            ],
        )],
        sliders=[dict(
            active=0, y=0, x=0.15, len=0.8, xanchor="left", yanchor="top",
            pad=dict(t=0, b=10),
            font=dict(color="white", size=15),
            currentvalue=dict(font=dict(color="white", size=16), prefix="t = "),
            steps=[dict(method="animate", label=f"{(t_fine[idx]-t_fine[mid])/3600:+.1f}h",
                       args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")])
                  for k, idx in enumerate(frame_idxs)],
        )],
    )

    if save_path:
        fig.write_html(save_path)
        if str(save_path).endswith(".html"):
            _inject_eclipse_catalog_dropdown(save_path, default_mode=mode, default_event=event)
        print(f"Saved -> {save_path}")
    return fig


def plot_space_view_plotly(mode="lunar", search_days=None, save_path=None, verbose=True, event=None):
    """Recomputes the same real search as find_and_plot_eclipse() (so this
    can be called standalone) and renders the space-view panel in Plotly."""
    _, stats = find_and_plot_eclipse(mode=mode, search_days=search_days,
                                     save_path=None, verbose=verbose, event=event)
    window = _eclipse_window(mode, search_days=search_days, event=event, n_steps=4000, verbose=False)
    t_fine = window["t_s"]
    r_fine = window["r_moon"]
    sun_fine = window["sun_hat"]
    illum_fine = window["illum"]
    mid = window["peak_idx"]

    moon_r_km = r_fine[mid]
    sun_hat = sun_fine[mid]
    illum = illum_fine[mid]
    sep_deg = stats["angle_at_peak_deg"]

    size_boost = 16.0   # not 40 -- see the animated function for why
    fig = go.Figure()

    # Earth-fixed frame for both modes, same as the animated function —
    # Earth at the origin, Moon at its real position relative to Earth.
    earth_pos = np.array([0.0, 0.0, 0.0])
    moon_pos = moon_r_km
    occ_r_km = RE_KM if mode == "lunar" else R_MOON_KM
    cone_origin = earth_pos if mode == "lunar" else moon_pos
    cone_dir = -sun_hat
    cone_origin = cone_origin + cone_dir * (occ_r_km * size_boost * 0.03)

    _shadow_kwargs = {}
    if mode == "solar":
        _shadow_kwargs = dict(shadow_body_center_km=moon_pos, shadow_body_radius_km=R_MOON_KM)
    rotation_deg = _earth_rotation_for_window_sample(window, mid)
    fig.add_trace(_earth_mesh(sun_hat, radius_scale=size_boost, center=tuple(earth_pos),
                              rotation_deg=rotation_deg, **_shadow_kwargs))
    fig.add_trace(_earth_atmosphere_trace(center=tuple(earth_pos), radius_scale=size_boost))
    _real_moon_center = moon_r_km if mode == "lunar" else np.array([0.0, 0.0, 0.0])
    fig.add_trace(moon_mesh_plotly(moon_pos, R_MOON_KM*size_boost, sun_hat=sun_hat,
                                   real_center_km=_real_moon_center, mode=mode))

    ref_dist_km = D_MOON_A_KM
    base_r, umbra_len, pen_base_r, pen_len, pen_end_r, slope = _umbra_penumbra_geometry(
        occ_r_km, size_boost, ref_dist_km)
    fig.add_trace(_shadow_cone_trace(cone_origin, cone_dir, base_r, umbra_len,
                                     color="#1a0a0a", opacity=0.12, end_radius=0.0))
    fig.add_trace(_shadow_cone_trace(cone_origin, cone_dir, pen_base_r, pen_len,
                                     color="#4a5a7a", opacity=0.045, end_radius=pen_end_r))

    if mode == "solar":
        hit = None if illum > 0.999 else _shadow_ground_point(moon_pos, sun_hat, earth_pos, earth_r_real=RE_KM)
        if hit is not None:
            unit = (hit - earth_pos); unit = unit / np.linalg.norm(unit)
            hit_display = earth_pos + unit * RE_KM * size_boost
            dist_from_moon = np.linalg.norm(hit - moon_pos)
            r_umbra = abs(base_r - slope*dist_from_moon)
            r_pen = pen_base_r + slope*1.35*dist_from_moon
            for tr in _shadow_footprint_traces(earth_pos, hit_display, sun_hat, r_umbra, r_pen):
                fig.add_trace(tr)

    ref_r = np.linalg.norm(moon_r_km)
    arrow_len = ref_r * 0.55
    arrow_origin = earth_pos + sun_hat * (RE_KM*size_boost*1.15 if mode == "lunar"
                                          else np.linalg.norm(moon_pos) + R_MOON_KM*size_boost*1.15)
    for tr in _sun_direction_arrow(arrow_origin, sun_hat, arrow_len):
        fig.add_trace(tr)

    lim = (ref_r + max(RE_KM, R_MOON_KM)*size_boost) * 1.35
    fig.add_trace(_starfield_trace(lim))
    # Close initial zoom: a small `eye` vector (short distance from origin,
    # in the scene's own normalized units) starts the camera zoomed in near
    # the Earth/Moon system rather than Plotly's usual zoomed-out default —
    # scroll/drag to zoom out from there, same as everywhere else this has
    # come up in this project.
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-lim, lim], title=dict(text="X [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            yaxis=dict(range=[-lim, lim], title=dict(text="Y [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            zaxis=dict(range=[-lim, lim], title=dict(text="Z [km]", font=dict(size=18, color="white")),
                      tickfont=dict(size=14, color="white"),
                      backgroundcolor="black", gridcolor="#333", color="white"),
            bgcolor="black",
            aspectmode="cube",
            camera=dict(eye=_lunar_or_solar_camera_eye(mode, sun_hat)),
        ),
        paper_bgcolor="black",
        font=dict(color="white", size=18),
        title=dict(text=f"{_window_title_prefix(mode, window)} — view from space<br>"
                        f"<sub>illum={illum:.3f}, angle at peak={sep_deg:.3f}°</sub>",
                  x=0.5, font=dict(color="white", size=24)),
        margin=dict(l=0, r=0, t=90, b=0),
        showlegend=False,
    )

    if save_path:
        if save_path.endswith(".html"):
            fig.write_html(save_path)
            _inject_eclipse_catalog_dropdown(save_path, default_mode=mode, default_event=event)
        else:
            fig.write_image(save_path, width=1000, height=900, scale=1)
        print(f"Saved -> {save_path}")
    return fig, stats


if __name__ == "__main__":
    from ssatk.plots.figpath import figpath

    plot_space_view_plotly(mode="lunar", save_path=figpath("eclipse_space_lunar.html"))
    plot_space_view_plotly(mode="solar", save_path=figpath("eclipse_space_solar.html"))
