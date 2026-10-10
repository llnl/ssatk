"""
tle_updater.py
==============
Automatic TLE updater for coverage_analysis.py

Fetches the latest TLEs from Space-Track.org (primary)
or Celestrak (fallback) and updates your SATELLITES list
before each analysis run.

Usage in coverage_analysis.py:
    from tle_updater import update_satellites_auto
    SATELLITES = update_satellites_auto(SATELLITES)

Setup:
    1. Create a free account at https://www.space-track.org/auth/createAccount
    2. Provide your credentials either via a local tle_credentials_local.py
       file (never committed) or the SPACETRACK_USER / SPACETRACK_PASSWORD
       environment variables — see the Credentials section below
    3. That's it — runs automatically every time coverage_analysis.py runs
"""

import os
import json
import time
import urllib.request
import urllib.parse
import ssl

# This module lives at ssatk/io/, so the repo root is three levels up.
# Defined here, above _load_local_credentials(), because that function needs it
# to locate tle_credentials_local.py at the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Credentials ──────────────────────────────────────────────────────────────
# Do NOT hardcode your Space-Track credentials here — this file is committed
# to a public repo. Instead, choose one of:
#
#   1. Create a local-only file called tle_credentials_local.py (already
#      listed in .gitignore, so it will never be committed) in this same
#      directory, containing:
#          ST_USER     = "you@example.com"
#          ST_PASSWORD = "your-password"
#
#   2. Or set environment variables before running:
#          Windows (PowerShell):  $env:SPACETRACK_USER = "you@example.com"
#                                  $env:SPACETRACK_PASSWORD = "your-password"
#          macOS/Linux:            export SPACETRACK_USER="you@example.com"
#                                   export SPACETRACK_PASSWORD="your-password"
#
# If neither is set, ST_USER stays as the placeholder below, and the
# existing "ST_USER != 'your_email@example.com'" checks elsewhere in this
# file correctly skip Space-Track and fall back to Celestrak only.
def _load_local_credentials():
    """Read ST_USER / ST_PASSWORD from a repo-root tle_credentials_local.py.

    A plain `from tle_credentials_local import ...` worked while this module
    sat at the repo root. Now that it lives in the package, that import only
    succeeds if the repo root happens to be on sys.path -- true when running
    from the repo, false for an installed wheel. Loading by explicit path
    makes it independent of both sys.path and the working directory.
    """
    cred_path = os.path.join(_REPO_ROOT, "tle_credentials_local.py")
    if not os.path.isfile(cred_path):
        return None, None
    import importlib.util
    spec = importlib.util.spec_from_file_location("tle_credentials_local", cred_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[tle_updater] could not read {os.path.basename(cred_path)}: {exc}")
        return None, None
    return getattr(module, "ST_USER", None), getattr(module, "ST_PASSWORD", None)


_local_user, _local_password = _load_local_credentials()
if _local_user:
    ST_USER, ST_PASSWORD = _local_user, _local_password or ""
else:
    ST_USER     = os.environ.get("SPACETRACK_USER", "your_email@example.com")
    ST_PASSWORD = os.environ.get("SPACETRACK_PASSWORD", "")

# ── Settings ─────────────────────────────────────────────────────────────────
# How old a cached TLE can be before re-fetching (seconds)
CACHE_MAX_AGE_SECONDS = 7200   # 2 hours

# save_satellites() skips mirroring into coverage_analysis.py above this many
# satellites -- see save_satellites()'s docstring for why.
PY_MIRROR_MAX_SATELLITES = 500

# coverage_analysis.py still lives in plots; generated data does not.
_PLOTS_DIR = os.path.join(_REPO_ROOT, "ssatk", "plots")


def _data_file(name):
    """Use configured output, then a legacy path, without failing import."""
    legacy_path = os.path.join(os.path.expanduser("~"), "ssatk_data", name)
    configured = bool(os.environ.get("SSATK_OUTPUT_DIR"))
    if not configured and os.path.isfile(legacy_path):
        return legacy_path
    try:
        from .ssatk_data import datapath
        return str(datapath(name))
    except (OSError, RuntimeError):
        return legacy_path


def _cache_file():
    """Resolve the cache lazily so configuration changes take effect."""
    return _data_file("tle_cache.json")

# SSL context — verifies certificates by default. Only bypass verification
# if you're on a restricted network that requires it, by setting:
#   SPACETRACK_INSECURE_SSL=1
# as an environment variable. This keeps the insecure behavior opt-in
# instead of silently disabling certificate checks for every user.
SSL_CTX = ssl.create_default_context()
if os.environ.get("SPACETRACK_INSECURE_SSL") == "1":
    SSL_CTX.check_hostname = False
    SSL_CTX.verify_mode    = ssl.CERT_NONE


# ═════════════════════════════════════════════════════════════════════════════
# CACHE
# ═════════════════════════════════════════════════════════════════════════════

def _load_cache():
    """Load local TLE cache from disk."""
    cache_file = _cache_file()
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache):
    """Save TLE cache to disk."""
    try:
        with open(_cache_file(), 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"  [tle_updater] Could not save cache: {e}")


def _cache_is_fresh(cache, norad_id):
    """Return True if the cached TLE for norad_id is less than CACHE_MAX_AGE_SECONDS old."""
    key = str(norad_id)
    if key not in cache:
        return False
    fetched_at = cache[key].get("fetched_at", 0)
    age = time.time() - fetched_at
    return age < CACHE_MAX_AGE_SECONDS


# ═════════════════════════════════════════════════════════════════════════════
# NORAD ID PARSER
# ═════════════════════════════════════════════════════════════════════════════

def _norad_from_tle(line1):
    """Extract NORAD catalog ID from TLE line 1."""
    try:
        return int(line1[2:7].strip())
    except Exception:
        return None


# ═════════════════════════════════════════════════════════════════════════════
# CELESTRAK FETCHER (no account needed)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_tle_celestrak(norad_id):
    """
    Fetch the latest TLE for a satellite from Celestrak.
    No account required. Uses SSL bypass for restricted networks.
    Returns (line1, line2) or (None, None) on failure.
    """
    url = (f"https://celestrak.org/NORAD/elements/gp.php"
           f"?CATNR={norad_id}&FORMAT=JSON")
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ssapy-tle-updater/1.0"}
        )
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
            data = json.loads(r.read().decode())
            if data and len(data) > 0:
                return data[0].get("TLE_LINE1"), data[0].get("TLE_LINE2")
    except Exception as e:
        print(f"  [celestrak] NORAD {norad_id} failed: {e}")
    return None, None


# ═════════════════════════════════════════════════════════════════════════════
# SPACE-TRACK FETCHER (requires free account)
# ═════════════════════════════════════════════════════════════════════════════

def fetch_tle_spacetrack(norad_id, session_cookie=None):
    """
    Fetch the latest TLE for a satellite from Space-Track.org.
    Requires a free account — set ST_USER and ST_PASSWORD at the top of this file.
    Returns (line1, line2, cookie) or (None, None, None) on failure.
    cookie: reuse the session cookie across multiple calls to avoid re-login.
    """
    login_url  = "https://www.space-track.org/ajaxauth/login"
    query_url  = (f"https://www.space-track.org/basicspacedata/query/class/gp/"
              f"NORAD_CAT_ID/{norad_id}/orderby/EPOCH%20desc/limit/1/format/json")

    try:
        # Login if no session cookie yet
        if session_cookie is None:
            login_data = urllib.parse.urlencode({
                "identity": ST_USER,
                "password": ST_PASSWORD
            }).encode()
            req = urllib.request.Request(
                login_url,
                data=login_data,
                method="POST"
            )
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
                session_cookie = r.headers.get("Set-Cookie", "")

        # Query TLE
        req = urllib.request.Request(query_url)
        if session_cookie:
            req.add_header("Cookie", session_cookie)
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
            data = json.loads(r.read().decode())
            if data and len(data) > 0:
                name = data[0].get("OBJECT_NAME", f"NORAD-{norad_id}").strip()
                line1 = data[0].get("TLE_LINE1")
                line2 = data[0].get("TLE_LINE2")
                # Cache the name
                cache = _load_cache()
                cache[str(norad_id)] = cache.get(str(norad_id), {})
                cache[str(norad_id)]["object_name"] = name
                _save_cache(cache)
                return line1, line2, session_cookie

    except Exception as e:
        print(f"  [space-track] NORAD {norad_id} failed: {e}")

    return None, None, session_cookie


def _spacetrack_login(session_cookie=None):
    """
    Shared login helper — logs in only if session_cookie is None, otherwise
    just returns what was passed in. Factored out so fetch_active_catalog()
    doesn't duplicate the login block that fetch_tle_spacetrack() and
    fetch_by_name() each already have inline.
    """
    if session_cookie is not None:
        return session_cookie
    login_data = urllib.parse.urlencode({
        "identity": ST_USER,
        "password": ST_PASSWORD
    }).encode()
    req = urllib.request.Request(
        "https://www.space-track.org/ajaxauth/login",
        data=login_data, method="POST"
    )
    with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
        return r.headers.get("Set-Cookie", "")


# ═════════════════════════════════════════════════════════════════════════════
# BULK CATALOG FETCHER — every currently-active satellite, in ONE request
# instead of one request per satellite
# ═════════════════════════════════════════════════════════════════════════════
#
# fetch_group() and fetch_tle_spacetrack() above are built for a short,
# hand-picked list: one HTTP request per NORAD ID, with a 2.2s sleep between
# each to respect Space-Track's general rate limit. That's fine for ~100
# satellites (a few minutes) but does not scale to the full catalog.
#
# This function was originally written to paginate the bulk query into
# ~2000-record pages via limit/offset. That turned out to be wrong on two
# counts, both found by actually reading Space-Track's rate-limit table
# (https://www.space-track.org/documentation, "API Use Guidelines" — most
# of the page is login-gated, but this table isn't):
#
#   GP (aka TLEs) | 1 / hour | ...add "/decay_date/null-val/epoch/%3Enow-10/"
#   to the URL to ensure you only retrieve propagable ephemerides for
#   on-orbit objects.
#
# Space-Track expects ONE GP-class request per hour to cover everything you
# need, not many small paginated ones seconds apart — the latter is exactly
# the kind of per-endpoint hammering their throttling exists to catch, and
# is a plausible cause of a request outright failing (e.g. HTTP 500) rather
# than just returning fewer records than paginating would expect.
#
# The query below is Space-Track's own literal link target from their
# Alpha-5 FAQ (the surrounding prose renders "<" and ">" unescaped for
# readability, but the actual href is percent-encoded — worth checking the
# real link target directly rather than the human-readable text around it,
# which is the mistake in this function's first version: it sent a raw,
# unencoded "<" in NORAD_CAT_ID/<100000, which almost certainly caused the
# 500 error seen when this was first tried):
#
#   class/gp/NORAD_CAT_ID/%3C100000/decay_date/null-val/epoch/%3Enow-10/
#     orderby/norad_cat_id/format/json
#
# Field meaning:
#   NORAD_CAT_ID/%3C100000 — "<100000", excludes Alpha-5 (6-character
#                           alphanumeric) catalog numbers, which classic
#                           TLE-based propagators (including this repo's,
#                           and satellite.js as used in the 3D viewer)
#                           can't parse as a plain integer NORAD ID anyway.
#   decay_date/null-val   — only objects that haven't decayed/reentered.
#   epoch/%3Enow-10        — ">now-10", only element sets updated in the
#                           last 10 days (Space-Track's own recommended
#                           window for this specific hourly-refresh use
#                           case) -- "actively maintained", not a stale
#                           historic entry nobody's tracking anymore. This
#                           is what actually gets you to "~25,000-30,000"
#                           rather than the ~69,000+ SATCAT total, which
#                           includes decades of decayed debris and rocket
#                           bodies with no current TLE at all.

def fetch_active_catalog(session_cookie=None, max_satellites=None, verbose=True):
    """
    Fetch every currently-active, non-decayed satellite from Space-Track in
    a single request, matching Space-Track's own documented usage pattern
    for the GP class (one comprehensive query, not many small paginated
    ones -- see the comment block above this function for why).

    Parameters
    ----------
    session_cookie : str
        Reuse an existing Space-Track session cookie.
    max_satellites : int or None
        Add a server-side limit/N to the query, for testing this on a small
        slice without waiting for/downloading the full response. None =
        fetch everything available (this is what you want for real use --
        remember Space-Track's own guidance is ONE such request per hour,
        so there's no benefit to capping it lower "to be safe" the way you
        might with a paginated API).
    verbose : bool
        Print progress.

    Returns
    -------
    list of dicts, session_cookie

    NOTE: this hits Space-Track's live API and has not been run against a
    real account by me while writing this — I don't have (and shouldn't
    ask you for) your credentials to test it end-to-end. The query itself
    is Space-Track's own literal documented link target, and the response
    parsing mirrors fetch_tle_spacetrack()/fetch_by_name() elsewhere in
    this file, but if the actual JSON field names or response shape differ
    from what's coded here, that'll show up as every record failing to
    parse — if that happens, print(data[0]) for one raw record and compare
    against what's expected below (OBJECT_NAME / TLE_LINE1 / TLE_LINE2).

    Also worth knowing before running this for real: Space-Track's own
    table states GP-class queries like this one are meant to be run at
    most once per hour for automated scripts. This function doesn't
    enforce that itself (it's a single manual CLI invocation, not a
    background loop) -- it's on you not to re-run --fetch-all-active
    repeatedly within the same hour.
    """
    if ST_USER == "your_email@example.com":
        print("  [active-catalog] Space-Track credentials not set — see the "
              "Credentials section at the top of this file.")
        return [], session_cookie

    try:
        session_cookie = _spacetrack_login(session_cookie)
    except Exception as e:
        print(f"  [active-catalog] Login failed: {e}")
        return [], session_cookie

    url = ("https://www.space-track.org/basicspacedata/query/class/gp/"
           "NORAD_CAT_ID/%3C100000/decay_date/null-val/epoch/%3Enow-10/"
           "orderby/norad_cat_id")
    if max_satellites is not None:
        url += f"/limit/{max_satellites}"
    url += "/format/json"

    if verbose:
        print("\n  Fetching active catalog from Space-Track in a single "
              "request (non-decayed, epoch within 10 days)...")
        print("  (this can take a little while and the response can be "
              "several MB — that's expected for ~25,000-30,000 objects)")

    req = urllib.request.Request(url)
    req.add_header("Cookie", session_cookie)

    data = None
    for attempt in range(2):  # one retry, same spirit as fetch_group()
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=180) as r:
                data = json.loads(r.read().decode())
            break
        except Exception as e:
            if attempt == 0:
                if verbose:
                    print(f"    failed ({e}), retrying once after 10s...")
                time.sleep(10)
            else:
                print(f"    failed twice ({e}) — see the docstring above "
                      f"this function for the two most likely causes "
                      f"(query encoding, or the once-per-hour GP limit if "
                      f"this was run again too soon after a prior attempt).")
                return [], session_cookie

    results = []
    for record in (data or []):
        name  = record.get("OBJECT_NAME", f"NORAD-{record.get('NORAD_CAT_ID')}")
        line1 = record.get("TLE_LINE1")
        line2 = record.get("TLE_LINE2")
        if line1 and line2:
            results.append({
                "name":  name.strip() if isinstance(name, str) else name,
                "type":  "tle",
                "line1": line1,
                "line2": line2,
            })

    if verbose:
        print(f"  Done — {len(results)} active satellites fetched "
              f"in 1 request ({len(data or []) - len(results)} records in "
              f"the response were missing a usable TLE and were skipped).")

    return results, session_cookie


def add_active_catalog_to_satellites(satellites, max_satellites=None, verbose=True):
    """
    Fetch the full active catalog and merge into an existing SATELLITES
    list. Skips duplicates based on NORAD ID, same as add_group_to_satellites().
    """
    new_sats, _ = fetch_active_catalog(max_satellites=max_satellites, verbose=verbose)

    existing_norads = set()
    for sat in satellites:
        if sat.get("type") == "tle" and sat.get("line1"):
            nid = _norad_from_tle(sat["line1"])
            if nid:
                existing_norads.add(nid)

    added = 0
    for sat in new_sats:
        nid = _norad_from_tle(sat["line1"])
        if nid and nid not in existing_norads:
            satellites.append(sat)
            existing_norads.add(nid)
            added += 1

    if verbose:
        print(f"  Added {added} new satellites from the active catalog "
              f"({len(new_sats) - added} already present)")

    return satellites

# ═════════════════════════════════════════════════════════════════════════════
# GROUP FETCHER — pull entire categories from Space-Track
# ═════════════════════════════════════════════════════════════════════════════

# Predefined groups — add your own by finding the Space-Track catalog numbers
SATELLITE_GROUPS = {
    "iss":          [25544],
    "hubble":       [20580],
    "jwst":         [50463],
    "gps":          [24876, 25933, 26360, 26407, 26605, 27663, 28129,
                     28361, 28474, 28874, 29486, 29601, 32260, 32384,
                     32711, 35752, 36585, 37753, 38833, 39166, 39533,
                     39741, 40105, 40294, 40534, 40730, 41019, 41328,
                     43873, 44506],
    "galileo":      [37846, 37847, 38857, 38858, 40128, 40129, 40544,
                     40545, 41175, 41174, 41550, 41549, 43055, 43056,
                     43057, 43058, 45598, 45600],
    "glonass":      [32276, 32275, 32393, 32395, 33108, 33110, 33111,
                     33112, 34953, 36111, 36112, 36113, 36114, 36400,
                     36401, 37136, 37137, 37138, 37139, 39620, 40001],
    "beidou":       [43706, 43907, 44204, 44337, 44709, 44794, 44864,
                     44865, 45344, 45345, 45807, 46605, 4751],
    "weather_noaa": [33591, 38771, 43689, 25338, 28654, 43013, 37849],
    "weather_goes": [41866, 43226],
    "weather_metop":[29499, 38771, 43689],
    "landsat":      [39084, 49260],
    "sentinel":     [39634, 40697, 41335, 43437, 44413, 45596],
    "terra_aqua":   [25994, 27424],
    "icesat":       [43613],
    "chandra":      [25867],
    "fermi":        [33053],
    "swift":        [28485],
    "css":          [54216, 48274, 49044],
    "starlink_sample": [44235, 44237, 44238, 44239, 44240,
                        44241, 44244, 44245, 44246, 44247],
    # TESS — cross-checked against toolkit_gui.py's own PRESET_NORAD dict
    # (same NORAD ID used there for the "TESS (HEO)" preset), not a fresh
    # unverified lookup.
    "tess":         [43435],
}

# ── Convenience meta-groups ───────────────────────────────────────────────────
# These don't introduce any new/unverified NORAD IDs — they just bundle the
# individual groups above into broader categories, so you can fetch a whole
# theme in one call (e.g. fetch_group("earth_observation")) instead of
# looping over several fetch_group() calls yourself.
SATELLITE_GROUPS["navigation_all"] = (
    SATELLITE_GROUPS["gps"] + SATELLITE_GROUPS["galileo"]
    + SATELLITE_GROUPS["glonass"] + SATELLITE_GROUPS["beidou"]
)
SATELLITE_GROUPS["weather_all"] = (
    SATELLITE_GROUPS["weather_noaa"] + SATELLITE_GROUPS["weather_goes"]
    + SATELLITE_GROUPS["weather_metop"]
)
SATELLITE_GROUPS["earth_observation"] = (
    SATELLITE_GROUPS["landsat"] + SATELLITE_GROUPS["sentinel"]
    + SATELLITE_GROUPS["terra_aqua"] + SATELLITE_GROUPS["icesat"]
)
SATELLITE_GROUPS["science_observatories"] = (
    SATELLITE_GROUPS["hubble"] + SATELLITE_GROUPS["jwst"]
    + SATELLITE_GROUPS["chandra"] + SATELLITE_GROUPS["fermi"]
    + SATELLITE_GROUPS["swift"] + SATELLITE_GROUPS["tess"]
)
SATELLITE_GROUPS["space_stations"] = (
    SATELLITE_GROUPS["iss"] + SATELLITE_GROUPS["css"]
)
# Broadest convenience group: one of everything above, deduplicated.
SATELLITE_GROUPS["all_sample"] = sorted(set(
    SATELLITE_GROUPS["navigation_all"] + SATELLITE_GROUPS["weather_all"]
    + SATELLITE_GROUPS["earth_observation"] + SATELLITE_GROUPS["science_observatories"]
    + SATELLITE_GROUPS["space_stations"] + SATELLITE_GROUPS["starlink_sample"]
))


def fetch_group(group_name, session_cookie=None, verbose=True):
    """
    Fetch TLEs for a predefined group of satellites from Space-Track.
    Returns a list of satellite dicts ready to add to SATELLITES.

    Parameters
    ----------
    group_name : str
        Key from SATELLITE_GROUPS dict, e.g. 'gps', 'starlink_sample'
    session_cookie : str
        Reuse existing Space-Track session cookie.
    verbose : bool
        Print status.

    Returns
    -------
    list of dicts, session_cookie
    """
    if group_name not in SATELLITE_GROUPS:
        print(f"  [group] Unknown group '{group_name}'")
        print(f"  Available: {', '.join(SATELLITE_GROUPS.keys())}")
        return [], session_cookie

    norad_ids = SATELLITE_GROUPS[group_name]
    results   = []
    _consecutive_fails = 0

    if verbose:
        print(f"\n  Fetching group '{group_name}' ({len(norad_ids)} satellites)...")

    for norad_id in norad_ids:
        line1, line2, session_cookie = fetch_tle_spacetrack(
            norad_id, session_cookie)

        # A run of several fails in a row (with no exception message from
        # fetch_tle_spacetrack itself) almost always means throttling, not
        # "these specific satellites don't exist" — Space-Track enforces
        # both a per-minute AND a per-hour cap, so this can kick in even at
        # a well-paced request rate if earlier runs this session already
        # used up part of the hourly budget. Back off and retry once rather
        # than just logging more fails and ploughing through the rest.
        if not (line1 and line2):
            _consecutive_fails += 1
            if _consecutive_fails >= 3:
                if verbose:
                    print(f"    [throttle?] {_consecutive_fails} fails in a row — "
                          f"pausing 60s and retrying {norad_id}...")
                time.sleep(60)
                line1, line2, session_cookie = fetch_tle_spacetrack(norad_id, session_cookie)
                if line1 and line2:
                    _consecutive_fails = 0
        else:
            _consecutive_fails = 0

        if line1 and line2:
            # Extract name from TLE line 2 checksum area is not useful
            # Use Space-Track name from cache if available
            cache = _load_cache()
            cached = cache.get(str(norad_id), {})
            name = cached.get("object_name", f"NORAD-{norad_id}")
            results.append({
                "name":  name,
                "type":  "tle",
                "line1": line1,
                "line2": line2,
            })
            if verbose:
                print(f"    [ok] {norad_id} -> {name}")
        else:
            if verbose:
                print(f"    [fail] {norad_id}")
        # Space-Track's documented limit is ~30 requests/minute; 0.3s was
        # ~195 req/min (6x over), which silently throttles large batches —
        # Space-Track tends to return an empty-but-valid response when
        # throttled rather than a clear error, which is why failures here
        # show no underlying exception message. 2.2s keeps well under 30/min.
        time.sleep(2.2)

    return results, session_cookie


def fetch_by_name(search_term, max_results=10,
                  session_cookie=None, verbose=True):
    """
    Search Space-Track for satellites by name keyword.
    e.g. fetch_by_name('STARLINK') fetches Starlink satellites.

    Parameters
    ----------
    search_term : str
        Keyword to search in satellite names.
    max_results : int
        Maximum number of satellites to return.
    session_cookie : str
        Reuse existing Space-Track session.
    verbose : bool
        Print status.

    Returns
    -------
    list of dicts, session_cookie
    """
    import urllib.parse

    if ST_USER == "your_email@example.com":
        print("  [search] Space-Track credentials not set.")
        return [], session_cookie

    # Login if needed
    if session_cookie is None:
        login_data = urllib.parse.urlencode({
            "identity": ST_USER,
            "password": ST_PASSWORD
        }).encode()
        req = urllib.request.Request(
            "https://www.space-track.org/ajaxauth/login",
            data=login_data, method="POST"
        )
        try:
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
                session_cookie = r.headers.get("Set-Cookie", "")
        except Exception as e:
            print(f"  [search] Login failed: {e}")
            return [], session_cookie

    # Search by name
    url = (f"https://www.space-track.org/basicspacedata/query/class/gp/"
           f"OBJECT_NAME/~~{urllib.parse.quote(search_term)}/"
           f"orderby/EPOCH%20desc/limit/{max_results}/format/json")

    results = []
    try:
        req = urllib.request.Request(url)
        req.add_header("Cookie", session_cookie)
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
            data = json.loads(r.read().decode())

        if verbose:
            print(f"\n  Search '{search_term}': found {len(data)} results")

        for record in data:
            name  = record.get("OBJECT_NAME", f"NORAD-{record.get('NORAD_CAT_ID')}")
            line1 = record.get("TLE_LINE1")
            line2 = record.get("TLE_LINE2")
            if line1 and line2:
                results.append({
                    "name":  name.strip(),
                    "type":  "tle",
                    "line1": line1,
                    "line2": line2,
                })
                if verbose:
                    print(f"    [ok] {name.strip()}")

    except Exception as e:
        print(f"  [search] Query failed: {e}")

    return results, session_cookie


def add_group_to_satellites(satellites, group_name, verbose=True):
    """
    Fetch a group and merge into existing SATELLITES list.
    Skips duplicates based on NORAD ID.

    Parameters
    ----------
    satellites : list
        Your existing SATELLITES list.
    group_name : str
        Group name from SATELLITE_GROUPS.
    verbose : bool
        Print status.

    Returns
    -------
    Updated SATELLITES list.
    """
    new_sats, _ = fetch_group(group_name, verbose=verbose)

    # Get existing NORAD IDs to avoid duplicates
    existing_norads = set()
    for sat in satellites:
        if sat.get("type") == "tle" and sat.get("line1"):
            nid = _norad_from_tle(sat["line1"])
            if nid:
                existing_norads.add(nid)

    added = 0
    for sat in new_sats:
        nid = _norad_from_tle(sat["line1"])
        if nid not in existing_norads:
            satellites.append(sat)
            existing_norads.add(nid)
            added += 1

    if verbose:
        print(f"  Added {added} new satellites from group '{group_name}'")

    return satellites


def add_search_to_satellites(satellites, search_term,
                              max_results=10, verbose=True):
    """
    Search by name and merge results into SATELLITES list.

    Parameters
    ----------
    satellites : list
        Your existing SATELLITES list.
    search_term : str
        Keyword to search e.g. 'STARLINK', 'HUBBLE', 'JAMES WEBB'
    max_results : int
        Max satellites to add.

    Returns
    -------
    Updated SATELLITES list.
    """
    new_sats, _ = fetch_by_name(search_term, max_results, verbose=verbose)

    existing_norads = set()
    for sat in satellites:
        if sat.get("type") == "tle" and sat.get("line1"):
            nid = _norad_from_tle(sat["line1"])
            if nid:
                existing_norads.add(nid)

    added = 0
    for sat in new_sats:
        nid = _norad_from_tle(sat["line1"])
        if nid and nid not in existing_norads:
            satellites.append(sat)
            existing_norads.add(nid)
            added += 1

    if verbose:
        print(f"  Added {added} satellites matching '{search_term}'")

    return satellites

# ═════════════════════════════════════════════════════════════════════════════
# MAIN UPDATER
# ═════════════════════════════════════════════════════════════════════════════

def update_satellites_auto(satellites, use_spacetrack=True, use_celestrak=True,
                            force_refresh=False, verbose=True):
    """
    Update TLEs in your SATELLITES list automatically.

    For each satellite with type='tle':
      1. Check if cached TLE is still fresh (< 2 hours old)
      2. If stale, try Space-Track first (most accurate)
      3. Fall back to Celestrak if Space-Track fails
      4. Update the satellite dict with the new TLE
      5. Save to local cache for next run

    Parameters
    ----------
    satellites : list
        Your SATELLITES list from coverage_analysis.py
    use_spacetrack : bool
        Try Space-Track.org first. Requires ST_USER and ST_PASSWORD.
    use_celestrak : bool
        Fall back to Celestrak if Space-Track fails.
    force_refresh : bool
        Ignore cache and always fetch fresh TLEs.
    verbose : bool
        Print update status for each satellite.

    Returns
    -------
    list
        Updated SATELLITES list with fresh TLEs.
    """

    cache          = _load_cache()
    session_cookie = None
    updated        = 0
    cached         = 0
    failed         = 0

    if verbose:
        print("\n── TLE Updater ─────────────────────────────────────────")
        print(f"  Cache file : {_cache_file()}")
        print(f"  Max age    : {CACHE_MAX_AGE_SECONDS // 60} minutes")
        print(f"  Sources    : {'Space-Track + ' if use_spacetrack else ''}{'Celestrak' if use_celestrak else ''}")
        print()

    for sat in satellites:
        if sat.get("type") != "tle":
            continue

        # Get NORAD ID from existing TLE line1
        norad_id = _norad_from_tle(sat.get("line1", ""))
        if norad_id is None:
            if verbose:
                print(f"  [skip] {sat['name']} — could not parse NORAD ID")
            continue

        cache_key = str(norad_id)

        # Use cache if fresh and not forcing refresh
        if not force_refresh and _cache_is_fresh(cache, norad_id):
            cached_entry = cache[cache_key]
            sat["line1"] = cached_entry["line1"]
            sat["line2"] = cached_entry["line2"]
            age_min = int((time.time() - cached_entry["fetched_at"]) / 60)
            if verbose:
                print(f"  [cache]  {sat['name']:<22} NORAD {norad_id}  ({age_min} min old)")
            cached += 1
            continue

        # Try Space-Track
        line1, line2 = None, None
        if use_spacetrack and ST_USER != "your_email@example.com":
            line1, line2, session_cookie = fetch_tle_spacetrack(
                norad_id, session_cookie)
            if line1:
                source = "space-track"

        # Fall back to Celestrak
        if line1 is None and use_celestrak:
            line1, line2 = fetch_tle_celestrak(norad_id)
            if line1:
                source = "celestrak"

        if line1 and line2:
            sat["line1"] = line1
            sat["line2"] = line2
            cache[cache_key] = {
                "name":       sat["name"],
                "norad_id":   norad_id,
                "line1":      line1,
                "line2":      line2,
                "fetched_at": time.time(),
                "source":     source,
            }
            updated += 1
            if verbose:
                epoch = line1[18:32].strip()
                print(f"  [update] {sat['name']:<22} NORAD {norad_id}  epoch {epoch}  via {source}")
        else:
            failed += 1
            if verbose:
                print(f"  [FAIL]   {sat['name']:<22} NORAD {norad_id}  — keeping existing TLE")

        # Same Space-Track rate-limit reasoning as fetch_group() above —
        # 0.3s is ~195 req/min, well over the ~30/min Space-Track allows.
        time.sleep(2.2)

    _save_cache(cache)

    if verbose:
        print(f"\n  Updated: {updated}  |  From cache: {cached}  |  Failed: {failed}")
        print(f"  Cache saved -> {_cache_file()}")
        print()

    return satellites

SATELLITES_JSON = _data_file("ssapy_satellites.json")


# ═════════════════════════════════════════════════════════════════════════════
# VERIFICATION — confirm what's actually in the local store, rather than
# just trusting the fetch's own summary line
# ═════════════════════════════════════════════════════════════════════════════
#
# "31,072 fetched" on its own doesn't confirm three separate things worth
# checking:
#   1. The data that landed in the JSON store is intact (right count, no
#      corruption, no duplicate NORAD IDs slipping past the merge dedup).
#   2. Every individual TLE is well-formed (right length, checksum matches,
#      and line 1 / line 2 actually agree on which object they describe --
#      easy to get wrong if a response ever got truncated or mis-parsed).
#   3. "Satellites" is actually what's in there. The fetch_active_catalog()
#      filter (decay_date/null-val + recent epoch) matches any non-decayed
#      TRACKED OBJECT with a current TLE -- that includes rocket bodies and
#      debris fragments, not just payloads. Space-Track's own SATCAT
#      documentation defines object type by substring in the name:
#        DEBRIS:      'DEB', 'COOLANT', 'SHROUD', or 'WESTFORD NEEDLES'
#        ROCKET BODY: 'R/B' or 'AKM' or 'PKM' (and not already DEBRIS)
#        otherwise:   treated as a payload/satellite
#      So the ~31k figure very likely undercounts how many are debris/rocket
#      bodies relative to how many are "satellites" in the everyday sense --
#      this function reports that breakdown rather than assuming it.

def _tle_checksum(line):
    """
    Space-Track's own documented algorithm: sum the digits in the first 68
    characters, treating '-' as 1 and everything else non-digit as 0; the
    checksum is that sum mod 10. Verified against their own worked example
    in the TLE Format section of their docs (ISS (ZARYA), checksums 3 and 6).
    """
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == '-':
            total += 1
    return total % 10


def _classify_object(name):
    """Payload / rocket body / debris, by Space-Track's own SATCAT naming rules."""
    upper = name.upper()
    if any(s in upper for s in ('DEB', 'COOLANT', 'SHROUD', 'WESTFORD NEEDLES')):
        return "debris"
    if any(s in upper for s in ('R/B', 'AKM', 'PKM')):
        return "rocket_body"
    return "payload"


def verify_satellites(satellites, verbose=True):
    """
    Run local integrity + composition checks against an already-loaded
    SATELLITES list. Read-only -- doesn't fetch anything or modify the list.

    Checks:
      - duplicate NORAD IDs (would mean the merge dedup let something through)
      - line1/line2 length (69 chars each, the standard TLE fixed width)
      - line1 vs line2 NORAD ID agreement (catches mismatched/corrupted pairs)
      - TLE checksum validity on both lines
      - payload / rocket body / debris breakdown (see comment block above)

    Returns a dict summary; also prints a report if verbose.
    """
    seen_norads = {}
    duplicates = []
    bad_length = []
    id_mismatch = []
    bad_checksum = []
    counts = {"payload": 0, "rocket_body": 0, "debris": 0}
    no_norad = 0

    for sat in satellites:
        if sat.get("type") != "tle":
            continue
        line1 = sat.get("line1", "") or ""
        line2 = sat.get("line2", "") or ""
        name  = sat.get("name", "")

        nid = _norad_from_tle(line1)
        if nid is None:
            no_norad += 1
            continue

        if nid in seen_norads:
            duplicates.append(nid)
        else:
            seen_norads[nid] = name

        if len(line1) != 69 or len(line2) != 69:
            bad_length.append(nid)
            continue  # checksum/NORAD-agreement checks below assume length 69

        nid2 = _norad_from_tle(line2)  # same [2:7] column slice works on either line
        if nid2 != nid:
            id_mismatch.append(nid)

        line1_ok = line1[68].isdigit() and _tle_checksum(line1) == int(line1[68])
        line2_ok = line2[68].isdigit() and _tle_checksum(line2) == int(line2[68])
        if not (line1_ok and line2_ok):
            bad_checksum.append(nid)

        counts[_classify_object(name)] += 1

    total_tle = sum(1 for s in satellites if s.get("type") == "tle")

    if verbose:
        print("\n── Verification report ─────────────────────────────────")
        print(f"  Total entries (type='tle'): {total_tle}")
        print(f"  Unique NORAD IDs:           {len(seen_norads)}")
        if no_norad:
            print(f"  [!] Could not parse a NORAD ID at all: {no_norad}")
        if duplicates:
            print(f"  [!] Duplicate NORAD IDs found: {len(duplicates)} "
                  f"(e.g. {duplicates[:5]}) -- the merge dedup should have "
                  f"caught these; worth reporting if you see any here.")
        else:
            print("  [ok] No duplicate NORAD IDs.")
        if bad_length:
            print(f"  [!] Wrong line length (not 69 chars): {len(bad_length)} "
                  f"entries (e.g. NORAD {bad_length[:5]})")
        else:
            print("  [ok] All TLE lines are the standard 69 characters.")
        if id_mismatch:
            print(f"  [!] line1/line2 NORAD ID disagree: {len(id_mismatch)} "
                  f"entries (e.g. NORAD {id_mismatch[:5]}) -- these pairs may "
                  f"be corrupted or mismatched.")
        else:
            print("  [ok] line1/line2 NORAD IDs agree on every entry.")
        if bad_checksum:
            print(f"  [!] Failed TLE checksum: {len(bad_checksum)} entries "
                  f"(e.g. NORAD {bad_checksum[:5]})")
        else:
            print("  [ok] Every TLE line's checksum is valid.")
        print()
        print("  Object type breakdown (by Space-Track's own SATCAT naming "
              "rules -- name-substring based, so treat as a good estimate "
              "rather than authoritative):")
        print(f"    payload:      {counts['payload']:>6}")
        print(f"    rocket body:  {counts['rocket_body']:>6}")
        print(f"    debris:       {counts['debris']:>6}")
        print(f"  If you specifically want satellites/payloads only, "
              f"{counts['rocket_body'] + counts['debris']} of these "
              f"{total_tle} entries are rocket bodies or debris, not payloads.")
        print()

    return {
        "total": total_tle,
        "unique_norads": len(seen_norads),
        "duplicates": duplicates,
        "bad_length": bad_length,
        "id_mismatch": id_mismatch,
        "bad_checksum": bad_checksum,
        "counts": counts,
        "no_norad": no_norad,
    }


def load_satellites(filepath=None):
    """
    Load the persisted satellite list. Tries the JSON store first (the real
    source of truth — immune to the regex-based .py splicing below ever
    silently failing to match), then falls back to importing SATELLITES
    from coverage_analysis.py for backward compatibility, then an empty list.
    """
    json_path = filepath or SATELLITES_JSON
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                sats = json.load(f)
            print(f"  Loaded {len(sats)} satellites from {json_path}")
            return sats
        except Exception as e:
            print(f"  Could not read {json_path} ({e}) — trying coverage_analysis.py")

    try:
        try:
            from ssatk.plots.coverage_analysis import SATELLITES
        except ImportError:
            # Legacy fallback: only works if coverage_analysis.py happens to
            # be sitting directly on sys.path (the old, pre-restructure
            # flat layout).
            from coverage_analysis import SATELLITES
        print(f"  Loaded {len(SATELLITES)} satellites from coverage_analysis.py "
              f"(no {os.path.basename(json_path)} yet)")
        return SATELLITES
    except ImportError:
        print("  Note: no JSON store and coverage_analysis.py not found — "
              "starting with empty list")
        return []


def save_satellites(satellites, json_path=None, py_filepath=None):
    """
    Persist the satellite list. Always writes the JSON store (real source
    of truth, round-trips every field exactly, can't silently fail to
    match like the regex-based .py rewrite can) and also mirrors into
    coverage_analysis.py's SATELLITES block for whoever imports it directly
    -- but only up to PY_MIRROR_MAX_SATELLITES. Past that, the .py mirror
    is skipped: at bulk-catalog scale (thousands of satellites), writing
    every one as a literal Python dict into coverage_analysis.py would produce
    a multi-megabyte source file that's slow to parse on import and painful
    to diff/version-control, for a list nobody's going to hand-edit anyway.
    The JSON store has no such issue and remains complete regardless.
    """
    json_path = json_path or SATELLITES_JSON
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(satellites, f, indent=2)
    print(f"  Saved {len(satellites)} satellites -> {json_path}")

    if len(satellites) > PY_MIRROR_MAX_SATELLITES:
        print(f"  ({len(satellites)} satellites is over the "
              f"{PY_MIRROR_MAX_SATELLITES}-entry .py mirror threshold — "
              f"skipping the coverage_analysis.py mirror. The JSON store above "
              f"is complete and is what load_satellites() reads first anyway.)")
        return

    try:
        save_satellites_to_file(satellites, filepath=py_filepath)
    except FileNotFoundError:
        print("  (coverage_analysis.py not found — skipping .py mirror, "
              f"JSON store at {json_path} is up to date)")


def save_satellites_to_file(satellites, filepath=None):
    """
    Write the updated SATELLITES list back to coverage_analysis.py
    so changes persist between runs.

    NOTE: prefer save_satellites()/load_satellites() (the JSON store) as
    the real source of truth — this regex-based .py rewrite is kept for
    backward compatibility with code that does
    `from coverage_analysis import SATELLITES` directly, but it silently does
    nothing if the SATELLITES block's formatting doesn't match the regex
    below (no crash, just a printed warning) — always check for that
    warning in the output.
    """
    if filepath is None:
        filepath = os.path.join(_PLOTS_DIR, "coverage_analysis.py")

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Build the new SATELLITES block. Every field on every satellite dict
    # is preserved (previously, "type": "tle" entries only ever wrote back
    # name/type/line1/line2, silently dropping any other field you'd added
    # per-satellite on every save).
    lines = ["SATELLITES = [\n"]
    for sat in satellites:
        lines.append("    {\n")
        for k, v in sat.items():
            lines.append(f'        {repr(k)}: {repr(v)},\n')
        lines.append("    },\n")
    lines.append("]\n")
    new_block = "".join(lines)

    # Find and replace the SATELLITES block in the file
    import re
    pattern = r'SATELLITES\s*=\s*\[.*?\n\]\n'
    if re.search(pattern, content, re.DOTALL):
        new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  Saved {len(satellites)} satellites to {filepath}")
    else:
        print(f"  Could not find SATELLITES block in {filepath}")
# ═════════════════════════════════════════════════════════════════════════════
# STANDALONE — run directly to test or force-refresh all TLEs
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import sys
    import argparse
    sys.path.insert(0, os.path.expanduser("~"))

    parser = argparse.ArgumentParser(description="TLE updater")
    parser.add_argument("--force",         action="store_true",
                        help="Force refresh all TLEs ignoring cache")
    parser.add_argument("--no-spacetrack", action="store_true",
                        help="Use Celestrak only")
    parser.add_argument("--no-celestrak",  action="store_true",
                        help="Use Space-Track only, no Celestrak fallback")
    parser.add_argument("--add-group",     type=str, default=None,
                        help="Add a satellite group")
    parser.add_argument("--search",        type=str, default=None,
                        help="Search Space-Track by name")
    parser.add_argument("--max",           type=int, default=10,
                        help="Max results for --search (default 10)")
    parser.add_argument("--fetch-all-active", action="store_true",
                        help="Fetch every currently-active satellite from "
                             "Space-Track (~25,000-30,000) via a single bulk "
                             "query, instead of a hand-picked group")
    parser.add_argument("--max-active",    type=int, default=None,
                        help="Add a server-side limit to --fetch-all-active "
                             "(default: no limit, fetch everything available). "
                             "Useful for a quick test run first.")
    parser.add_argument("--list-groups",   action="store_true",
                        help="List all available satellite groups")
    parser.add_argument("--verify",        action="store_true",
                        help="Run local integrity + payload/rocket-body/"
                             "debris breakdown checks on the current "
                             "satellite list, without fetching anything")
    args = parser.parse_args()

    if args.list_groups:
        print("\nAvailable satellite groups:")
        for name, ids in SATELLITE_GROUPS.items():
            print(f"  {name:<20} ({len(ids)} satellites)")
        sys.exit(0)

    try:
        SATELLITES = load_satellites()
    except Exception as e:
        SATELLITES = []
        print(f"Note: could not load existing satellites ({e}), starting with empty list")

    if args.verify:
        verify_satellites(SATELLITES, verbose=True)

    elif args.add_group:
        SATELLITES = add_group_to_satellites(
            SATELLITES, args.add_group, verbose=True)
        save_satellites(SATELLITES)
        print(f"\nTotal satellites: {len(SATELLITES)}")

    elif args.fetch_all_active:
        SATELLITES = add_active_catalog_to_satellites(
            SATELLITES, max_satellites=args.max_active, verbose=True)
        save_satellites(SATELLITES)
        print(f"\nTotal satellites: {len(SATELLITES)}")

    elif args.search:
        SATELLITES = add_search_to_satellites(
            SATELLITES, args.search,
            max_results=args.max, verbose=True)
        save_satellites(SATELLITES)
        print(f"\nTotal satellites: {len(SATELLITES)}")

    else:
        SATELLITES = update_satellites_auto(
            SATELLITES,
            use_spacetrack=not args.no_spacetrack,
            use_celestrak=not args.no_celestrak,
            force_refresh=args.force,
            verbose=True,
        )
        save_satellites(SATELLITES)

    print("\nDone.")
