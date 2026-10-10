import re
import numpy as np
import pandas as pd
from ..constants import EARTH_MU, EARTH_RADIUS, J2_wgs  # m^3/s^2, m, dimensionless
from astropy.time import Time

def read_3le(data_file, makedf=True, verbose=False):
    """
    Parse a 3-line element (0/1/2) text file into a dict of lists or a DataFrame,
    and compute classical Keplerian elements at the TLE epoch.

    All distances are meters (m), velocities m/s, angles rad/deg, times seconds.

    Added fields (SI):
      - mean_motion_rev_day, n_rad_s, period_s
      - a_m, p_m, rp_m, ra_m, hp_m, ha_m
      - inc_rad, raan_rad, argp_rad, M_rad
      - E_rad/deg, nu_rad/deg, u_rad/deg
      - ta_rad/deg, r_mag_m, v_mag_m_s
      - r_eci_m_[x,y,z], v_eci_m_s_[x,y,z] (two-body, at epoch)
      - epoch_gps (float seconds)
      - a_sgp4, e_sgp4, i_sgp4, pa_sgp4, raan_sgp4, ta_sgp4, M_sgp4, n_sgp4
    """
    DAY_S = 86400.0
    TWOPI = 2.0 * np.pi

    def split_line(line):
        payload = re.findall(r'"[^"]+"|\'[^\']+\'|\S+', line)
        return [s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'" else s for s in payload]

    def _days_in_year(y: int) -> float:
        return 366.0 if ((y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)) else 365.0

    def _century_year(two_digit: str) -> int:
        yy = int(two_digit)
        return 1900 + yy if yy >= 57 else 2000 + yy

    def vprint(message):
        if verbose:
            print(message)

    def parse_ecc(tok: str) -> float:
        # TLE eccentricity: '0006703' -> 0.0006703; allow normal floats too.
        t = str(tok).strip()
        if t == "" or t.lower() == "nan":
            return np.nan
        if "." in t:
            return float(t)
        if re.fullmatch(r"\d+", t):
            return float("0." + t)
        raise ValueError(f"Bad eccentricity token: {tok!r}")

    with open(data_file, "r", encoding="utf-8", errors="replace") as f:
        def split_mantassa(text):
            t = text.strip()
            if not t or t.strip('0+-') == '':
                return 0.0
            # exponent is last two chars like '+5' or '-2'
            if len(t) < 3 or t[-2] not in '+-' or not t[-1].isdigit():
                raise ValueError(f"Bad TLE exp field: {text!r}")
            sign = '-' if t[0] == '-' else ''
            mant = t[1:-2] if t[0] in '+-' else t[:-2]
            return float(f"{sign}0.{mant}e{t[-2:]}")

        def looks_like_intl(tok: str) -> bool:
            return bool(re.fullmatch(r"\d{2}\d{3}[A-Za-z]{1,3}", tok))

        def parse_epoch_fields(epoch_token: str):
            t = epoch_token.replace(" ", "")
            yy = int(t[:2])
            epoch_year = (1900 + yy) if yy >= 57 else (2000 + yy)
            epoch_day = float(t[2:])  # DDD.dddddd
            return epoch_year, epoch_day

        cols = [
            "name",
            "satellite#",
            "classification",
            "intl_designator",
            "launch_year",
            "launch_number",
            "launch_piece",
            "elset_epoch",
            "epoch_year",
            "epoch_day",
            "decimal_year",
            "ndot",
            "nddot",
            "drag",
            "elset_type",
            "elset#",
            "inc_deg",
            "raan_deg",
            "ecc",
            "pa_deg",
            "mean_anomaly_deg",
            "mean_motion_rev_day",
            "epoch_rev",
            "check_sum",
        ]

        data = {key: [] for key in cols}

        for raw in f:
            line = raw.rstrip("\r\n")
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")
            if line == "":
                continue

            key = line[0]
            rest = line[1:]
            splits = split_line(rest)
            vprint(splits)

            if key == "0":
                data["name"].append(rest.strip())

            elif key == "1":
                if not splits:
                    continue
                data["satellite#"].append(splits[0][:-1])
                data["classification"].append(splits[0][-1])

                has_intl = len(splits) > 1 and looks_like_intl(splits[1])
                i = 0 if has_intl else -1

                if has_intl:
                    data["intl_designator"].append(splits[1 + i])
                    data["launch_year"].append(_century_year(splits[1 + i][:2]))
                    data["launch_number"].append(int(splits[1 + i][2:5]))
                    data["launch_piece"].append(splits[1 + i][5:])
                else:
                    data["intl_designator"].append(None)
                    data["launch_year"].append(None)
                    data["launch_number"].append(None)
                    data["launch_piece"].append(None)

                data["elset_epoch"].append(splits[2 + i])
                ey, ed = parse_epoch_fields(splits[2 + i])
                data["epoch_year"].append(ey)
                data["epoch_day"].append(ed)

                denom = _days_in_year(int(ey))
                dec_year = float(ey) + (float(ed) - 1.0) / float(denom)
                data["decimal_year"].append(dec_year)

                data["ndot"].append(float(splits[3 + i]))
                data["nddot"].append(split_mantassa(splits[4 + i]))
                data["drag"].append(split_mantassa(splits[5 + i]))
                data["elset_type"].append(int(splits[6 + i]))
                data["elset#"].append(int(splits[7 + i]))

            elif key == "2":
                # Use fixed-width slices per TLE spec; token splits are brittle here.
                if len(line) < 69:
                    continue
                data["inc_deg"].append(float(line[8:16]))
                data["raan_deg"].append(float(line[17:25]))
                data["ecc"].append(parse_ecc(line[26:33]))
                data["pa_deg"].append(float(line[34:42]))
                data["mean_anomaly_deg"].append(float(line[43:51]))
                data["mean_motion_rev_day"].append(float(line[52:63]))
                data["epoch_rev"].append(int(line[63:68]))
                data["check_sum"].append(int(line[68]))
            else:
                pass

    if not makedf:
        return data

    # ---------- Build DataFrame (align primarily on the count/order of line '1') ----------
    base_n = len(data["satellite#"])

    def _get(k, i, default=np.nan):
        lst = data.get(k, [])
        return lst[i] if i < len(lst) else default

    rows = []
    for i in range(base_n):
        row = {k: _get(k, i) for k in cols}
        rows.append(row)

    df = pd.DataFrame(rows, columns=cols)

    # -------------------- Time convenience --------------------
    # The TLE epoch is a UTC day of year: Jan 1 00:00 UTC plus (day - 1) UTC
    # days. astropy's "decimalyear" spreads a year's leap second over the whole
    # year, which put 2008 epochs (leap second on Dec 31) up to 1 s late --
    # 0.72 s, or 5.5 km along-track for the ISS, at day 264.
    years = pd.to_numeric(df["epoch_year"], errors="coerce").to_numpy()
    days = pd.to_numeric(df["epoch_day"], errors="coerce").to_numpy()
    gps = np.full(years.shape, np.nan)
    mask = np.isfinite(years) & np.isfinite(days)
    if np.any(mask):
        jan1 = Time([f"{int(y):04d}-01-01T00:00:00" for y in years[mask]], scale="utc")
        gps[mask] = Time(jan1.jd1, jan1.jd2 + (days[mask] - 1.0), format="jd", scale="utc").gps
    df["epoch_gps"] = gps

    # -------------------- Orbital elements and derived quantities (SI) --------------------
    # Mean motion: TLE "mean_motion_rev_day" is rev/day
    df["mean_motion_rev_day"] = pd.to_numeric(df["mean_motion_rev_day"], errors="coerce")
    n_rad_s = df["mean_motion_rev_day"] * (TWOPI / DAY_S)
    df["n_rad_s"] = n_rad_s
    df["period_s"] = TWOPI / n_rad_s

    # Semi-major axis (m) via Kepler's 3rd law
    df["a_m"] = (EARTH_MU / (n_rad_s ** 2)) ** (1.0 / 3.0)

    # Eccentricity
    e = pd.to_numeric(df["ecc"], errors="coerce")

    # Semi-latus rectum and apsides (m)
    df["p_m"]  = df["a_m"] * (1.0 - e ** 2)
    df["rp_m"] = df["a_m"] * (1.0 - e)
    df["ra_m"] = df["a_m"] * (1.0 + e)
    df["hp_m"] = df["rp_m"] - EARTH_RADIUS
    df["ha_m"] = df["ra_m"] - EARTH_RADIUS

    # Angles in radians
    df["inc_rad"]  = np.radians(pd.to_numeric(df["inc_deg"], errors="coerce"))
    df["raan_rad"] = np.radians(pd.to_numeric(df["raan_deg"], errors="coerce"))
    df["argp_rad"] = np.radians(pd.to_numeric(df["pa_deg"],   errors="coerce"))
    df["M_rad"]    = np.radians(pd.to_numeric(df["mean_anomaly_deg"], errors="coerce"))

    # Solve Kepler's equation for E and true anomaly nu
    M = df["M_rad"].to_numpy(dtype=float)
    e_arr = e.to_numpy(dtype=float)

    def solve_kepler_E(M_rad, ecc, tol=1e-12, max_iter=50):
        M0 = np.mod(M_rad, TWOPI)
        E = np.where(ecc < 0.8, M0, np.pi * np.ones_like(M0))
        mask = np.isfinite(M0) & np.isfinite(ecc)
        if not np.any(mask):
            return np.full_like(M0, np.nan)
        E_work = E.copy()
        for _ in range(max_iter):
            f = E_work - ecc * np.sin(E_work) - M0
            fp = 1.0 - ecc * np.cos(E_work)
            delta = -f / fp
            E_work = E_work + delta
            if np.all(np.abs(delta[mask]) < tol):
                break
        E[mask] = E_work[mask]
        E[~mask] = np.nan
        return E

    E = solve_kepler_E(M, e_arr)
    df["E_rad"] = E
    df["E_deg"] = np.degrees(E)

    denom = (1.0 - e_arr * np.cos(E))
    sinv = np.sqrt(1.0 - e_arr**2) * np.sin(E) / denom
    cosv = (np.cos(E) - e_arr) / denom
    nu = np.arctan2(sinv, cosv)
    df["nu_rad"] = nu
    df["nu_deg"] = np.degrees(nu)
    df["ta_rad"] = df["nu_rad"]
    df["ta_deg"] = df["nu_deg"]

    # Argument of latitude
    df["u_rad"] = df["argp_rad"] + df["nu_rad"]
    df["u_deg"] = np.degrees(df["u_rad"])

    # Magnitudes at epoch (two-body)
    r_mag = df["p_m"] / (1.0 + e * np.cos(nu))                       # m
    v_mag = np.sqrt(EARTH_MU * (2.0 / r_mag - 1.0 / df["a_m"]))      # m/s
    df["r_mag_m"] = r_mag
    df["v_mag_m_s"] = v_mag

    # Perifocal state (m, m/s) with small-denominator guard
    cosv_arr, sinv_arr = np.cos(nu), np.sin(nu)
    p = df["p_m"].to_numpy(dtype=float)
    denom_pf = 1.0 + e_arr * cosv_arr
    denom_pf = np.where(np.abs(denom_pf) < 1e-15, np.sign(denom_pf) * 1e-15, denom_pf)
    r_pf_x = p * cosv_arr / denom_pf
    r_pf_y = p * sinv_arr / denom_pf
    r_pf_z = np.zeros_like(r_pf_x)
    sqrt_mu_p = np.sqrt(EARTH_MU / p)
    v_pf_x = -sqrt_mu_p * sinv_arr
    v_pf_y =  sqrt_mu_p * (e_arr + cosv_arr)
    v_pf_z = np.zeros_like(v_pf_x)

    # Rotation 3-1-3 to ECI
    O, i, w = df["raan_rad"].to_numpy(), df["inc_rad"].to_numpy(), df["argp_rad"].to_numpy()
    cO, sO = np.cos(O), np.sin(O)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(w), np.sin(w)

    R11 = cO*cw - sO*sw*ci
    R12 = -cO*sw - sO*cw*ci
    R13 = sO*si
    R21 = sO*cw + cO*sw*ci
    R22 = -sO*sw + cO*cw*ci
    R23 = -cO*si
    R31 = sw*si
    R32 = cw*si
    R33 = ci

    df["r_eci_m_x"]   = R11*r_pf_x + R12*r_pf_y + R13*r_pf_z
    df["r_eci_m_y"]   = R21*r_pf_x + R22*r_pf_y + R23*r_pf_z
    df["r_eci_m_z"]   = R31*r_pf_x + R32*r_pf_y + R33*r_pf_z

    df["v_eci_m_s_x"] = R11*v_pf_x + R12*v_pf_y + R13*v_pf_z
    df["v_eci_m_s_y"] = R21*v_pf_x + R22*v_pf_y + R23*v_pf_z
    df["v_eci_m_s_z"] = R31*v_pf_x + R32*v_pf_y + R33*v_pf_z

    # -------------------- SGP4 recovery (un-Kozai) and SGP4-ready columns --------------------
    # Inputs
    e_ = pd.to_numeric(df["ecc"], errors="coerce").to_numpy()
    i_ = df["inc_rad"].to_numpy()
    # TLE mean motion: rev/day -> rad/min
    n_tle_rad_min = (df["mean_motion_rev_day"].to_numpy() * 2.0*np.pi) / 1440.0

    # SGP4 constants: xke = sqrt(mu)*60 / R_earth^(3/2) [1/min]
    xke = np.sqrt(EARTH_MU) * 60.0 / (EARTH_RADIUS**1.5)
    # SGP4 uses k2 = J2/2 in Earth-radii units; appear in compact form below
    k2 = 0.5 * J2_wgs

    cosi   = np.cos(i_)
    theta2 = cosi*cosi
    x3thm1 = 3.0*theta2 - 1.0
    e2     = e_*e_
    betao2 = 1.0 - e2
    betao  = np.sqrt(betao2)

    valid = np.isfinite(n_tle_rad_min) & (n_tle_rad_min > 0) & np.isfinite(e_) & (betao2 > 0)

    a1   = np.full_like(n_tle_rad_min, np.nan)
    ao   = np.full_like(n_tle_rad_min, np.nan)
    n0   = np.full_like(n_tle_rad_min, np.nan)

    if np.any(valid):
        a1v  = (xke / n_tle_rad_min[valid])**(2.0/3.0)  # Earth-radii
        del1 = 1.5 * k2 * x3thm1[valid] / (a1v*a1v * betao[valid]*betao2[valid])  # 0.75*J2/(a1^2*beta^3)
        aov  = a1v * (1.0 - del1*(1.0/3.0 + del1*(1.0 + 134.0*del1/81.0)))
        del0 = 1.5 * k2 * x3thm1[valid] / (aov*aov * betao[valid]*betao2[valid])
        n0v  = n_tle_rad_min[valid] / (1.0 + del0)
        a1[valid] = a1v
        ao[valid] = aov
        n0[valid] = n0v

    a0_er = np.where(np.isfinite(n0), (xke / n0)**(2.0/3.0), np.nan)  # Earth-radii
    a_sgp4 = a0_er * EARTH_RADIUS                                     # meters
    n_sgp4 = n0 / 60.0                                                # rad/s

    # Store SGP4-ready columns
    df["a_sgp4"]    = a_sgp4
    df["e_sgp4"]    = pd.to_numeric(df["ecc"], errors="coerce")
    df["i_sgp4"]    = df["inc_rad"]
    df["pa_sgp4"]   = df["argp_rad"]
    df["raan_sgp4"] = df["raan_rad"]
    df["M_sgp4"]    = df["M_rad"]      # what SGP4 expects
    df["ta_sgp4"]   = df["nu_rad"]     # convenience; not used by SGP4
    df["n_sgp4"]    = n_sgp4           # recovered mean motion (rad/s)

    return df
