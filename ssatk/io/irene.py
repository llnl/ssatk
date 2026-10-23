"""IRENE V1.58 mean flux grids and SHIELDOSE2 absorbed-dose outputs.

The official runtime and model databases are supplied separately by AFRL:
https://www.vdl.afrl.af.mil/programs/ae9ap9/downloads.php
No surrogate flux or dose is substituted when IRENE is unavailable.
"""

from pathlib import Path
import csv
import io
import json
import os
import re
import shutil
import subprocess

import numpy as np
from astropy.time import Time

from ssatk.coordinates.earth_fixed import gcrf_to_itrf_astropy

__all__ = ["find_irene_runtime", "prepare_irene_run", "prepare_irene_flux_grid",
           "run_irene", "read_irene_dose", "read_irene_flux_grid"]


def find_irene_runtime(*, executable=None, model_data=None):
    """Return validated executable/database paths, or None when unconfigured.

    Explicit arguments override IRENE_EXECUTABLE / IRENE_DB. IRENE_HOME may
    point at the extracted Irene directory. Otherwise search PATH.
    Incomplete or ambiguous installations raise instead of silently skipping.
    """
    executable = executable or os.environ.get("IRENE_EXECUTABLE")
    model_data = model_data or os.environ.get("IRENE_DB")
    home = os.environ.get("IRENE_HOME")
    if home:
        home = Path(home).expanduser()
        if not executable:
            name = "CmdLineIrene.exe" if os.name == "nt" else "CmdLineIrene"
            candidates = [path for path in home.rglob(name) if path.is_file()]
            if len(candidates) != 1:
                raise FileNotFoundError("IRENE_HOME must contain one runtime; set IRENE_EXECUTABLE explicitly")
            executable = candidates[0]
        model_data = model_data or home / "modelData"
    executable = executable or shutil.which("CmdLineIrene")
    if not executable:
        if model_data:
            raise FileNotFoundError("Official CmdLineIrene runtime missing; set IRENE_EXECUTABLE")
        return None
    executable = Path(executable).expanduser().resolve(strict=True)
    if not executable.is_file():
        raise FileNotFoundError(f"IRENE executable is not a file: {executable}")
    model_data = model_data or executable.parent.parent.parent / "modelData"
    database = Path(model_data).expanduser().resolve(strict=True)
    for name in ("AE9V15_runtime_tables.mat", "AP9V15_runtime_tables.mat", "igrfDB.h5", "sd2DB.h5"):
        if not (database / name).is_file():
            raise FileNotFoundError(f"IRENE database missing: {name}")
    return executable, database


def prepare_irene_run(r_m, times, directory):
    """Write GEO/km/MJD ephemeris and AE9/AP9 mean-dose input files.

    Returns a new run directory; existing runs are never overwritten. Settings
    are an illustrative 4-pi aluminum sphere, 1/3/10 mm, with a tissue detector.
    These are absorbed dose assumptions, not an Orion shielding model.
    """
    positions = np.asarray(r_m, dtype=float)
    times = Time(times, format="gps" if np.asarray(times).dtype.kind in "iuf" else None)
    mjd = np.atleast_1d(times.utc.mjd)
    if positions.shape != (len(mjd), 3) or len(mjd) < 2:
        raise ValueError("r_m must have shape (N, 3), with N >= 2 matching times")
    if not np.isfinite(positions).all() or not np.isfinite(mjd).all() or np.any(np.diff(mjd) <= 0):
        raise ValueError("Positions and times must be finite, with strictly increasing times")
    if np.any((mjd < 33282) | (mjd >= 69807)):
        raise ValueError("IRENE supports dates from 1950 through 2049")
    geo_km = gcrf_to_itrf_astropy(positions, times) / 1000
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savetxt(directory / "orbit.csv", np.column_stack([mjd, geo_km]),
               delimiter=",", fmt="%.12f", header="MJD UTC, GEO X km, GEO Y km, GEO Z km")
    for model, low, high in (("AE9", 0.04, 10), ("AP9", 0.1, 2000)):
        energies = ",".join(f"{energy:.8g}" for energy in np.geomspace(low, high, 100))
        (directory / f"{model}.input.txt").write_text(
            f"ModelType: {model}\nModelDBDir: $IRENE_DB\n"
            f"OrbitFile: orbit.csv\nOutFile: {model}.txt\n"
            "TimeSpec: MJD\nInTimeSpec: MJD\nCoordSys: GEO\nInCoordSys: GEO\n"
            "CoordUnits: km\nInCoordUnits: km\nDataDelim: Comma\nInDataDelim: Comma\n"
            f"FluxType: 1PtDiff\nEnergies: {energies}\nFluxOut: Mean\n"
            "AccumMode: Cumul\nDoseRateOut: True\nDoseAccumOut: True\n"
            "DoseDepths: 1,3,10\nDoseDepthUnits: mm\nDoseDetector: tissue\n"
            "DoseGeometry: spherical4pi\nDoseWithBrems: True\nDoseAttnMode: nuclearandneutrons\n",
            encoding="utf-8",
        )
    metadata = {
        "model": "IRENE AE9/AP9", "mode": "mean", "dose_engine": "SHIELDOSE2",
        "detector": "tissue", "shield": "aluminum", "depths_mm": [1, 3, 10],
        "geometry": "spherical4pi", "ephemeris_frame": "GEO", "distance_unit": "km",
        "time_scale": "UTC", "time_format": "MJD", "start_mjd": float(mjd[0]),
        "samples": len(mjd), "excluded": ["solar energetic particles", "galactic cosmic rays"],
    }
    (directory / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def prepare_irene_flux_grid(geo_m, epoch, directory):
    """Prepare mean integral flux from GEO positions in metres and GPS/Time.

    AE9 electrons above 1 MeV and AP9 protons above 10 MeV are evaluated
    independently of the plotted IGRF/T96 field. No dose is computed on a grid.
    Positions are converted to kilometres only in the IRENE ephemeris file.
    """
    positions = np.asarray(geo_m, dtype=float)
    time = Time(epoch, format="gps" if np.asarray(epoch).dtype.kind in "iuf" else None).utc
    if not time.isscalar or not np.isfinite(time.mjd) or not 33282 <= time.mjd < 69807:
        raise ValueError("epoch must be one finite time from 1950 through 2049")
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError("geo_m must be a finite (N, 3) Cartesian grid")
    time.precision = 9
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savetxt(directory / "orbit.csv", np.column_stack([np.full(len(positions), time.mjd), positions / 1000]),
               delimiter=",", fmt="%.12f", header="MJD UTC, GEO X km, GEO Y km, GEO Z km")
    for model, energy in (("AE9", 1), ("AP9", 10)):
        (directory / f"{model}.input.txt").write_text(
            f"ModelType: {model}\nModelDBDir: $IRENE_DB\n"
            f"OrbitFile: orbit.csv\nOutFile: {model}.txt\n"
            "TimeSpec: MJD\nInTimeSpec: MJD\nCoordSys: GEO\nInCoordSys: GEO\n"
            "CoordUnits: km\nInCoordUnits: km\nDataDelim: Comma\nInDataDelim: Comma\n"
            f"FluxType: Integral\nEnergies: {energy}\nFluxOut: Mean\n", encoding="utf-8")
    metadata = {"product": "integral_flux_grid", "model": "IRENE AE9/AP9", "mode": "mean",
                "ephemeris_frame": "GEO", "distance_unit": "km", "time_scale": "UTC",
                "time_format": "MJD", "epoch_utc": time.isot + "Z", "samples": len(positions),
                "electron_threshold_mev": 1, "proton_threshold_mev": 10,
                "flux_unit": "cm^-2 s^-1", "excluded": ["solar energetic particles", "galactic cosmic rays"]}
    (directory / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return directory


def run_irene(directory, *, executable=None, model_data=None, timeout=600):
    """Run both models with the official CmdLineIrene executable (no shell).

    Uses find_irene_runtime for explicit, environment or PATH configuration.
    A failure preserves the input and logs and never returns fabricated results.
    """
    runtime = find_irene_runtime(executable=executable, model_data=model_data)
    if runtime is None:
        raise FileNotFoundError("Official CmdLineIrene runtime missing; obtain IRENE from AFRL downloads")
    executable, database = runtime
    directory = Path(directory).resolve(strict=True)
    if any(directory.glob("*_mean_*.txt")) or any(
        (directory / name).exists() for name in ("AE9.log", "AP9.log", "dose.json", "flux.json", "dose_vs_mission_day.csv")
    ):
        raise FileExistsError("IRENE outputs already exist; prepare a fresh run directory")
    for name in ("AE9.input.txt", "AP9.input.txt", "run.json", "orbit.csv"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"IRENE input missing: {name}")
    env = dict(os.environ, IRENE_DB=str(database))
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    for model in ("AE9", "AP9"):
        with (directory / f"{model}.log").open("w", encoding="utf-8") as log:
            subprocess.run([str(executable), "-i", f"{model}.input.txt"], cwd=directory,
                           env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    return (read_irene_flux_grid(directory) if metadata.get("product") == "integral_flux_grid"
            else read_irene_dose(directory))


def read_irene_flux_grid(directory):
    """Read AE9/AP9 mean integral flux with GEO positions in metres.

    Validate grid, units and model headers; metadata describes on-disk units.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if (metadata.get("product"), metadata.get("electron_threshold_mev"),
        metadata.get("proton_threshold_mev")) != ("integral_flux_grid", 1, 10):
        raise ValueError("Expected AE9 >1 MeV / AP9 >10 MeV integral flux grid metadata")
    expected_metadata = {"model": "IRENE AE9/AP9", "mode": "mean", "ephemeris_frame": "GEO",
                         "distance_unit": "km", "time_scale": "UTC", "time_format": "MJD",
                         "flux_unit": "cm^-2 s^-1"}
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError("Expected mean integral GEO/km/UTC flux metadata")
    orbit = np.loadtxt(directory / "orbit.csv", delimiter=",", ndmin=2)
    if orbit.shape[1] != 4 or not len(orbit) or not np.isfinite(orbit).all() or not np.all(orbit[:, 0] == orbit[0, 0]):
        raise ValueError("Expected finite single-epoch MJD/GEO XYZ grid")
    epoch = Time(metadata.get("epoch_utc")).utc
    if (metadata.get("samples") != len(orbit) or not epoch.isscalar or not np.isfinite(epoch.mjd)
            or not np.allclose(orbit[:, 0], epoch.mjd, rtol=0, atol=1e-8)):
        raise ValueError("Flux grid does not match metadata samples or epoch")
    fluxes = []
    for model, species, energy in (("AE9", "e-", "1"), ("AP9", "H+", "10")):
        path = directory / f"{model}_mean_flux.txt"
        content = path.read_text(encoding="utf-8")
        header = "\n".join(line for line in content.splitlines() if line.lstrip().startswith("#"))
        expected = {"Model type": model, "Particle species": species, "Flux type": "Integral",
                    "Flux direction": "Omnidirectional", "Energy levels (MeV)": energy,
                    "Time format": "Modified Julian Date",
                    "Coordinate system": "GEO (Geographic) Cartesian in km"}
        for label, value in expected.items():
            match = re.search(r"^#\s*" + re.escape(label) + r":\s*(.*?)\s*$", header, re.M | re.I)
            if match is None or " ".join(match[1].split()).lower() != value.lower():
                raise ValueError(f"{path.name}: unexpected {label}")
        if "flux(#/cm^2/s)" not in header:
            raise ValueError(f"{path.name}: expected integral flux units #/cm^2/s")
        version = re.search(r"Irene Software Version:\s*(\S+)", header, re.I)
        if version is None or ("version" in metadata and metadata["version"] != version[1]):
            raise ValueError(f"{path.name}: missing or mismatched IRENE version")
        metadata["version"] = version[1]
        rows = np.loadtxt(io.StringIO(content), delimiter=",", ndmin=2)
        serialized = np.column_stack([np.round(orbit[:, 0], 8), np.round(orbit[:, 1:], 6)])
        if rows.shape != (len(orbit), 5) or not np.isfinite(rows).all() or not np.array_equal(rows[:, :4], serialized):
            raise ValueError(f"{path.name}: invalid flux rows or mismatched input grid")
        if np.any(rows[:, 4] < 0):
            raise ValueError(f"{path.name}: negative or out-of-domain flux")
        fluxes.append(rows[:, 4].tolist())
    result = {"geo_m": (orbit[:, 1:] * 1000).tolist(), "electron_flux": fluxes[0],
              "proton_flux": fluxes[1], "metadata": metadata}
    (directory / "flux.json").write_text(json.dumps(result), encoding="utf-8")
    return result


def read_irene_dose(directory):
    """Read the controlled MJD/GEO/three-depth cumulative dose output format.

    Reject incompatible units, negative values, mismatched ephemerides, missing
    samples and invalid/out-of-domain results. Returns rad and mGy, never mSv.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    orbit = np.loadtxt(directory / "orbit.csv", delimiter=",", comments="#", ndmin=2)
    if orbit.shape[1] != 4 or len(orbit) < 2 or not np.isfinite(orbit).all() or np.any(np.diff(orbit[:, 0]) <= 0):
        raise ValueError("Invalid input ephemeris: expected finite, increasing MJD/GEO XYZ rows")
    if metadata.get("depths_mm") != [1, 3, 10]:
        raise ValueError("Run metadata must specify depths_mm [1, 3, 10]")
    doses = []
    for model in ("AE9", "AP9"):
        path = directory / f"{model}_mean_doseaccum.txt"
        content = path.read_text(encoding="utf-8")
        header = "\n".join(line for line in content.splitlines() if line.lstrip().startswith("#"))
        if not re.search(r"\brads?\b", header, re.I) or re.search(r"\b[mk]rad\b|rad\s*/\s*s|\bmSv\b", header, re.I):
            raise ValueError(f"{path.name}: expected cumulative absorbed dose in rad")
        expected_headers = {
            "Model type": model, "Accumulation mode": "Cumulative",
            "Particle species": "e-" if model == "AE9" else "H+",
            "Flux type": "1PtDiff", "Flux direction": "Omnidirectional",
            "Dose calculation method": "ShieldDose2 Model", "Shielding geometry": "Spherical4pi",
            "Shielding depths (mm)": "1 3 10", "Detector type": "Tissue",
            "Time format": "Modified Julian Date",
            "Nuclear atten. mode": "NuclearAndNeutrons",
            "Coordinate system": "GEO (Geographic) Cartesian in km",
        }
        for label, expected in expected_headers.items():
            match = re.search(r"^#\s*" + re.escape(label) + r":\s*(.*?)\s*$", header, re.M | re.I)
            if match is None or " ".join(match[1].split()).lower() != expected.lower():
                raise ValueError(f"{path.name}: unexpected {label}")
        version = re.search(r"Irene Software Version:\s*(\S+)", header, re.I)
        if version is None or ("version" in metadata and metadata["version"] != version[1]):
            raise ValueError(f"{path.name}: missing or mismatched IRENE version")
        metadata["version"] = version[1]
        rows = np.loadtxt(io.StringIO(content), delimiter=",", comments="#", ndmin=2)
        if rows.shape != (len(orbit), 7) or not np.isfinite(rows).all():
            raise ValueError(f"{path.name}: expected finite MJD/GEO XYZ/three-depth dose rows")
        # IRENE V1.58 prints MJD to 8 decimals and GEO/km positions to 6.
        serialized_orbit = np.column_stack([np.round(orbit[:, 0], 8), np.round(orbit[:, 1:4], 6)])
        if not np.array_equal(rows[:, :4], serialized_orbit):
            raise ValueError(f"{path.name}: output does not match input ephemeris")
        dose = rows[:, 4:]
        # The official AP9 run showed a one-digit reversal in its %.8e dose output.
        magnitude = np.maximum(np.abs(dose[:-1]), np.abs(dose[1:]))
        resolution = 10.0 ** (np.floor(np.log10(np.maximum(magnitude, np.finfo(float).tiny))) - 8)
        if np.any(dose < 0) or np.any(np.diff(dose, axis=0) < -resolution * (1 + np.finfo(float).eps)):
            raise ValueError(f"{path.name}: dose must be nonnegative and cumulative")
        doses.append(dose)
    total_rad = doses[0] + doses[1]
    result = {
        "mission_day": (orbit[:, 0] - orbit[0, 0]).tolist(),
        "depths_mm": metadata["depths_mm"], "electron_rad": doses[0].tolist(),
        "proton_rad": doses[1].tolist(), "total_rad": total_rad.tolist(),
        "total_mgy": (total_rad * 10).tolist(), "metadata": metadata,
    }
    with (directory / "dose_vs_mission_day.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["mission_day", "dose_1mm_mGy", "dose_3mm_mGy", "dose_10mm_mGy"])
        writer.writerows(zip(result["mission_day"], *np.asarray(result["total_mgy"]).T))
    (directory / "dose.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
