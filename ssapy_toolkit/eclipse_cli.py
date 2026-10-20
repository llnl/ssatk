"""Canonical command-line interface for V22.2 eclipse products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_STATE_BACKENDS = ("reference", "auto", "ssapy-core", "ssapy")
_SEARCH_BACKENDS = ("auto", "ssapy", "ssapy-core", "swisseph", "analytic", "catalog")
_MATERIALIZE_BACKENDS = ("auto", "ssapy", "ssapy-core", "swisseph", "analytic")
_QUALITIES = ("motion", "motion_high", "high", "ultra", "cinema")


def _state_backend_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend", choices=_STATE_BACKENDS, default="auto",
        help=("state driver: auto prefers SSAPy+Toolkit; ssapy is strict full LLNL; "
              "ssapy-core is strict ephemeris/orientation with Astropy frames; "
              "reference uses the validated NASA/GSFC reconstruction"),
    )


def _event_arguments(parser: argparse.ArgumentParser, *, optional_kind: bool = False) -> None:
    parser.add_argument("kind", choices=("solar", "lunar"), nargs="?" if optional_kind else None,
                        default="solar" if optional_kind else None)
    parser.add_argument("--samples", type=int, default=121, help="minimum solver-state count; exact eclipse contacts are inserted and may increase the resolved count")
    parser.add_argument("--solar-scope", choices=("local", "global"), default="global")
    _state_backend_argument(parser)


def _observer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--observer-lat", type=float)
    parser.add_argument("--observer-lon", type=float)
    parser.add_argument("--observer-elevation-m", type=float, default=0.0)
    parser.add_argument("--pressure-hpa", type=float, default=1010.0)
    parser.add_argument("--temperature-c", type=float, default=10.0)
    parser.add_argument("--relative-humidity-pct", type=float, default=0.0)
    parser.add_argument("--wavelength-um", type=float, default=0.55)
    parser.add_argument("--weather-file", type=Path,
                        help="JSON/CSV pressure, temperature, humidity, and wavelength provenance")
    parser.add_argument("--no-refraction", action="store_true")
    parser.add_argument("--horizon-profile", type=Path)
    parser.add_argument("--lunar-limb-profile", type=Path)


def _observer_from_args(args):
    if getattr(args, "observer_lat", None) is None and getattr(args, "observer_lon", None) is None:
        return None
    if args.observer_lat is None or args.observer_lon is None:
        raise SystemExit("--observer-lat and --observer-lon must be supplied together")
    from ssapy_toolkit.coordinates.eclipse_observer_geometry import HorizonProfile, ObserverConfig
    horizon = HorizonProfile.from_file(args.horizon_profile) if args.horizon_profile else None
    fields = dict(
        latitude_deg=args.observer_lat,
        longitude_east_deg=args.observer_lon,
        elevation_m=args.observer_elevation_m,
        pressure_hpa=args.pressure_hpa,
        temperature_c=args.temperature_c,
        relative_humidity_pct=args.relative_humidity_pct,
        wavelength_um=args.wavelength_um,
        apply_refraction=not args.no_refraction,
        horizon=horizon,
        name="CLI WGS-84 observer",
    )
    if args.weather_file:
        return ObserverConfig.from_weather_file(args.weather_file, **fields)
    return ObserverConfig(**fields)


def _limb_from_args(args):
    if not getattr(args, "lunar_limb_profile", None):
        return None
    from ssapy_toolkit.coordinates.eclipse_lunar_geometry import LunarLimbProfile
    return LunarLimbProfile.from_file(args.lunar_limb_profile)


def _event_request_from_args(args):
    from ssapy_toolkit.eclipse_api_impl import EventRequest
    return EventRequest(
        kind=args.kind,
        backend=args.backend,
        sample_count=args.samples,
        solar_scope=args.solar_scope,
        observer=_observer_from_args(args),
        lunar_limb_profile=_limb_from_args(args),
    )


# Default output locations, resolved lazily in main(). Relative defaults such
# as Path("output/selected_eclipse") were written into whatever directory the
# command happened to run from, which for a repo checkout means the repo. An
# empty string means the subcommand writes a directory rather than a file.
_DEFAULT_OUTPUT = {
    "audit": "ssapy_runtime_audit.json",
    "data-audit": "ssapy_data_asset_audit.json",
    "promote": "ssapy_promotion_report.json",
    "validate": "eclipse_validation.json",
    "scientific": "",
}


def _resolve_default_output(args) -> None:
    """Point an unset --output at the figure gallery.

    figpath is imported here rather than at module scope because it lives
    under ssapy_toolkit.plots, whose package __init__ imports every plotting
    module. Deferring it keeps --help and any run with an explicit --output
    from paying that cost.
    """
    name = _DEFAULT_OUTPUT.get(getattr(args, "command", None))
    if name is None or getattr(args, "output", None) is not None:
        return
    from ssapy_toolkit.plots.figpath import figpath
    base = "demo_gallery/figures/eclipse"
    args.output = Path(figpath(f"{base}/{name}" if name else base))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssapy-eclipse",
        description=("Discover, build, validate, and render finite-Sun eclipses with immutable "
                     "ephemeris/frame provenance."),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="probe SSAPy/Toolkit and write a JSON audit")
    audit.add_argument("--output", type=Path, default=None,
                       help="default: <figures>/demo_gallery/figures/eclipse/ssapy_runtime_audit.json")

    data_audit = sub.add_parser(
        "data-audit", help="resolve SSAPy-Data images/kernels and write provenance"
    )
    data_audit.add_argument(
        "--policy", choices=("data-first", "strict-data", "package-first", "packaged-only"),
        default="data-first",
    )
    data_audit.add_argument("--output", type=Path, default=None,
                            help="default: <figures>/demo_gallery/figures/eclipse/ssapy_data_asset_audit.json")

    data_resolve = sub.add_parser("data-resolve", help="resolve one logical SSAPy-Data asset")
    data_resolve.add_argument(
        "asset", choices=("earth_albedo", "moon_albedo", "de430_kernel", "moon_pa_kernel",
                          "earth_clouds", "earth_normal", "moon_normal")
    )
    data_resolve.add_argument(
        "--policy", choices=("data-first", "strict-data", "package-first", "packaged-only"),
        default="data-first",
    )
    data_resolve.add_argument("--required", action="store_true")
    data_resolve.add_argument("--output", type=Path)

    promote = sub.add_parser("promote", help="execute the strict SSAPy promotion gate")
    promote.add_argument("--output", type=Path, default=None,
                         help="default: <figures>/demo_gallery/figures/eclipse/ssapy_promotion_report.json")
    promote.add_argument("--samples", type=int, default=41)
    promote.add_argument("--require-strict", action="store_true")

    strict_acceptance = sub.add_parser(
        "strict-acceptance",
        help="run the complete no-fallback LLNL acceptance suite on the destination system",
    )
    strict_acceptance.add_argument("--output-dir", type=Path, required=True)
    strict_acceptance.add_argument("--samples", type=int, default=61)
    strict_acceptance.add_argument("--render", action="store_true")
    strict_acceptance.add_argument("--quality", choices=_QUALITIES, default="high")
    strict_acceptance.add_argument("--require-strict", action="store_true")

    search = sub.add_parser("search", help="discover arbitrary solar/lunar eclipses in a UTC interval")
    search.add_argument("--start", required=True, help="UTC date/time or Julian date")
    search.add_argument("--end", required=True, help="UTC date/time or Julian date")
    search.add_argument("--kind", choices=("all", "solar", "lunar"), default="all")
    search.add_argument("--backend", choices=_SEARCH_BACKENDS, default="auto")
    search.add_argument("--output", type=Path, required=True)

    corpus = sub.add_parser("reference-corpus", help="write the bundled NASA/GSFC 2021-2030 regression corpus")
    corpus.add_argument("--kind", choices=("all", "solar", "lunar"), default="all")
    corpus.add_argument("--output", type=Path, required=True)

    validate_corpus = sub.add_parser("validate-corpus", help="compare an independent discovery backend with the NASA/GSFC regression corpus")
    validate_corpus.add_argument("--start", default="2021-01-01")
    validate_corpus.add_argument("--end", default="2031-01-01")
    validate_corpus.add_argument("--kind", choices=("all", "solar", "lunar"), default="all")
    validate_corpus.add_argument("--backend", choices=_SEARCH_BACKENDS, default="swisseph")
    validate_corpus.add_argument("--greatest-threshold-s", type=float, default=15.0)
    validate_corpus.add_argument("--magnitude-threshold", type=float, default=0.0012)
    validate_corpus.add_argument("--lunar-duration-threshold-min", type=float, default=0.35)
    validate_corpus.add_argument("--output", type=Path, required=True)

    uncertainty = sub.add_parser("uncertainty", help="propagate a recorded conditional uncertainty budget through a validated eclipse")
    uncertainty.add_argument("kind", choices=("solar", "lunar"))
    uncertainty.add_argument("--samples", type=int, default=512)
    uncertainty.add_argument("--seed", type=int)
    uncertainty.add_argument("--budget", type=Path, help="JSON object overriding UncertaintyBudget fields")
    uncertainty.add_argument("--zero-budget", action="store_true", help="collapse all random terms for a deterministic regression")
    uncertainty.add_argument("--output", type=Path, required=True)

    materialize = sub.add_parser("materialize", help="build a renderer-ready event from a discovery catalog")
    materialize.add_argument("--catalog", type=Path, required=True)
    selector = materialize.add_mutually_exclusive_group(required=True)
    selector.add_argument("--key")
    selector.add_argument("--index", type=int)
    materialize.add_argument("--state-backend", choices=_MATERIALIZE_BACKENDS, default="auto")
    materialize.add_argument("--samples", type=int, default=121)
    materialize.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build", help="build and serialize one bundled reference event")
    _event_arguments(build)
    _observer_arguments(build)
    build.add_argument("--output", type=Path, required=True)

    validate = sub.add_parser("validate", help="validate a serialized or newly built event")
    validate.add_argument("--input", type=Path)
    validate.add_argument("--output", type=Path, default=None,
                          help="default: <figures>/demo_gallery/figures/eclipse/eclipse_validation.json")
    validate.add_argument("--kind", choices=("solar", "lunar"), default="solar")
    validate.add_argument("--samples", type=int, default=61)
    validate.add_argument("--solar-scope", choices=("local", "global"), default="global")
    _state_backend_argument(validate)
    _observer_arguments(validate)

    render = sub.add_parser("render", help="render a built-in or serialized arbitrary event")
    render.add_argument("product", choices=("cinematic", "plotly", "scientific-suite"))
    _event_arguments(render, optional_kind=True)
    _observer_arguments(render)
    render.add_argument("--input-event", type=Path,
                        help="serialized schema 1.7/1.8/1.9/2.0 event; overrides kind/backend/sample arguments")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--quality", choices=_QUALITIES, default="ultra")
    render.add_argument("--animate", action="store_true")
    render.add_argument("--playback-seconds", type=float)
    render.add_argument(
        "--photometry",
        choices=("uniform-disc", "linear-visible", "quadratic-visible"),
        default="quadratic-visible",
        help="surface irradiance law; geometric contacts always remain uniform-disc",
    )

    terrain = sub.add_parser("terrain-horizon", help="derive a WGS-84 horizon from SRTM/DEM terrain")
    terrain.add_argument("--terrain", type=Path, nargs="+", required=True)
    terrain.add_argument("--latitude", type=float, required=True)
    terrain.add_argument("--longitude-east", type=float, required=True)
    terrain.add_argument("--elevation-m", type=float, default=0.0)
    terrain.add_argument("--azimuth-step-deg", type=float, default=1.0)
    terrain.add_argument("--max-distance-km", type=float, default=250.0)
    terrain.add_argument("--radial-samples", type=int, default=700)
    terrain.add_argument("--missing", choices=("skip", "sea-level"), default="skip")
    terrain.add_argument("--terrarium-zoom", type=int)
    terrain.add_argument("--terrarium-x", type=int)
    terrain.add_argument("--terrarium-y", type=int)
    terrain.add_argument("--output", type=Path, required=True)

    limb = sub.add_parser("lola-limb", help="derive an observer-dependent lunar limb from LOLA/LDEM PDS data")
    limb.add_argument("--label", type=Path, required=True)
    limb.add_argument("--image", type=Path)
    limb.add_argument("--event", type=Path, required=True)
    limb.add_argument("--jd-utc", type=float)
    limb.add_argument("--position-angles", type=int, default=1440)
    limb.add_argument("--output", type=Path, required=True)

    fetch = sub.add_parser(
        "fetch-data",
        help="atomically acquire a terrain/LOLA product and write checksum provenance",
    )
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--source-name", required=True)
    fetch.add_argument("--sha256")
    fetch.add_argument("--citation", default="")
    fetch.add_argument("--licence", default="")
    fetch.add_argument("--overwrite", action="store_true")

    browser = sub.add_parser(
        "browser-matrix",
        help="exercise final self-contained HTML products in available WebGL browsers",
    )
    browser.add_argument("html", type=Path, nargs="+")
    browser.add_argument("--output-dir", type=Path, required=True)

    attest = sub.add_parser("attest", help="write one unambiguous stage-aware V22.2 release attestation")
    attest.add_argument("--release-root", type=Path, required=True)
    attest.add_argument("--release", required=True)
    attest.add_argument("--package-version", required=True)
    attest.add_argument(
        "--artifact", action="append", default=[], metavar="ID:STAGE:ROLE:PATH",
        help="repeat for each distribution artifact; artifact paths must be unique",
    )
    attest.add_argument("--authoritative-id", required=True)
    attest.add_argument("--output", type=Path, required=True)

    verify_attest = sub.add_parser("verify-attestation", help="validate an attestation and optionally verify files")
    verify_attest.add_argument("--input", type=Path, required=True)
    verify_attest.add_argument("--artifact-root", type=Path)
    verify_attest.add_argument("--output", type=Path)

    scientific = sub.add_parser(
        "scientific",
        help="generate the selected event's scientific summary, timeline, state, validation, and interactive counterpart",
    )
    _event_arguments(scientific, optional_kind=True)
    _observer_arguments(scientific)
    scientific.add_argument("--input-event", type=Path,
                            help="serialized event; overrides kind/backend/sample arguments")
    scientific.add_argument("--output", type=Path, default=None,
                            help="output DIRECTORY; default "
                                 "<figures>/demo_gallery/figures/eclipse")
    scientific.add_argument("--quality", choices=_QUALITIES, default="high")
    scientific.add_argument("--animate", action="store_true",
                            help="include a contact-aware interactive animation rather than a peak-only 3-D view")
    scientific.add_argument("--playback-seconds", type=float)
    scientific.add_argument(
        "--photometry", choices=("uniform-disc", "linear-visible", "quadratic-visible"),
        default="quadratic-visible",
    )
    scientific.add_argument(
        "--frames", type=int,
        help="deprecated alias for --samples; interpreted as the minimum solver-state count",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_default_output(args)

    if args.command == "audit":
        from ssapy_toolkit.compute.eclipse_capability_audit import write_capability_audit
        print(json.dumps({"audit": write_capability_audit(args.output)}, indent=2)); return 0

    if args.command == "data-audit":
        from ssapy_toolkit.io.eclipse_asset_resolver import write_asset_audit
        output = write_asset_audit(args.output, policy=args.policy)
        payload = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps({
            "asset_audit": str(output.resolve()),
            "rendering_ready": payload["rendering_ready"],
            "strict_ssapy_data_ready": payload["strict_ssapy_data_ready"],
        }, indent=2))
        return 0 if payload["rendering_ready"] else 1

    if args.command == "data-resolve":
        from ssapy_toolkit.io.eclipse_asset_resolver import resolve_asset
        resolved = resolve_asset(args.asset, policy=args.policy, required=args.required)
        payload = resolved.to_dict() if resolved is not None else {
            "logical_name": args.asset, "resolved": False, "policy": args.policy,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        return 0 if resolved is not None else 1

    if args.command == "promote":
        from ssapy_toolkit.compute.eclipse_promotion import write_ssapy_promotion_report
        try:
            output = write_ssapy_promotion_report(args.output, samples=args.samples, require_strict=args.require_strict)
        except Exception as exc:
            print(json.dumps({"promotion_report": str(args.output.expanduser().resolve()),
                              "passed": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
            return 2
        print(json.dumps({"promotion_report": output, "passed": True}, indent=2)); return 0

    if args.command == "strict-acceptance":
        from ssapy_toolkit.compute.eclipse_strict_acceptance import run_strict_acceptance
        try:
            report = run_strict_acceptance(
                args.output_dir, samples=args.samples, render=args.render,
                quality=args.quality, require_strict=args.require_strict,
            )
        except Exception as exc:
            print(json.dumps({
                "acceptance": str(args.output_dir.expanduser().resolve()),
                "passed": False, "error": f"{type(exc).__name__}: {exc}",
            }, indent=2))
            return 2
        print(json.dumps({
            "acceptance": str(args.output_dir.expanduser().resolve()),
            "passed": bool(report.get("passed", False)),
            "status": report.get("status"),
        }, indent=2))
        return 0 if report.get("passed", False) else (2 if args.require_strict else 0)

    if args.command == "reference-corpus":
        from ssapy_toolkit.compute.eclipse_reference_corpus import export_reference_corpus
        output = export_reference_corpus(args.output, mode=args.kind)
        print(json.dumps({"reference_corpus": str(output)}, indent=2)); return 0

    if args.command == "validate-corpus":
        from ssapy_toolkit.compute.eclipse_multi_event_validation import ValidationThresholds, validate_reference_corpus
        thresholds = ValidationThresholds(
            greatest_time_s=args.greatest_threshold_s, magnitude=args.magnitude_threshold,
            lunar_duration_min=args.lunar_duration_threshold_min,
        )
        payload = validate_reference_corpus(
            start=args.start, end=args.end, mode=args.kind, backend=args.backend, thresholds=thresholds,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps({"validation": str(args.output), "passed": payload["passes_release_gate"],
                          "summary": payload["summary"]}, indent=2))
        return 0 if payload["passes_release_gate"] else 1

    if args.command == "uncertainty":
        from ssapy_toolkit.compute.eclipse_uncertainty import UncertaintyBudget, propagate_uncertainty
        if args.zero_budget:
            budget = UncertaintyBudget.zero()
        elif args.budget:
            budget = UncertaintyBudget.from_mapping(json.loads(args.budget.read_text(encoding="utf-8")))
        else:
            budget = UncertaintyBudget()
        seed = args.seed if args.seed is not None else (20240408 if args.kind == "solar" else 20250314)
        payload = propagate_uncertainty(args.kind, samples=args.samples, seed=seed, budget=budget)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n", encoding="utf-8")
        print(json.dumps({"uncertainty": str(args.output), "kind": args.kind,
                          "samples": args.samples, "seed": seed}, indent=2)); return 0

    if args.command == "search":
        from ssapy_toolkit.compute.eclipse_event_search import discover_eclipses, write_discovery_catalog
        records = discover_eclipses(args.start, args.end, kind=args.kind, backend=args.backend)
        output = write_discovery_catalog(records, args.output)
        print(json.dumps({"catalog": output, "count": len(records),
                          "keys": [record.key for record in records]}, indent=2)); return 0

    if args.command == "materialize":
        from ssapy_toolkit.eclipse_api_impl import export_event_state
        from ssapy_toolkit.compute.eclipse_event_search import build_discovered_event, read_discovery_catalog
        records = read_discovery_catalog(args.catalog)
        if args.key is not None:
            matches = [record for record in records if record.key == args.key]
            if len(matches) != 1:
                raise SystemExit(f"catalog contains {len(matches)} records with key {args.key!r}")
            record = matches[0]
        else:
            try:
                record = records[args.index]
            except IndexError as exc:
                raise SystemExit(f"catalog index {args.index} is out of range") from exc
        event = build_discovered_event(record, n_frames=args.samples, state_backend=args.state_backend)
        print(json.dumps({"event_state": export_event_state(event, args.output),
                          "event_key": event.definition.key}, indent=2)); return 0

    if args.command == "build":
        from ssapy_toolkit.eclipse_api_impl import build_event, export_event_state
        event = build_event(_event_request_from_args(args))
        print(json.dumps({"event_state": export_event_state(event, args.output)}, indent=2)); return 0

    if args.command == "validate":
        from ssapy_toolkit.eclipse_api_impl import EventRequest, build_event, load_event_state, validate_event
        event = load_event_state(args.input) if args.input else build_event(EventRequest(
            kind=args.kind, backend=args.backend, sample_count=args.samples,
            solar_scope=args.solar_scope, observer=_observer_from_args(args),
            lunar_limb_profile=_limb_from_args(args),
        ))
        result = validate_event(event, observer=_observer_from_args(args), lunar_limb_profile=_limb_from_args(args))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"validation": str(args.output), "passed": result.passed}, indent=2)); return 0 if result.passed else 1

    if args.command == "render":
        from ssapy_toolkit.eclipse_api_impl import RenderRequest, load_event_state, render_product
        event = load_event_state(args.input_event) if args.input_event else _event_request_from_args(args)
        request = RenderRequest(product=args.product, output=args.output, event=event,
                                quality=args.quality, animate=args.animate,
                                playback_seconds=args.playback_seconds,
                                photometry=args.photometry)
        print(json.dumps({"output": render_product(request)}, indent=2)); return 0

    if args.command == "terrain-horizon":
        from ssapy_toolkit.coordinates.eclipse_terrain_horizon import TerrainMosaic, build_horizon_profile, load_terrain, write_horizon_profile
        options = {}
        if args.terrarium_zoom is not None:
            options = {"zoom": args.terrarium_zoom, "x": args.terrarium_x, "y": args.terrarium_y}
        sources = [load_terrain(path, **options) for path in args.terrain]
        terrain = sources[0] if len(sources) == 1 else TerrainMosaic(sources, source="CLI terrain mosaic")
        profile = build_horizon_profile(
            latitude_deg=args.latitude, longitude_east_deg=args.longitude_east,
            observer_elevation_m=args.elevation_m, terrain=terrain,
            azimuth_step_deg=args.azimuth_step_deg, max_distance_km=args.max_distance_km,
            radial_samples=args.radial_samples, missing=args.missing,
        )
        print(json.dumps({"horizon_profile": write_horizon_profile(profile, args.output),
                          "source": profile.source, "samples": len(profile.azimuth_deg)}, indent=2)); return 0

    if args.command == "lola-limb":
        from ssapy_toolkit.eclipse_api_impl import load_event_state
        from ssapy_toolkit.io.eclipse_lola_limb import LolaGlobalDem, limb_profile_for_event, write_limb_profile
        event = load_event_state(args.event)
        dem = LolaGlobalDem.from_pds(args.label, args.image)
        profile = limb_profile_for_event(event, dem, jd_utc=args.jd_utc,
                                         n_position_angles=args.position_angles)
        print(json.dumps({"lunar_limb_profile": write_limb_profile(profile, args.output),
                          "source": profile.source, "samples": len(profile.position_angle_deg)}, indent=2)); return 0

    if args.command == "fetch-data":
        from ssapy_toolkit.io.eclipse_data_acquisition import acquire_file
        record = acquire_file(
            args.url, args.output, source_name=args.source_name,
            expected_sha256=args.sha256, citation=args.citation,
            licence=args.licence, overwrite=args.overwrite,
        )
        print(json.dumps(record.to_dict(), indent=2)); return 0

    if args.command == "browser-matrix":
        from ssapy_toolkit.plots.eclipse_browser_matrix import main as browser_main
        browser_args = [*(str(path) for path in args.html), "--output-dir", str(args.output_dir)]
        return int(browser_main(browser_args))

    if args.command == "attest":
        from ssapy_toolkit.io.eclipse_attestation import artifact_record, build_attestation, write_attestation
        records = []
        for value in args.artifact:
            try:
                artifact_id, stage, role, path = value.split(":", 3)
            except ValueError as exc:
                raise SystemExit("--artifact must use ID:STAGE:ROLE:PATH") from exc
            source = Path(path).expanduser().resolve()
            records.append(artifact_record(
                source, artifact_id=artifact_id, stage=stage, role=role,
                display_path=source.name,
            ))
        payload = build_attestation(
            release=args.release, package_version=args.package_version,
            release_root=args.release_root, artifacts=records,
            authoritative_distribution=args.authoritative_id,
        )
        print(json.dumps({"attestation": write_attestation(payload, args.output)}, indent=2)); return 0

    if args.command == "verify-attestation":
        from ssapy_toolkit.io.eclipse_attestation import read_attestation, verify_attestation_files, validate_attestation
        payload = read_attestation(args.input)
        result = (verify_attestation_files(payload, artifact_root=args.artifact_root)
                  if args.artifact_root else validate_attestation(payload))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2)); return 0 if result.get("files_passed", True) else 1

    if args.command == "scientific":
        from ssapy_toolkit.eclipse_api_impl import EventRequest, RenderRequest, load_event_state, render_product
        if args.input_event:
            event = load_event_state(args.input_event)
        else:
            minimum = int(args.frames) if args.frames is not None else int(args.samples)
            event = EventRequest(
                kind=args.kind, backend=args.backend, sample_count=minimum,
                solar_scope=args.solar_scope, observer=_observer_from_args(args),
                lunar_limb_profile=_limb_from_args(args),
            )
        result = render_product(RenderRequest(
            product="scientific-suite", output=args.output, event=event,
            quality=args.quality, animate=args.animate,
            playback_seconds=args.playback_seconds, photometry=args.photometry,
        ))
        print(json.dumps({"scientific_suite": result}, indent=2)); return 0

    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
