"""Command-line interface for finite-Sun eclipse events and products."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_STATE_BACKENDS = ("de430", "reference")
_QUALITIES = ("motion", "motion_high", "high", "ultra", "cinema")
_PRODUCTS = ("cinematic", "plotly", "scientific-suite", "observer-suite")


def _event_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("kind", choices=("solar", "lunar"), nargs="?", default="solar")
    parser.add_argument("--samples", type=int, default=121,
                        help="minimum state count; exact contacts are inserted and may add samples")
    parser.add_argument("--solar-scope", choices=("local", "global"), default="global")
    parser.add_argument("--backend", choices=_STATE_BACKENDS, default="de430",
                        help=("de430 evaluates JPL DE430 Sun/Moon states; reference replays "
                              "the NASA/GSFC Besselian elements of the bundled events"))
    parser.add_argument("--observer-lat", type=float)
    parser.add_argument("--observer-lon", type=float)
    parser.add_argument("--observer-elevation-m", type=float, default=0.0)
    parser.add_argument("--no-refraction", action="store_true")
    parser.add_argument("--horizon-profile", type=Path,
                        help="JSON or CSV azimuth/altitude horizon for the observer")


def _observer_from_args(args):
    if args.observer_lat is None and args.observer_lon is None:
        return None
    if args.observer_lat is None or args.observer_lon is None:
        raise SystemExit("--observer-lat and --observer-lon must be supplied together")
    from ssatk.coordinates.eclipse_observer_geometry import HorizonProfile, ObserverConfig
    return ObserverConfig(
        latitude_deg=args.observer_lat,
        longitude_east_deg=args.observer_lon,
        elevation_m=args.observer_elevation_m,
        apply_refraction=not args.no_refraction,
        horizon=HorizonProfile.from_file(args.horizon_profile) if args.horizon_profile else None,
        name="CLI WGS-84 observer",
    )


def _event_request_from_args(args):
    from ssatk.eclipse_api_impl import EventRequest
    return EventRequest(
        kind=args.kind,
        backend=args.backend,
        sample_count=args.samples,
        solar_scope=args.solar_scope,
        observer=_observer_from_args(args),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ssatk-eclipse",
        description="Discover, build, validate, and render finite-Sun solar and lunar eclipses.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="discover solar/lunar eclipses in a UTC interval from DE430 states")
    search.add_argument("--start", required=True, help="UTC date/time or Julian date")
    search.add_argument("--end", required=True, help="UTC date/time or Julian date")
    search.add_argument("--kind", choices=("all", "solar", "lunar"), default="all")
    search.add_argument("--output", type=Path, required=True)

    materialize = sub.add_parser("materialize", help="build a renderer-ready event from a search catalog")
    materialize.add_argument("--catalog", type=Path, required=True)
    selector = materialize.add_mutually_exclusive_group(required=True)
    selector.add_argument("--key")
    selector.add_argument("--index", type=int)
    materialize.add_argument("--samples", type=int, default=121)
    materialize.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build", help="build and serialize one bundled event")
    _event_arguments(build)
    build.add_argument("--output", type=Path, required=True)

    validate = sub.add_parser("validate", help="validate a serialized or newly built event")
    _event_arguments(validate)
    validate.add_argument("--input", type=Path, help="serialized event; overrides the event arguments")
    validate.add_argument("--output", type=Path, required=True)

    render = sub.add_parser("render", help="render a bundled or serialized event")
    render.add_argument("product", choices=_PRODUCTS)
    _event_arguments(render)
    render.add_argument("--input-event", type=Path, help="serialized event; overrides the event arguments")
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--quality", choices=_QUALITIES, default="high")
    render.add_argument("--animate", action="store_true")
    render.add_argument("--playback-seconds", type=float)
    render.add_argument(
        "--photometry",
        choices=("uniform-disc", "linear-visible", "quadratic-visible"),
        default="quadratic-visible",
        help="surface irradiance law; geometric contacts always use the uniform disc",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "search":
        from ssatk.compute.eclipse_event_search import discover_eclipses, write_discovery_catalog
        records = discover_eclipses(args.start, args.end, kind=args.kind)
        output = write_discovery_catalog(records, args.output)
        print(json.dumps({"catalog": output, "count": len(records),
                          "keys": [record.key for record in records]}, indent=2))
        return 0

    if args.command == "materialize":
        from ssatk.eclipse_api_impl import export_event_state
        from ssatk.compute.eclipse_event_search import build_discovered_event, read_discovery_catalog
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
        event = build_discovered_event(record, n_frames=args.samples)
        print(json.dumps({"event_state": export_event_state(event, args.output),
                          "event_key": event.definition.key}, indent=2))
        return 0

    if args.command == "build":
        from ssatk.eclipse_api_impl import build_event, export_event_state
        event = build_event(_event_request_from_args(args))
        print(json.dumps({"event_state": export_event_state(event, args.output)}, indent=2))
        return 0

    if args.command == "validate":
        from ssatk.eclipse_api_impl import build_event, load_event_state, validate_event
        observer = _observer_from_args(args)
        event = load_event_state(args.input) if args.input else build_event(_event_request_from_args(args))
        result = validate_event(event, observer=observer)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps({"validation": str(args.output), "passed": result.passed}, indent=2))
        return 0 if result.passed else 1

    if args.command == "render":
        from ssatk.eclipse_api_impl import RenderRequest, load_event_state, render_product
        event = load_event_state(args.input_event) if args.input_event else _event_request_from_args(args)
        request = RenderRequest(product=args.product, output=args.output, event=event,
                                quality=args.quality, animate=args.animate,
                                playback_seconds=args.playback_seconds,
                                photometry=args.photometry)
        print(json.dumps({"output": render_product(request)}, indent=2, default=str))
        return 0

    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
