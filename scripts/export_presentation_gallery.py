#!/usr/bin/env python3
"""Export referenced media from a PowerPoint deck as a portable gallery."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import posixpath
import re
import zipfile
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET


NS = {"r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
      "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SUPPORTED = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4"}


def _safe_zip_path(name: str) -> str | None:
    name = name.replace("\\", "/")
    if name.startswith("/"):
        return None
    clean = posixpath.normpath(name)
    if clean == ".." or clean.startswith("../"):
        return None
    return clean


def _resolve(source: str, target: str) -> str | None:
    if target.startswith("/"):
        return _safe_zip_path(target[1:])
    return _safe_zip_path(posixpath.join(posixpath.dirname(source), target))


def _rels(zf: zipfile.ZipFile, source: str) -> dict[str, tuple[str, str]]:
    relpath = posixpath.join(posixpath.dirname(source), "_rels", posixpath.basename(source) + ".rels")
    try:
        root = ET.fromstring(zf.read(relpath))
    except (KeyError, ET.ParseError):
        return {}
    result = {}
    for rel in root.findall(f"{{{REL_NS}}}Relationship"):
        rid = rel.get("Id")
        target = rel.get("Target", "")
        mode = rel.get("TargetMode", "Internal")
        if rid and mode.lower() != "external":
            resolved = _resolve(source, target)
            if resolved:
                result[rid] = (resolved, rel.get("Type", ""))
    return result


def _slide_order(zf: zipfile.ZipFile) -> list[str]:
    source = "ppt/presentation.xml"
    try:
        root = ET.fromstring(zf.read(source))
        rels = _rels(zf, source)
        ordered = []
        for node in root.findall(".//p:sldId", NS):
            rid = node.get(f"{{{NS['r']}}}id")
            if rid in rels and rels[rid][0].startswith("ppt/slides/"):
                ordered.append(rels[rid][0])
        if ordered:
            return ordered
    except (KeyError, ET.ParseError):
        pass
    return sorted(
        (n for n in zf.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
        key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1)),
    )


def _slide_title(root: ET.Element, number: int) -> str:
    # The first title placeholder is preferable; otherwise use the first text.
    for shape in root.findall(".//p:sp", NS):
        ph = shape.find(".//p:ph", NS)
        text = " ".join(t.text or "" for t in shape.findall(".//a:t", NS)).strip()
        if text and ph is not None and ph.get("type") in {"title", "ctrTitle"}:
            return text
    for text in root.findall(".//a:t", NS):
        if (text.text or "").strip():
            return (text.text or "").strip()
    return f"Slide {number}"


def _media_refs(root: ET.Element) -> list[str]:
    refs = []
    for node in root.iter():
        for key, value in node.attrib.items():
            if key == f"{{{NS['r']}}}embed" or key == f"{{{NS['r']}}}link":
                if value not in refs:
                    refs.append(value)
    return refs


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_example_href(value: object) -> str | None:
    if (not isinstance(value, str) or not value or value != value.strip()
            or re.search(r"[\x00-\x20\x7f\\]", value)):
        return None
    if value.startswith("/") or value.startswith("//") or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        return None
    if value.lower().startswith(("javascript:", "data:", "vbscript:")):
        return None
    return value


def _load_catalog(path: Path, media_names: set[str]) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("groups", []), list):
        raise ValueError("catalog must be an object with a groups list")
    for group in data.get("groups", []):
        if not isinstance(group, dict):
            raise ValueError("catalog groups must be objects")
        media = group.get("media", [])
        if not isinstance(media, list) or any(name not in media_names for name in media):
            raise ValueError(f"catalog group {group.get('id', '<unnamed>')!r} references missing media")
        examples = group.get("examples", [])
        for example in examples:
            if not isinstance(example, dict) or _safe_example_href(example.get("href")) is None:
                raise ValueError("catalog examples require safe local relative href values")
    return data


def export(deck: Path, output: Path, catalog_path: Path | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    media_dir = output / "media"
    media_dir.mkdir(exist_ok=True)
    source_sha = hashlib.sha256(deck.read_bytes()).hexdigest()
    media = []
    by_hash: dict[str, dict] = {}
    slides = []
    with zipfile.ZipFile(deck) as zf:
        names = set(zf.namelist())
        for slide_number, slide in enumerate(_slide_order(zf), 1):
            try:
                root = ET.fromstring(zf.read(slide))
            except (KeyError, ET.ParseError):
                continue
            slides.append({"number": slide_number, "title": _slide_title(root, slide_number)})
            relationships = _rels(zf, slide)
            for rid in _media_refs(root):
                resolved = relationships.get(rid, (None, ""))[0]
                if not resolved or resolved not in names:
                    continue
                suffix = Path(resolved).suffix.lower()
                if suffix not in SUPPORTED:
                    continue
                payload = zf.read(resolved)
                digest = _sha(payload)
                item = by_hash.get(digest)
                if item is None:
                    ext = suffix
                    preferred = Path(resolved).name
                    if any(existing["filename"] == preferred for existing in media):
                        preferred = f"{Path(preferred).stem}-{digest[:8]}{ext}"
                    item = {"filename": preferred, "sha256": digest,
                            "bytes": len(payload), "type": "video" if suffix == ".mp4" else "image",
                            "slides": []}
                    by_hash[digest] = item
                    (media_dir / item["filename"]).write_bytes(payload)
                    media.append(item)
                if slide_number not in item["slides"]:
                    item["slides"].append(slide_number)
    media_names = {item["filename"] for item in media}
    catalog = _load_catalog(catalog_path, media_names) if catalog_path else None
    manifest_media = [{k: v for k, v in item.items() if k != "bytes"} | {"size": item["bytes"]} for item in media]
    manifest = {"source": {"filename": deck.name, "sha256": source_sha},
                "slides": slides,
                "media": manifest_media}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    page_title = str(catalog.get("title", deck.stem)) if catalog else deck.stem
    (output / "index.html").write_text(_render_html(page_title, media, catalog), encoding="utf-8")
    return manifest


def _render_html(title: str, media: list[dict], catalog: dict | None) -> str:
    by_name = {item["filename"]: item for item in media}
    groups = catalog.get("groups", []) if catalog else []
    used = set()
    cards = []
    def media_markup(item: dict, label: str) -> str:
        used.add(item["filename"])
        name = html.escape(quote(item["filename"]), quote=True)
        if item["type"] == "video":
            content = f'<video controls preload="metadata" src="media/{name}"></video>'
        else:
            content = f'<a href="media/{name}"><img loading="lazy" src="media/{name}" alt="{html.escape(label, quote=True)}"></a>'
        slides = ", ".join(map(str, item["slides"]))
        return f'<figure>{content}<figcaption>{html.escape(label)} · slide(s): {slides}</figcaption></figure>'
    for group_number, group in enumerate(groups, 1):
        group_media = list(dict.fromkeys(group.get("media", [])))
        title_text = str(group.get("title", group.get("id", "Group")))
        body = "".join(media_markup(by_name[name], title_text) for name in group_media)
        links = "".join(f'<a href="{html.escape(_safe_example_href(e["href"]), quote=True)}">{html.escape(str(e.get("label", e["href"])))}</a> '
                         for e in group.get("examples", []))
        cards.append(f'<section id="group-{group_number}"><article><h2>{html.escape(title_text)}</h2><div class="grid">{body}</div><p>{html.escape(str(group.get("note", "")))}</p><p>{links}</p></article></section>')
    for item in media:
        if item["filename"] not in used:
            cards.append(f'<section><article><h2>Ungrouped media</h2><div class="grid">{media_markup(item, item["filename"])}</div></article></section>')
    return """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Presentation reference media</title><style>body{margin:0;background:#10131a;color:#e7eaf0;font:16px system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:2rem}h1{margin-bottom:.25rem}header{color:#aeb8c8}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}article{background:#1b2230;border:1px solid #303b4e;border-radius:10px;padding:1rem}img,video{display:block;max-width:100%;max-height:360px;margin:auto;border-radius:5px}a{color:#8dc8ff}.meta{color:#aeb8c8;font-size:.9em}</style><main><h1>Presentation reference media</h1><header>Archived source visuals; not regenerated or independently validated simulation results.</header>""" + f"<p>{html.escape(title)}</p>" + "".join(cards) + "</main></html>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    export(args.deck, args.output, args.catalog)


if __name__ == "__main__":
    main()
