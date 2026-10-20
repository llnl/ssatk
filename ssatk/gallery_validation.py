"""Validate local assets before publishing a portable static gallery.

This checks HTML attributes, CSS URLs and the gallery manifest. It does not
execute JavaScript or contact external sites; dynamic fetches and external
services need separate browser checks.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


_CSS_URL = re.compile(r"""url\(\s*(?:"((?:\\.|[^"\\])*)"|'((?:\\.|[^'\\])*)'|((?:\\(?:[0-9a-fA-F]{1,6}\s?|[^\r\n\f])|[^\s)'"\\])+))\s*\)""", re.I)
_CSS_TOKEN = re.compile(
    r"/\*.*?\*/|" + _CSS_URL.pattern + r"""|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'""", re.S | re.I)
_CSS_ESCAPE = re.compile(r"\\([0-9a-fA-F]{1,6}\s?|[^\r\n\f])")


def _css_urls(css):
    # Tokenize comments/ordinary strings too, so their text is never read as a URL.
    def unescape(match):
        value = match[1]
        if re.fullmatch(r"[0-9a-fA-F]{1,6}\s?", value):
            code = int(value.strip(), 16)
            return chr(code) if 0 < code <= 0x10ffff else "\ufffd"
        return value
    return [_CSS_ESCAPE.sub(unescape, next(group for group in match.groups() if group is not None))
            for match in _CSS_TOKEN.finditer(css) if any(group is not None for group in match.groups())]


class _Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []
        self.bases: list[str] = []
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "base" and attrs.get("href"):
            self.bases.append(attrs["href"])
        for key in ("href", "src", "poster"):
            if attrs.get(key) and tag != "base":
                self.urls.append(attrs[key])
        if tag == "object" and attrs.get("data"):
            self.urls.append(attrs["data"])
        self.urls.extend(_css_urls(attrs.get("style", "")))
        if tag == "style":
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False

    def handle_data(self, data):
        if self.in_style:
            self.urls.extend(_css_urls(data))


def validate_gallery(root: str | Path) -> dict:
    """Raise ValueError for missing assets or paths outside the published tree.

    Relative paths retain their browser meaning at a project URL such as
    ``https://software.llnl.gov/ssatk/``. Root-relative URLs are rejected rather
    than accidentally checked against the local site's root.
    """
    root = Path(root).expanduser().resolve()
    errors = []
    links = 0
    external = set()

    def check(source, url):
        nonlocal links
        parsed = urlsplit(url.strip())
        if parsed.scheme in ("http", "https") or parsed.netloc:
            external.add(url)
            return
        if parsed.scheme in ("data", "mailto", "tel", "blob"):
            return
        if not parsed.path and not parsed.scheme:
            return
        links += 1
        path = unquote(parsed.path)
        target = (source.parent / path).resolve()
        label = f"{source.relative_to(root)}: {url}"
        if parsed.scheme or path.startswith(("/", "\\")) or "\\" in path:
            errors.append(f"Non-portable URL: {label}")
        elif not target.is_relative_to(root):
            errors.append(f"Asset escapes site: {label}")
        elif not target.is_file() and not (target.is_dir() and (target / "index.html").is_file()):
            errors.append(f"Missing asset: {label}")

    if not (root / "index.html").is_file():
        errors.append("Missing site entry point: index.html")
    entries = sorted(root.rglob("*"))
    for path in entries:
        if path.is_symlink():
            errors.append(f"Symlinks are not portable gallery assets: {path.relative_to(root)}")
    files = [p for p in entries if p.is_file()]
    for path in files:
        if not path.resolve().is_relative_to(root):
            errors.append(f"File escapes site through a symlink: {path.relative_to(root)}")
            continue
        if path.suffix.lower() in (".html", ".htm"):
            parser = _Links()
            parser.feed(path.read_text(encoding="utf-8"))
            for base in parser.bases:
                if base not in ("", "./"):
                    errors.append(f"Non-portable base URL: {path.relative_to(root)}: {base}")
            for url in parser.urls:
                check(path, url)
        elif path.suffix.lower() == ".css":
            for url in _css_urls(path.read_text(encoding="utf-8")):
                check(path, url)
    manifest = root / "manifest.json"
    if manifest.is_file():
        for demo in json.loads(manifest.read_text(encoding="utf-8")):
            for url in demo["files"]:
                check(root / "index.html", url)
    if errors:
        raise ValueError("Gallery cannot be published:\n" + "\n".join(sorted(set(errors))))
    return {"files": len(files), "local_references_checked": links,
            "external_urls_not_checked": sorted(external)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site", type=Path)
    args = parser.parse_args(argv)
    try:
        report = validate_gallery(args.site)
    except ValueError as exc:
        parser.exit(1, str(exc) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
