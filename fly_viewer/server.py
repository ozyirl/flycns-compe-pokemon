"""Dependency-light local server for the visualization-only Fly CNS browser."""

from __future__ import annotations

import json
import mimetypes
import webbrowser
from argparse import ArgumentParser, Namespace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from fly_viewer.catalog import load_catalog


STATIC_ROOT = Path(__file__).with_name("static")


def parse_args() -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("artifacts/fly_viewer_catalog.json.gz"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="open the local UI in a browser")
    return parser.parse_args()


def make_handler(catalog: dict) -> type[BaseHTTPRequestHandler]:
    catalog_bytes = json.dumps(catalog, separators=(",", ":")).encode()

    class ViewerHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            path = unquote(urlparse(self.path).path)
            if path == "/api/catalog":
                self._send(catalog_bytes, "application/json; charset=utf-8")
                return
            if path == "/healthz":
                self._send(b'{"status":"ok","mode":"visualization-only"}', "application/json")
                return
            relative = "index.html" if path in {"", "/"} else path.lstrip("/")
            target = (STATIC_ROOT / relative).resolve()
            if STATIC_ROOT.resolve() not in target.parents and target != STATIC_ROOT.resolve():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(target.read_bytes(), content_type)

        def _send(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin-allow-popups")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            print(f"viewer: {format % args}")

    return ViewerHandler


def main() -> None:
    args = parse_args()
    if not args.catalog.exists():
        raise SystemExit(
            f"Missing {args.catalog}. Build it first with: .venv/bin/python build_fly_viewer.py"
        )
    catalog = load_catalog(args.catalog)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(catalog))
    url = f"http://{args.host}:{args.port}"
    print(f"Fly CNS viewer: {url}")
    print("Visualization-only mode: no environment, PPO, model, or training code loaded.")
    if args.open:
        webbrowser.open_new_tab(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
