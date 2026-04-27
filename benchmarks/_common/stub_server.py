"""Tiny HTTP stub mimicking cellmap-flow's chunk-URL contract.

Spins up a stdlib http.server in a background thread that responds to URLs of
the shape `/<dataset>/s<scale>/<z>.<y>.<x>` with random bytes after an optional
artificial delay. Used to smoke-test the B1 client without booting a real
cellmap-flow server.
"""

from __future__ import annotations

import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

_CHUNK_RE = re.compile(r"^/(?P<dataset>[^/]+)/s(?P<scale>\d+)/(?P<z>\d+)\.(?P<y>\d+)\.(?P<x>\d+)/?$")


def _make_handler(payload_bytes: int, fixed_delay_ms: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            if not _CHUNK_RE.match(self.path):
                self.send_error(404)
                return
            if fixed_delay_ms > 0:
                time.sleep(fixed_delay_ms / 1000.0)
            payload = os.urandom(payload_bytes)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:  # silence default noisy logs
            pass

    return Handler


class StubServer:
    """Context-manageable HTTP stub. Use:

        with StubServer(payload_bytes=8192, fixed_delay_ms=10) as srv:
            url = srv.url  # http://127.0.0.1:<port>
    """

    def __init__(self, payload_bytes: int = 8192, fixed_delay_ms: float = 10.0):
        self.payload_bytes = payload_bytes
        self.fixed_delay_ms = fixed_delay_ms
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def __enter__(self) -> "StubServer":
        handler = _make_handler(self.payload_bytes, self.fixed_delay_ms)
        self._server = HTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None

    @property
    def url(self) -> str:
        if self._server is None:
            raise RuntimeError("stub server not started")
        host, port = self._server.server_address
        return f"http://{host}:{port}"
