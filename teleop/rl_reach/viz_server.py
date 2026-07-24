"""
Live 3D reach visualizer — a small web server launched by `eval_reach.py`.

Serves an interactive three.js scene (arm skeleton + target vs. end-effector)
next to the robot's live camera feed. It is **display only**: it never touches
the serial port. The eval loop (the sole serial owner) pushes state to it via
`publish()`; the browser pulls that state from `/state` and the camera frames
from `/video`.

Runs uvicorn in a daemon thread so it doesn't block the eval loop. Reuses the
shared `CameraFeed` (teleop.vision.camera_feed) for the MJPEG passthrough.

Ports: robot 8000, vision 8080, mirror 8090, this 8091.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..vision.camera import macos_camera_names
from ..vision.camera_feed import CameraFeed

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"


class CameraReq(BaseModel):
    index: int   # negative = off


class ReachVizServer:
    """Display-only visualizer server. Never opens the serial port."""

    def __init__(self, config: dict, host: str = "127.0.0.1", port: int = 8091,
                 camera_index: int | None = None):
        self.host, self.port = host, port
        self._config = config                     # static: ee bounds, success thresh
        self._state_lock = threading.Lock()
        self._state: dict = {}
        self.camera_feed = CameraFeed()
        self._camera_index = camera_index
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._app = self._build_app()

    # ── data in (from the eval loop) ──────────────────────────────────────
    def publish(self, state: dict) -> None:
        with self._state_lock:
            self._state = state

    def switch_camera(self, index: int | None) -> None:
        self.camera_feed.switch(index)

    # ── FastAPI app ───────────────────────────────────────────────────────
    def _build_app(self) -> FastAPI:
        app = FastAPI(title="SO-101 reach visualizer")
        app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

        @app.get("/")
        def index():
            return FileResponse(STATIC / "index.html")

        @app.get("/config")
        def config():
            return self._config

        @app.get("/state")
        def state():
            with self._state_lock:
                return JSONResponse(self._state)

        @app.get("/video")
        def video():
            return StreamingResponse(
                self.camera_feed.mjpeg(),
                media_type="multipart/x-mixed-replace; boundary=frame",
            )

        @app.get("/api/cameras")
        def cameras():
            # Indices are the reliable handle; names are an unordered macOS hint.
            return {"indices": list(range(5)), "names": macos_camera_names(),
                    "current": self.camera_feed.current()}

        @app.post("/api/camera")
        def set_camera(req: CameraReq):
            self.camera_feed.switch(None if req.index < 0 else req.index)
            return {"ok": True, "index": req.index}

        return app

    # ── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        self.camera_feed.start()
        if self._camera_index is not None:
            self.camera_feed.switch(self._camera_index)
        cfg = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="warning")
        self._server = uvicorn.Server(cfg)
        self._server.install_signal_handlers = lambda: None   # runs off the main thread
        self._thread = threading.Thread(target=self._server.run, name="viz-server", daemon=True)
        self._thread.start()
        logger.info("reach visualizer at %s", self.url)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        self.camera_feed.stop()

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
