"""
Threaded webcam capture → MJPEG stream, independent of any serial/robot state.

Reused by the robot dashboard (`teleop.robot.server`) and the RL eval visualizer
(`teleop.rl_reach.viz_server`) so both share one capture/encode implementation.
Wraps `teleop.vision.camera.Camera`; switchable live; off by default.
"""

from __future__ import annotations

import logging
import threading
import time

import cv2

from .camera import Camera

logger = logging.getLogger(__name__)


class CameraFeed:
    def __init__(self, width=1280, height=720, fps=30, jpeg_quality=80):
        self.width, self.height, self.fps, self.jpeg_quality = width, height, fps, jpeg_quality
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._event = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._index: int | None = None      # None = off
        self._desired: int | None = None
        self._dirty = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def switch(self, index: int | None) -> None:
        with self._lock:
            self._desired = index
            self._dirty = True

    def current(self) -> int | None:
        with self._lock:
            return self._index

    def _run(self) -> None:
        camera: Camera | None = None
        current: int | None = None
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        while not self._stop.is_set():
            with self._lock:
                dirty, desired = self._dirty, self._desired
                self._dirty = False
            if dirty and desired != current:
                if camera is not None:
                    camera.release()
                    camera = None
                current = desired
                with self._lock:
                    self._index = current
                if desired is not None:
                    try:  # not a selfie view — no mirror flip
                        camera = Camera(index=desired, width=self.width, height=self.height,
                                        fps=self.fps, mirror=False)
                    except RuntimeError as exc:
                        logger.warning("camera %s failed to open: %s", desired, exc)
                        camera = None
            if camera is None:
                time.sleep(0.1)
                continue
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            ok, buf = cv2.imencode(".jpg", frame, encode)
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
                self._event.set()
        if camera is not None:
            camera.release()

    def mjpeg(self):
        boundary = b"--frame\r\n"
        while not self._stop.is_set():
            self._event.wait(timeout=1.0)
            self._event.clear()
            with self._lock:
                jpeg = self._jpeg
            if jpeg is None:
                continue
            yield boundary + b"Content-Type: image/jpeg\r\n"
            yield f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
            yield jpeg + b"\r\n"
