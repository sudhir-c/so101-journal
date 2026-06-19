#!/usr/bin/env python3
"""
Robot-clip compressor.

A tiny tkinter desktop app that compresses screen-recorded robot-arm clips with
ffmpeg so they fit under GitHub's 10 MB embed limit, with an optional speed-up
(2x / 5x) and a burned-in speed badge.

Dependencies: ffmpeg + ffprobe on PATH, and the Python standard library.
"""

import os
import re
import shutil
import struct
import subprocess
import tempfile
import threading
import queue
import zlib
import tkinter as tk
from tkinter import ttk, filedialog


# --- Constants ---------------------------------------------------------------

MAX_BYTES = 10 * 1024 * 1024        # GitHub embed limit: 10 MB
LONG_EDGE_CAP = 1280                # cap long edge at 720p-class (1280x720)
START_CRF = 28
MAX_CRF = 34
CRF_STEP = 4
PRESET = "slow"

# Bitmap glyphs (6x9) for the speed badge — rendered into a PNG with the
# standard library, so the badge works on any ffmpeg build (no freetype /
# drawtext required; we composite it with the universal `overlay` filter).
BADGE_GLYPHS = {
    "2": [".####.", "#....#", ".....#", "....#.", "...#..",
          "..#...", ".#....", "#.....", "######"],
    "5": ["######", "#.....", "#.....", "#####.", ".....#",
          ".....#", "#....#", "#....#", ".####."],
    "x": ["......", "#....#", ".#..#.", "..##..", "..##..",
          ".#..#.", "#....#", "......", "......"],
}
GLYPH_W, GLYPH_H = 6, 9

# Speed options: label -> (factor, suffix)
SPEEDS = {
    "Normal": (1, ""),
    "2×": (2, "_2x"),
    "5×": (5, "_5x"),
}


# --- ffmpeg helpers ----------------------------------------------------------

# Places ffmpeg commonly lives, for when the app is launched outside a shell
# (double-clicked / Finder / py2app) and PATH is the bare macOS default.
FALLBACK_BIN_DIRS = [
    "/opt/homebrew/bin",   # Apple-silicon Homebrew
    "/usr/local/bin",      # Intel Homebrew / manual installs
    "/opt/local/bin",      # MacPorts
    "/usr/bin",
]


def _locate(name):
    """Find a binary on PATH, falling back to common install directories."""
    found = shutil.which(name)
    if found:
        return found
    for d in FALLBACK_BIN_DIRS:
        candidate = os.path.join(d, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def find_ffmpeg():
    """Return (ffmpeg, ffprobe) paths or (None, None) if missing."""
    return _locate("ffmpeg"), _locate("ffprobe")


def make_badge_png(text, path, scale=16, pad=26, gap=14, corner=10):
    """
    Render `text` (e.g. "2×") as a PNG: white bitmap glyphs on a
    semi-transparent dark, slightly rounded box. Pure standard library, so it
    needs no font files and no drawtext/freetype support in ffmpeg.
    """
    glyphs = [BADGE_GLYPHS["x" if c in "x×" else c] for c in text]
    text_w = len(glyphs) * GLYPH_W * scale + gap * (len(glyphs) - 1)
    text_h = GLYPH_H * scale
    W, H = text_w + 2 * pad, text_h + 2 * pad

    px = bytearray([0, 0, 0, 150] * (W * H))   # semi-transparent dark box

    def block(x, y):                            # paint one white scaled pixel
        for dy in range(scale):
            base = ((y + dy) * W + x) * 4
            for dx in range(scale):
                px[base + dx * 4: base + dx * 4 + 4] = b"\xff\xff\xff\xff"

    x0 = pad
    for g in glyphs:
        for ry, row in enumerate(g):
            for rx, ch in enumerate(row):
                if ch == "#":
                    block(x0 + rx * scale, pad + ry * scale)
        x0 += GLYPH_W * scale + gap

    # Soften the four corners so the box reads as rounded.
    for cx, cy in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        for dy in range(corner):
            for dx in range(corner):
                xx, yy = (cx and cx - dx) or dx, (cy and cy - dy) or dy
                if dx * dx + dy * dy > corner * corner:
                    px[(yy * W + xx) * 4 + 3] = 0

    raw = bytearray()
    for y in range(H):
        raw.append(0)                           # 'none' filter for this row
        raw += px[y * W * 4:(y + 1) * W * 4]
    comp = zlib.compress(bytes(raw), 9)

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data +
                struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", comp))
        f.write(chunk(b"IEND", b""))
    return path


def probe_duration(ffprobe, path):
    """Return clip duration in seconds (float), or 0.0 if it can't be read."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=False,
        )
        return float(out.stdout.strip())
    except (ValueError, Exception):
        return 0.0


def probe_dimensions(ffprobe, path):
    """Return 'WxH' for the first video stream, or '?' on failure."""
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=s=x:p=0", path],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


# Cap the long edge while preserving aspect ratio; never upscale; even dims.
SCALE = (f"scale=w='if(gt(iw,ih),min(iw,{LONG_EDGE_CAP}),-2)':"
         f"h='if(gt(iw,ih),-2,min(ih,{LONG_EDGE_CAP}))'")


def build_command(ffmpeg, src, dst, crf, factor, badge_png, keep_audio):
    """
    Assemble the ffmpeg command.
      - Normal: just scale, keep audio.
      - Sped up: setpts + scale, drop audio, and (if badge_png is given)
        overlay the speed badge in the top-right corner via `overlay`
        (universal — no drawtext/freetype needed).
    """
    venc = ["-c:v", "libx264", "-crf", str(crf), "-preset", PRESET,
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    if factor == 1:
        cmd = [ffmpeg, "-y", "-i", src, "-vf", SCALE,
               *venc, "-c:a", "aac", "-b:a", "128k"]
    elif badge_png:
        # Badge height tracks the video (1/6 of frame height); margin 28px.
        fc = (f"[0:v]setpts=PTS/{factor},{SCALE}[v];"
              f"[1:v][v]scale2ref=w=-1:h=main_h/6[bdg][vid];"
              f"[vid][bdg]overlay=W-w-28:28[out]")
        cmd = [ffmpeg, "-y", "-i", src, "-i", badge_png,
               "-filter_complex", fc, "-map", "[out]", "-an", *venc]
    else:                                       # badge unavailable: skip it
        cmd = [ffmpeg, "-y", "-i", src,
               "-vf", f"setpts=PTS/{factor},{SCALE}", "-an", *venc]

    cmd += ["-progress", "pipe:1", "-nostats", dst]
    return cmd


_TIME_RE = re.compile(r"out_time=(\d+):(\d+):(\d+(?:\.\d+)?)")


def run_encode(cmd, total_seconds, progress_cb):
    """
    Run one ffmpeg encode, streaming progress to progress_cb(fraction).
    Returns (returncode, stderr_text).
    """
    stderr_buf = tempfile.TemporaryFile(mode="w+")
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=stderr_buf, text=True,
        )
        for line in proc.stdout:
            m = _TIME_RE.search(line)
            if m and total_seconds > 0:
                h, mnt, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
                done = h * 3600 + mnt * 60 + s
                frac = max(0.0, min(1.0, done / total_seconds))
                progress_cb(frac)
        proc.wait()
        stderr_buf.seek(0)
        return proc.returncode, stderr_buf.read()
    finally:
        stderr_buf.close()


# --- Compression workflow (runs in a background thread) ----------------------

def compress_worker(ctx, src, factor, suffix, out_q):
    """
    ctx: dict with ffmpeg/ffprobe.
    Posts ('status'|'progress'|'done'|'error', payload) to out_q.
    """
    ffmpeg = ctx["ffmpeg"]
    ffprobe = ctx["ffprobe"]

    def post(kind, payload):
        out_q.put((kind, payload))

    badge_png = None
    try:
        if not os.path.isfile(src):
            post("error", "Selected file no longer exists.")
            return

        in_size = os.path.getsize(src)
        duration = probe_duration(ffprobe, src)
        out_duration = duration / factor if factor else duration

        base, _ = os.path.splitext(src)
        dst = f"{base}{suffix}_compressed.mp4"

        keep_audio = (factor == 1)

        # Build the speed badge for sped-up clips. If anything goes wrong we
        # simply skip it rather than failing the whole job.
        if factor != 1:
            try:
                fd, badge_png = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                make_badge_png("2×" if factor == 2 else "5×", badge_png)
            except Exception:
                badge_png = None

        crf = START_CRF
        final_size = None
        exceeded = False

        while True:
            post("status", f"Encoding at CRF {crf} (preset {PRESET})…")
            post("progress", 0.0)

            cmd = build_command(ffmpeg, src, dst, crf, factor,
                                badge_png, keep_audio)
            rc, err = run_encode(cmd, out_duration,
                                  lambda f: post("progress", f))

            if rc != 0:
                tail = "\n".join(err.strip().splitlines()[-4:]) or "unknown error"
                post("error", f"ffmpeg failed (CRF {crf}):\n{tail}")
                return

            post("progress", 1.0)
            final_size = os.path.getsize(dst)

            if final_size <= MAX_BYTES:
                break
            if crf >= MAX_CRF:
                exceeded = True
                break
            crf = min(crf + CRF_STEP, MAX_CRF)

        dims = probe_dimensions(ffprobe, dst)
        post("done", {
            "dst": dst,
            "in_size": in_size,
            "out_size": final_size,
            "dims": dims,
            "crf": crf,
            "exceeded": exceeded,
        })
    except Exception as e:  # never crash the UI thread
        post("error", f"Unexpected error: {e}")
    finally:
        if badge_png and os.path.exists(badge_png):
            try:
                os.remove(badge_png)
            except OSError:
                pass


# --- UI ----------------------------------------------------------------------

def human_size(n):
    mb = n / (1024 * 1024)
    if mb >= 1:
        return f"{mb:.2f} MB"
    return f"{n / 1024:.0f} KB"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Clip Compressor")
        self.root.resizable(False, False)

        self.ffmpeg, self.ffprobe = find_ffmpeg()

        self.src_path = None
        self.speed_var = tk.StringVar(value="Normal")
        self.queue = queue.Queue()

        self._build_widgets()
        self._check_ffmpeg()
        self.root.after(100, self._poll_queue)

    def _build_widgets(self):
        pad = {"padx": 12, "pady": 6}
        frm = ttk.Frame(self.root, padding=14)
        frm.grid(sticky="nsew")

        self.choose_btn = ttk.Button(frm, text="Choose video…",
                                     command=self._choose)
        self.choose_btn.grid(row=0, column=0, sticky="w", **pad)

        self.file_lbl = ttk.Label(frm, text="No file selected",
                                  foreground="#666", width=42, anchor="w")
        self.file_lbl.grid(row=0, column=1, sticky="w", **pad)

        speed_frm = ttk.LabelFrame(frm, text="Speed", padding=8)
        speed_frm.grid(row=1, column=0, columnspan=2, sticky="ew", **pad)
        for i, label in enumerate(SPEEDS):
            ttk.Radiobutton(speed_frm, text=label, value=label,
                            variable=self.speed_var).grid(row=0, column=i,
                                                          padx=10)

        self.compress_btn = ttk.Button(frm, text="Compress", state="disabled",
                                       command=self._start)
        self.compress_btn.grid(row=2, column=0, columnspan=2, **pad)

        self.progress = ttk.Progressbar(frm, length=420, mode="determinate",
                                        maximum=100)
        self.progress.grid(row=3, column=0, columnspan=2, **pad)

        self.status = tk.Text(frm, width=52, height=6, wrap="word",
                              relief="flat", background="#f4f4f4")
        self.status.grid(row=4, column=0, columnspan=2, **pad)
        self.status.configure(state="disabled")

    def _set_status(self, text, color="#222"):
        self.status.configure(state="normal")
        self.status.delete("1.0", "end")
        self.status.insert("1.0", text)
        self.status.tag_configure("c", foreground=color)
        self.status.tag_add("c", "1.0", "end")
        self.status.configure(state="disabled")

    def _check_ffmpeg(self):
        if not self.ffmpeg or not self.ffprobe:
            self._set_status(
                "ffmpeg / ffprobe not found on your PATH.\n"
                "Install ffmpeg (e.g. `brew install ffmpeg`) and reopen.",
                color="#b00020")
            self.choose_btn.configure(state="disabled")
        else:
            self._set_status("Ready. Choose a video to begin.")

    def _choose(self):
        path = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[("Video files", "*.mp4 *.MP4 *.mov *.MOV"),
                       ("All files", "*.*")],
        )
        if not path:
            return
        if not path.lower().endswith((".mp4", ".mov")):
            self._set_status("Please pick an .mp4 or .mov file.", color="#b00020")
            return
        if not os.path.isfile(path):
            self._set_status("That file could not be opened.", color="#b00020")
            return
        self.src_path = path
        self.file_lbl.configure(text=os.path.basename(path), foreground="#222")
        self.compress_btn.configure(state="normal")
        self._set_status(f"Selected {os.path.basename(path)} "
                         f"({human_size(os.path.getsize(path))}).")

    def _start(self):
        if not self.src_path:
            return
        label = self.speed_var.get()
        factor, suffix = SPEEDS[label]

        self.choose_btn.configure(state="disabled")
        self.compress_btn.configure(state="disabled")
        self.progress["value"] = 0
        self._set_status("Starting…")

        ctx = {"ffmpeg": self.ffmpeg, "ffprobe": self.ffprobe}
        t = threading.Thread(target=compress_worker,
                             args=(ctx, self.src_path, factor, suffix,
                                   self.queue),
                             daemon=True)
        t.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.queue.get_nowait()
                if kind == "progress":
                    self.progress["value"] = payload * 100
                elif kind == "status":
                    self._set_status(payload)
                elif kind == "error":
                    self.progress["value"] = 0
                    self._set_status(payload, color="#b00020")
                    self._finish()
                elif kind == "done":
                    self._show_done(payload)
                    self._finish()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _show_done(self, r):
        self.progress["value"] = 100
        lines = [
            f"Saved: {os.path.basename(r['dst'])}",
            f"Folder: {os.path.dirname(r['dst'])}",
            f"Size: {human_size(r['in_size'])} → "
            f"{human_size(r['out_size'])}   ({r['dims']}, CRF {r['crf']})",
        ]
        if r["exceeded"]:
            lines.append(
                f"⚠ Still over 10 MB at CRF {MAX_CRF} — kept this "
                "file anyway. Try a higher speed-up to shrink it further.")
            color = "#8a6d00"
        else:
            lines.append("✓ Under 10 MB — ready to embed on GitHub.")
            color = "#1b7a32"
        self._set_status("\n".join(lines), color=color)

    def _finish(self):
        self.choose_btn.configure(state="normal")
        self.compress_btn.configure(
            state="normal" if self.src_path else "disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
