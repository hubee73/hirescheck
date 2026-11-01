#!/usr/bin/env python3
"""Inspect a PCM or DSD file's bandwidth, look for upsampling or PCM origin, and
estimate a 48 kHz round trip for high-rate PCM.

This measures signal properties, not audibility or recording provenance.
Python 3.10+; numpy, scipy; FFmpeg (ffmpeg + ffprobe on PATH); matplotlib for optional plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path



def detect_lang() -> str:
    """'zh' when the system locale is Chinese, otherwise 'en'."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(var, "")
        if value:
            return "zh" if value.lower().startswith("zh") else "en"
    try:
        import locale
        value = locale.getlocale()[0] or ""
    except (ValueError, ImportError):
        value = ""
    if not value and sys.platform == "darwin":
        try:
            value = subprocess.run(["defaults", "read", "-g", "AppleLanguages"], capture_output=True,
                                   text=True, timeout=2).stdout.replace("(", "").strip().strip('"')
        except (OSError, subprocess.SubprocessError):
            value = ""
    return "zh" if value.lower().startswith(("zh", "chinese")) else "en"


LANG = detect_lang()


def set_lang(lang: str) -> None:
    global LANG
    LANG = lang


def tr(zh: str, en: str) -> str:
    """The text in the output language; both versions sit side by side at each call site."""
    return zh if LANG == "zh" else en


def sep() -> str:
    return tr("；", "; ")


try:
    import numpy as np
    from scipy import ndimage, signal
except ImportError as exc:
    raise SystemExit(
        tr(f"缺少依赖：{exc.name}。请运行：\n", f"Missing dependency: {exc.name}. Install with:\n")
        + 'python -m pip install numpy scipy matplotlib'
    ) from exc


VERSION = "1.5.0"
REFERENCE_RATE = 48000
FLOOR = 1e-300
STANDARD_RATES = (44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000)
# Empirical band-limit rules; all are written to report.json.
EDGE_BIN_HZ = 100
EDGE_SEARCH_FROM_HZ = 10000
EDGE_MIN_DROP_DB = 25
# A brick wall (core 20-80% of the drop within 1.2 kHz, onto a flat floor) counts from
# a smaller drop and margin: weak treble over a low floor, common in CD-sourced SACDs.
EDGE_BRICKWALL_DROP_DB = 15
EDGE_BRICKWALL_CORE_HZ = 1200
EDGE_BRICKWALL_FLOOR_STD_DB = 1.5
EDGE_MAX_WIDTH_HZ = 3000
EDGE_MIN_SLOPE_DB_PER_OCTAVE = 120
EDGE_MIN_LEVEL_DB = -175
EDGE_EMPTY_MARGIN_DB = 20
EDGE_IMAGED_MARGIN_DB = 6
IMAGE_MIN_CORRELATION = 0.7
IMAGE_STANDALONE_CORRELATION = 0.9
IMAGE_MIN_SHAPE_DB = 1.5
LOSSY_EDGE_BELOW_HZ = 19000
# A source rate R is plausible when the edge midpoint lies within this share of R/2:
# filters pass to at least ~80% of Nyquist and cannot keep content far above it.
SOURCE_NYQUIST_RATIO = (0.80, 1.03)
# Steep resamplers put the midpoint a little below Nyquist, FFmpeg's swr right at
# it; candidates are ranked by how close the midpoint is to this share of R/2.
SOURCE_TYPICAL_RATIO = 0.975
# A cliff this deep at a standard Nyquist is a band limit even with weaker energy
# returning above it (strong images or noise added after the band was limited).
EDGE_STANDARD_CLIFF_DB = 30
# DSD: FFmpeg decodes to PCM at 1/8 of the bit rate, and its decimation filter is
# faithful (<1 dB) only up to about half of that PCM Nyquist; analysis stops there.
DSD_ANALYSIS_SHARE = 0.5
# Lossy-codec checks (calibrated on MP3/AAC/Opus/Vorbis round trips of CD rips):
# codecs low-pass at 16-20.4 kHz with a cliff; CD anti-alias filters sit at
# 20.75 kHz and above. Psychoacoustic coders also leave short-time holes above 16 kHz.
LOSSY_MAX_CUTOFF_HZ = 20500
LOSSY_SEARCH_FROM_HZ = 14000
LOSSY_STEP_DB = 25
LOSSY_HOLE_DB = 30
LOSSY_HOLES_PERCENT = 1.0
LOSSY_HOLES_NOTE_PERCENT = 0.5
# A frame with at least this share of its 16 kHz+ band empty is a musical pause, not a codec hole.
LOSSY_GAP_SHARE = 0.5
LOSSY_FRAME_SECONDS = 0.093
LOSSY_MAX_FRAMES = 30000
DSD_OBSERVABLE_RISE_DB = 20
# Processing-chain fingerprints (--check-sr / --learn-chain). A chain is a fixed
# production path (e.g. one shop's converter set-up) that leaves the same narrow
# tones and the same fine ultrasonic spectral shape on every file it touches.
CHAIN_FORMAT = 1
CHAIN_MIN_RATE = 88200            # below this there is no ultrasonic band to fingerprint
CHAIN_BAND_LOW_HZ = 30000         # shape band: 30 kHz .. 0.495 fs
CHAIN_POOL_BINS = 4               # shape resolution: 4 FFT bins (~12 Hz at 192 kHz)
CHAIN_SMOOTH_BINS = 101           # trend removed from the shape (~1.2 kHz at 192 kHz)
CHAIN_TONE_HALF_HZ = 15           # tone peak searched within +-15 Hz
CHAIN_TONE_SIDE_HZ = (300, 1500)  # local floor: median 0.3-1.5 kHz either side
CHAIN_TONE_NOTCH_HZ = 200         # tone bins are left out of the shape
CHAIN_SILENCE_DBFS = -90          # frames quieter than this are not averaged
CHAIN_LEARN_TONE_DB = 9           # a learned tone must stand this far out in every file
CHAIN_TONE_MARGIN_DB = 3          # match threshold = weakest learned prominence - 3 dB,
CHAIN_TONE_DB_RANGE = (8, 12)     # clamped to 8-12 dB
CHAIN_MAX_TONES = 8
CHAIN_SHAPE_STRONG = 0.15         # shape that upgrades a tone match to "very high"
CHAIN_SHAPE_ONLY = 0.30           # shape strong enough to match without the tones
CHAIN_MIN_FILES = 5               # --learn-chain: at least 5 files ...
CHAIN_MIN_FOLDERS = 3             # ... from at least 3 different folders (albums)
CHAIN_BUILTIN_DIR = Path(__file__).resolve().parent / "chains"  # package data
CHAIN_USER_DIR = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "hirescheck" / "chains"
DSD_HUMP_DB = 6


def db(power: float) -> float:
    """Power dB; use a finite display floor for exact digital silence."""
    return float(10 * np.log10(max(float(power), 1e-30)))


def band_power(freq: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """One-sided PSD integral. psd can contain channels in its leading axes."""
    mask = (freq >= lo) & (freq < hi)
    if hi >= freq[-1]:
        mask |= freq == freq[-1]
    return float(np.mean(np.sum(psd[..., mask], axis=-1)) * (freq[1] - freq[0]))


def relative_db(power: float, reference: float) -> float | None:
    return db(power / reference) if reference > FLOOR else None


def ffmpeg_input(path: Path) -> str:
    # "file:" keeps names such as "-x.flac" or "concat:a" from being read as options/protocols.
    return "file:" + str(path.resolve())


def missing_ffmpeg(tool: str) -> RuntimeError:
    return RuntimeError(tr(f"找不到 {tool}。请安装 FFmpeg，并确保 ffmpeg 与 ffprobe 在 PATH 中。",
                           f"{tool} not found. Install FFmpeg and make sure ffmpeg and ffprobe are on PATH."))


def probe(path: Path) -> dict:
    """Stream facts from ffprobe; the file extension is never trusted."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
             "stream=codec_name,sample_rate,channels,sample_fmt,bits_per_raw_sample,"
             "bits_per_sample,duration:format=format_name,duration",
             "-of", "json", ffmpeg_input(path)],
            capture_output=True, check=False,
        )
    except FileNotFoundError:
        raise missing_ffmpeg("ffprobe") from None
    if result.returncode:
        raise ValueError(tr("FFmpeg 无法读取该文件：", "FFmpeg cannot read this file: ")
                         + result.stderr.decode(errors="replace").strip())
    data = json.loads(result.stdout or b"{}")
    if not data.get("streams"):
        raise ValueError(tr("文件中没有音频流。", "The file has no audio stream."))
    stream, container = data["streams"][0], data.get("format", {})
    codec = stream.get("codec_name", "")
    rate = int(stream.get("sample_rate") or 0)
    if rate <= 0 or int(stream.get("channels") or 0) <= 0:
        # FFmpeg guesses a container from the extension even for non-audio files.
        raise ValueError(tr("无法读出采样率或声道数，文件可能损坏，或并不是音频。",
                            "Cannot read the sample rate or channel count; the file may be damaged or not audio."))
    dsd = codec.startswith(("dsd_", "dst"))
    bits = [int(v) for v in (stream.get("bits_per_raw_sample"), stream.get("bits_per_sample"))
            if str(v).isdigit() and int(v) > 0]
    durations = [float(v) for v in (stream.get("duration"), container.get("duration"))
                 if v not in (None, "N/A")]
    return {
        "container": container.get("format_name", ""), "codec": codec,
        "sample_rate_hz": rate, "channels": int(stream["channels"]),
        "dsd_rate_hz": rate * 8 if dsd else None,
        "sample_format": stream.get("sample_fmt", ""), "bits_per_sample": bits[0] if bits else None,
        "duration_seconds": durations[0] if durations else None,
    }


class PcmStream:
    """float64 PCM from FFmpeg at the file's own rate and channels: no resampling, mixing or dither."""

    def __init__(self, path: Path, channels: int, start: float = 0.0, seconds: float | None = None):
        cmd = ["ffmpeg", "-nostdin", "-v", "error"]
        if start > 0:
            cmd += ["-ss", f"{start:.6f}"]
        if seconds is not None:
            cmd += ["-t", f"{seconds:.6f}"]
        cmd += ["-i", ffmpeg_input(path), "-map", "0:a:0", "-f", "f64le", "-c:a", "pcm_f64le", "pipe:1"]
        self.channels = channels
        self.finished = False
        self.warnings = ""
        # A file rather than a pipe, so a flood of decoder warnings cannot stall FFmpeg.
        self.log = tempfile.TemporaryFile()
        try:
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=self.log)
        except FileNotFoundError:
            self.log.close()
            raise missing_ffmpeg("ffmpeg") from None

    def read(self, frames: int) -> np.ndarray:
        size = 8 * self.channels
        raw = self.proc.stdout.read(frames * size)
        if len(raw) < frames * size:
            self.finished = True
        raw = raw[:len(raw) - len(raw) % size]
        return np.frombuffer(raw, "<f8").reshape(-1, self.channels)

    def __enter__(self) -> PcmStream:
        return self

    def __exit__(self, exc_type, *_) -> None:
        self.proc.stdout.close()
        if exc_type or not self.finished:
            self.proc.kill()  # Stopped early on purpose; the exit status means nothing.
        code = self.proc.wait()
        self.log.seek(0)
        self.warnings = self.log.read().decode(errors="replace").strip()
        self.log.close()
        if exc_type is None and self.finished and code:
            raise RuntimeError(tr("FFmpeg 解码失败：", "FFmpeg decoding failed: ")
                               + (self.warnings or tr(f"退出码 {code}", f"exit code {code}")))


class LossyScan:
    """Short-window statistics above 12 kHz for lossy-codec checks.

    The long analysis window averages away codec artefacts; ~93 ms frames keep
    the holes a psychoacoustic coder leaves when it drops a band for a while.
    Per frame it stores the 2-8 kHz level and the 12-24 kHz spectrum pooled to
    ~50 Hz, at most LOSSY_MAX_FRAMES frames (evenly strided for long files).
    """

    def __init__(self, fs: int, bits: int | None, expected_seconds: float | None):
        self.n = 2 ** round(math.log2(fs * LOSSY_FRAME_SECONDS))
        self.hop = self.n // 2
        self.window = signal.windows.hann(self.n, sym=False)
        self.scale = 2 / (fs * np.sum(self.window**2))
        freq = np.fft.rfftfreq(self.n, 1 / fs)
        self.mid = (freq >= 2000) & (freq < 8000)
        per = max(1, round(50 / (freq[1] - freq[0])))
        lo = int(np.searchsorted(freq, 12000))
        hi = int(np.searchsorted(freq, min(fs / 2, 24000)))
        self.groups = max(0, (hi - lo) // per)
        self.lo, self.per = lo, per
        self.freq = freq[lo:lo + self.groups * per].reshape(self.groups, per).mean(axis=1) if self.groups else np.array([])
        expected = (expected_seconds or 0) * fs / self.hop
        self.stride = max(1, math.ceil(expected / LOSSY_MAX_FRAMES))
        # Rounding noise of the stored word length; content below it cannot be told from silence.
        self.floor_db = (10 * math.log10((2.0 ** (1 - bits)) ** 2 / 12 / (fs / 2))
                         if bits and bits <= 24 else None)
        self.carry = None
        self.count = 0
        self.mids: list[np.ndarray] = []
        self.bands: list[np.ndarray] = []

    def feed(self, block: np.ndarray) -> None:
        if not self.groups:
            return
        data = block if self.carry is None else np.concatenate((self.carry, block), axis=0)
        if len(data) < self.n:
            self.carry = data
            return
        frames = np.lib.stride_tricks.sliding_window_view(data, self.n, axis=0)[::self.hop]
        self.carry = data[len(frames) * self.hop:].copy()
        keep = (self.count + np.arange(len(frames))) % self.stride == 0
        self.count += len(frames)
        if not keep.any():
            return
        frames = frames[keep]
        power = (np.abs(np.fft.rfft(frames * self.window, axis=-1)) ** 2).mean(axis=1) * self.scale
        self.mids.append(power[:, self.mid].mean(axis=1).astype(np.float32))
        top = power[:, self.lo:self.lo + self.groups * self.per]
        self.bands.append(top.reshape(len(top), self.groups, self.per).mean(axis=2).astype(np.float32))

    def result(self, freq: np.ndarray, psd: np.ndarray, fs: int) -> dict | None:
        if not self.mids:
            return None
        mid = 10 * np.log10(np.concatenate(self.mids).astype(np.float64) + 1e-30)
        band = np.concatenate(self.bands).astype(np.float64)
        loud = mid > np.percentile(mid, 40)
        if loud.sum() < 20:
            return None
        level = 10 * np.log10(band[loud] + 1e-30)
        # Per-frame cutoff: highest band within 60 dB of the frame's mid level
        # and clearly above the word-length noise floor.
        need = mid[loud][:, None] - 60
        if self.floor_db is not None:
            need = np.maximum(need, self.floor_db + 31.5)
        above = level > need
        last = np.where(above.any(axis=1), self.groups - 1 - np.argmax(above[:, ::-1], axis=1), -1)
        cutoffs = np.where(last >= 0, self.freq[np.maximum(last, 0)], 12000.0)
        p95 = float(np.percentile(cutoffs, 95))
        # Steepest cliff of the average spectrum between 14 kHz and the codec range.
        top = min(fs / 2 - 900, 21800)
        avg = 10 * np.log10(np.maximum(psd, 1e-30))
        step = step_at = pre = None
        for c in np.arange(LOSSY_SEARCH_FROM_HZ, top, 50.0):
            b = float(np.median(avg[(freq >= c - 800) & (freq < c - 100)]))
            a = float(np.median(avg[(freq >= c + 100) & (freq < c + 800)]))
            if step is None or b - a > step:
                step, step_at, pre = b - a, float(c), b
        floor_ok = self.floor_db is None or (pre is not None and pre >= self.floor_db + 20)
        cliff = (step is not None and step >= LOSSY_STEP_DB and step_at <= LOSSY_MAX_CUTOFF_HZ and floor_ok)
        # Holes: time-frequency cells far below that band's own median in loud frames.
        # Codecs drop part of the band; a pause or gated hi-hat empties all of it,
        # so frames that are mostly empty are left out.
        ceiling = min(step_at if cliff else p95, 21000) - 300
        cells = (self.freq >= 16000) & (self.freq < ceiling)
        holes = gaps = None
        if cells.sum() >= 20:
            sub = band[loud][:, cells]
            hole = sub < np.median(sub, axis=0) * 10 ** (-LOSSY_HOLE_DB / 10)
            share = hole.mean(axis=1)
            partial = share < LOSSY_GAP_SHARE
            gaps = float(np.mean(~partial) * 100)
            holes = float(hole[partial].mean() * 100) if partial.any() else 0.0
        return {
            "frames_analyzed": int(loud.sum()), "frame_seconds": self.n / fs,
            "cliff_detected": bool(cliff), "cliff_hz": step_at, "cliff_drop_db": step,
            "frame_cutoff_p95_hz": p95,
            "loud_frames_above_20khz_percent": float(np.mean(cutoffs >= 20000) * 100),
            "holes_percent": holes, "full_band_gap_frames_percent": gaps,
            "holes_detected": bool(holes is not None and holes >= LOSSY_HOLES_PERCENT),
        }


class ChainScan:
    """Long-window average power spectrum of the mono mix, for chain fingerprints.

    ~fs/3-sample frames (65536 at 192 kHz, ~2.9 Hz bins) resolve the narrow
    converter tones; frames below CHAIN_SILENCE_DBFS are skipped.
    """

    def __init__(self, fs: int):
        self.fs = fs
        self.n = 2 ** round(math.log2(fs / 3))
        self.hop = self.n // 2
        self.window = signal.windows.hann(self.n, sym=False)
        self.scale = 2 / (fs * np.sum(self.window**2))
        self.sum = np.zeros(self.n // 2 + 1)
        self.count = 0
        self.carry = np.empty(0)

    def feed(self, block: np.ndarray) -> None:
        data = np.concatenate((self.carry, block.mean(axis=1)))
        if len(data) < self.n:
            self.carry = data
            return
        frames = np.lib.stride_tricks.sliding_window_view(data, self.n)[::self.hop]
        self.carry = data[len(frames) * self.hop:].copy()
        keep = np.mean(frames**2, axis=1) > 10 ** (CHAIN_SILENCE_DBFS / 10)
        if keep.any():
            x = frames[keep]
            x = x - x.mean(axis=1, keepdims=True)
            self.sum += (np.abs(np.fft.rfft(x * self.window, axis=1)) ** 2).sum(axis=0) * self.scale
            self.count += int(keep.sum())

    def result(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.count < 4:
            return None
        return np.fft.rfftfreq(self.n, 1 / self.fs), self.sum / self.count


def chain_spectrum(path: Path, max_seconds: float | None) -> tuple[dict, tuple | None]:
    """Decode one file only for its chain spectrum (used by --learn-chain)."""
    meta = probe(path)
    if meta["dsd_rate_hz"]:
        return meta, None
    scan = ChainScan(meta["sample_rate_hz"])
    with PcmStream(path, meta["channels"], seconds=max_seconds) as stream:
        while True:
            block = stream.read(1 << 18)
            if not len(block):
                break
            scan.feed(block)
    return meta, scan.result()


def tone_prominence(freq: np.ndarray, level: np.ndarray, hz: float) -> float | None:
    """Peak within +-15 Hz of `hz` above the median level 0.3-1.5 kHz either side (dB)."""
    near, far = CHAIN_TONE_SIDE_HZ
    peak = level[np.abs(freq - hz) <= CHAIN_TONE_HALF_HZ]
    gap = np.abs(freq - hz)
    side = level[(gap >= near) & (gap <= far)]
    if not len(peak) or not len(side):
        return None
    return float(peak.max() - np.median(side))


def chain_shape(freq: np.ndarray, psd: np.ndarray, fs: int, notch: list[float]) -> np.ndarray:
    """Detrended, standardised fine shape of the ultrasonic spectrum; tone bins are zeroed."""
    index = np.nonzero((freq >= CHAIN_BAND_LOW_HZ) & (freq < 0.495 * fs))[0]
    index = index[:len(index) // CHAIN_POOL_BINS * CHAIN_POOL_BINS]
    if len(index) < CHAIN_POOL_BINS * CHAIN_SMOOTH_BINS * 2:
        return np.zeros(0)
    pooled_f = freq[index].reshape(-1, CHAIN_POOL_BINS).mean(axis=1)
    pooled = 10 * np.log10(np.maximum(psd[index].reshape(-1, CHAIN_POOL_BINS).mean(axis=1), 1e-30))
    fine = pooled - ndimage.uniform_filter1d(pooled, CHAIN_SMOOTH_BINS, mode="nearest")
    keep = np.ones(len(fine), dtype=bool)
    for hz in notch:
        keep &= np.abs(pooled_f - hz) >= CHAIN_TONE_NOTCH_HZ
    shape = np.zeros(len(fine))
    spread = fine[keep].std()
    if spread > 1e-9:
        shape[keep] = (fine[keep] - fine[keep].mean()) / spread
    return shape


def load_chains() -> list[dict]:
    """Built-in chains shipped next to the program, then the user's learned ones."""
    chains, names = [], set()
    for folder, origin in ((CHAIN_BUILTIN_DIR, "builtin"), (CHAIN_USER_DIR, "user")):
        if not folder.is_dir():
            continue
        for file in sorted(folder.glob("*.json")):
            try:
                chain = json.loads(file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(tr(f"警告：无法读取加工链指纹 {file}：{exc}",
                         f"Warning: cannot read chain fingerprint {file}: {exc}"), file=sys.stderr)
                continue
            if chain.get("format") != CHAIN_FORMAT or chain.get("name") in names:
                continue
            names.add(chain["name"])
            chains.append({**chain, "origin": origin, "file": str(file)})
    return chains


def match_chain(freq: np.ndarray, psd: np.ndarray, fs: int, chain: dict) -> dict:
    level = 10 * np.log10(np.maximum(psd, 1e-30))
    tones = []
    for tone in chain["tones"]:
        prominence = tone_prominence(freq, level, tone["hz"])
        tones.append({"hz": tone["hz"], "minimum_db": tone["min_db"], "prominence_db": prominence,
                      "present": prominence is not None and prominence >= tone["min_db"]})
    template = np.asarray(chain["shape_template"], dtype=float)
    shape = chain_shape(freq, psd, fs, [t["hz"] for t in chain["tones"]])
    r = float(shape @ template / len(template)) if len(shape) == len(template) and len(shape) else None
    all_tones = len(tones) >= 2 and all(t["present"] for t in tones)
    strong = chain.get("shape_strong", CHAIN_SHAPE_STRONG)
    alone = chain.get("shape_only")  # None: this chain's shape is not distinctive enough on its own
    shape_ok = r is not None and r >= strong
    if all_tones and shape_ok:
        level_ = VERY_HIGH
    elif all_tones or (r is not None and alone is not None and r >= alone):
        level_ = HIGH
    else:
        level_ = None
    partial = level_ is None and (any(t["present"] for t in tones) or shape_ok)
    return {"name": chain["name"], "label": chain_field(chain, "label") or chain["name"], "origin": chain["origin"],
            "tones": tones, "shape_correlation": r, "shape_strong": strong, "shape_only": alone,
            "matched": level_ is not None, "level": level_, "partial": partial,
            "note": chain_field(chain, "note"), "source_max_hz": chain.get("source_max_hz")}


def chain_field(chain: dict, key: str) -> str:
    """A text field of a chain file in the output language ("label_en" etc.), else the default one."""
    return (LANG == "en" and chain.get(f"{key}_en")) or chain.get(key) or ""


def chain_check(info: dict, data: dict) -> dict:
    fs = info["sample_rate_hz"]
    if info["dsd_rate_hz"]:
        return {"performed": False, "results": [], "reason": tr(
            "DSD 文件不做加工链检查（目前的指纹都是 PCM 的）。",
            "DSD files are not checked against chains (all fingerprints are PCM).")}
    if fs < CHAIN_MIN_RATE:
        return {"performed": False, "results": [], "reason": tr(
            f"{khz(fs)} 没有超声频段，不做加工链检查。", f"{khz(fs)} has no ultrasonic band; no chain check.")}
    if data["chain"] is None:
        return {"performed": False, "results": [], "reason": tr(
            "有效声音太少，无法做加工链检查。", "Too little signal for a chain check.")}
    chains = [c for c in load_chains() if c["sample_rate_hz"] == fs]
    if not chains:
        return {"performed": False, "results": [], "reason": tr(
            f"没有 {khz(fs)} 的加工链指纹（可用 --learn-chain 学习）。",
            f"No chain fingerprint for {khz(fs)} (create one with --learn-chain).")}
    freq, psd = data["chain"]
    results = [match_chain(freq, psd, fs, c) for c in chains]
    order = {VERY_HIGH: 0, HIGH: 1, None: 2}
    results.sort(key=lambda m: (order[m["level"]], not m["partial"], -(m["shape_correlation"] or -1)))
    return {"performed": True, "reason": "", "results": results,
            "best": results[0] if results[0]["matched"] else None}


def analyze_spectrum(
    path: Path, max_seconds: float | None, n_fft: int = 32768, check_sr: bool = False
) -> tuple[dict, dict]:
    """Stream overlapping Hann windows; average *power*, never channel waveforms.

    Keep PSD sums in 5 dB RMS buckets so silence can be excluded after the
    recording's level is known. Memory is bounded independently of duration.
    Per-frame scalar metrics are pooled to at most 600 time columns.
    """
    meta = probe(path)
    fs, channels = meta["sample_rate_hz"], meta["channels"]
    expected = meta["duration_seconds"]
    if max_seconds is not None:
        expected = min(expected, max_seconds) if expected else max_seconds
    # The time grid is laid out before decoding; containers report duration reliably enough.
    timeline = expected if expected and expected > 0 else 3600.0

    hop = n_fft // 2
    freq = np.fft.rfftfreq(n_fft, 1 / fs)
    df = fs / n_fft
    window = signal.windows.hann(n_fft, sym=False)
    bucket_edges = np.arange(-160, 10, 5, dtype=float)
    sums = np.zeros((len(bucket_edges), channels, len(freq)))
    counts = np.zeros(len(bucket_edges), dtype=np.int64)
    # A compact, linear-frequency spectrogram and level timeline.
    n_time = max(1, min(600, math.ceil(timeline * fs / hop) - 1))
    spec_indices = np.array_split(np.arange(len(freq)), min(512, len(freq)))
    spec_starts = np.array([s[0] for s in spec_indices])
    spec_sizes = np.array([len(s) for s in spec_indices])
    spec_freq = np.array([np.mean(freq[s]) for s in spec_indices])
    heat = np.zeros((n_time, len(spec_indices)))
    time_counts = np.zeros(n_time, dtype=int)
    # For each RMS bucket and time column: count, audible and HF power.
    timeline_levels = np.zeros((len(bucket_edges), n_time, 3))
    masks = [(freq >= 20) & (freq < min(20000, fs / 2)), freq > 24000]
    frame_start = 0
    total_frames = 0
    decoded = 0
    peak = 0.0

    def consume(samples: np.ndarray, starts: np.ndarray) -> None:
        nonlocal total_frames
        # samples: [windows, channels, samples per window]
        # Taper a short final window over its actual length. Appending
        # zeros before tapering would create a false high-frequency edge.
        actual_length = samples.shape[-1]
        local_window = window if actual_length == n_fft else signal.windows.hann(actual_length, sym=False)
        scale = 1 / (fs * np.sum(local_window**2))
        rms_power = np.mean(samples**2, axis=(1, 2))
        levels = 10 * np.log10(np.maximum(rms_power, FLOOR))
        buckets = np.clip(
            np.floor((levels + 160) / 5).astype(int), 0, len(bucket_edges) - 1
        )
        centered = samples - samples.mean(axis=-1, keepdims=True)
        transformed = np.fft.rfft(centered * local_window, n=n_fft, axis=-1)
        powers = np.abs(transformed) ** 2 * scale
        powers[..., 1:-1] *= 2
        mono_power = powers.mean(axis=1)
        times = np.minimum((starts + actual_length / 2) / fs, timeline)
        columns = np.minimum((times / timeline * n_time).astype(int), n_time - 1)
        pooled = np.add.reduceat(mono_power, spec_starts, axis=-1) / spec_sizes
        audible = np.sum(mono_power[:, masks[0]], axis=1) * df
        ultrasonic = np.sum(mono_power[:, masks[1]], axis=1) * df
        for i, bucket in enumerate(buckets):
            sums[bucket] += powers[i]
            counts[bucket] += 1
            col = columns[i]
            heat[col] += pooled[i]
            time_counts[col] += 1
            timeline_levels[bucket, col] += (1, audible[i], ultrasonic[i])
        total_frames += len(samples)

    carry = np.empty((0, channels))
    lossy = None if meta["dsd_rate_hz"] else LossyScan(fs, meta["bits_per_sample"], expected)
    chain = ChainScan(fs) if check_sr and not meta["dsd_rate_hz"] and fs >= CHAIN_MIN_RATE else None
    with PcmStream(path, channels, seconds=max_seconds) as stream:
        while True:
            block = stream.read(hop * 16)
            if not len(block):
                break
            decoded += len(block)
            peak = max(peak, float(np.max(np.abs(block))))
            if lossy is not None:
                lossy.feed(block)
            if chain is not None:
                chain.feed(block)
            data = np.concatenate((carry, block), axis=0)
            if len(data) >= n_fft:
                frames = np.lib.stride_tricks.sliding_window_view(data, n_fft, axis=0)[::hop]
                starts = frame_start + np.arange(len(frames)) * hop
                consume(frames, starts)
                used = len(frames) * hop
                frame_start += used
                carry = data[used:].copy()
            else:
                carry = data
    if not decoded:
        raise ValueError(tr("文件为空，或指定的分析时长不足一个采样点。",
                            "The file is empty, or the analysis length is shorter than one sample."))

    # Cover the tail, including inputs shorter than one FFT window.
    if len(carry) > hop or not total_frames:
        consume(carry.T[None, ...], np.array([frame_start]))

    # Gate silence relative to the approximate 95th percentile frame RMS.
    cumulative = np.cumsum(counts)
    p95_bucket = int(np.searchsorted(cumulative, 0.95 * total_frames))
    gate = max(-80.0, float(bucket_edges[p95_bucket] - 35))
    active_buckets = bucket_edges >= gate
    active_count = int(counts[active_buckets].sum())
    selected = active_buckets if active_count else np.ones(len(counts), dtype=bool)
    selected_count = int(counts[selected].sum())
    psd_channels = sums[selected].sum(axis=0) / selected_count
    psd = psd_channels.mean(axis=0)
    levels_time = timeline_levels[selected].sum(axis=0)
    valid_times = levels_time[:, 0] > 0
    time_audible = np.divide(
        levels_time[:, 1], levels_time[:, 0],
        out=np.zeros(n_time), where=valid_times,
    )
    time_ultra = np.divide(
        levels_time[:, 2], levels_time[:, 0],
        out=np.zeros(n_time), where=valid_times,
    )
    corr_mask = valid_times & (time_audible > 1e-16) & (time_ultra > 1e-20)
    correlation = None
    if corr_mask.sum() >= 8:
        a = 10 * np.log10(time_audible[corr_mask])
        b = 10 * np.log10(time_ultra[corr_mask])
        if np.std(a) > 1 and np.std(b) > 1:
            correlation = float(np.corrcoef(a, b)[0, 1])
    info = {
        "path": str(path.resolve()), "container": meta["container"], "codec": meta["codec"],
        "bits_per_sample": meta["bits_per_sample"], "sample_format": meta["sample_format"],
        "sample_rate_hz": fs, "channels": channels, "dsd_rate_hz": meta["dsd_rate_hz"],
        "analysis_nyquist_hz": fs / 2 * (DSD_ANALYSIS_SHARE if meta["dsd_rate_hz"] else 1),
        "duration_seconds": meta["duration_seconds"] or decoded / fs,
        "analyzed_seconds": decoded / fs,
        "full_file_scanned": max_seconds is None or (
            meta["duration_seconds"] is not None and meta["duration_seconds"] <= max_seconds),
        "fft_size": n_fft, "frequency_resolution_hz": df,
        "window_count": total_frames, "active_window_count": active_count,
        "rms_gate_dbfs": gate, "peak_sample_dbfs": db(peak**2),
        "decoder_warnings": stream.warnings or None,
        "level_convention": "dBFS = 10 log10(mean square); full-scale sine RMS is -3.01 dBFS",
    }
    return info, {
        "freq": freq, "psd": psd, "psd_channels": psd_channels,
        "spectrogram": heat / np.maximum(time_counts[:, None], 1),
        "spectrogram_freq": spec_freq, "time_extent": timeline,
        "time": (np.arange(n_time) + 0.5) * timeline / n_time,
        "time_audible": time_audible, "time_ultra": time_ultra,
        "time_valid": valid_times, "envelope_correlation": correlation,
        "lossy": lossy.result(freq, psd, fs) if lossy is not None else None,
        "chain": chain.result() if chain is not None else None,
    }


def source_rates(onset: float, mid: float, width: float) -> list[int]:
    """Standard rates whose Nyquist can explain a filter starting to fall at `onset`.

    A filter passes at least ~80% of its Nyquist and cannot keep content far
    above it. Either the onset or the midpoint may be the better estimate:
    treble already sloping before the edge pulls the onset down, a lax filter
    pushes the midpoint up. A wide transition may also be a lax filter leaking
    past Nyquist, so a Nyquist just below the onset is accepted too. The rate
    whose Nyquist best matches the midpoint comes first.
    """
    low, high = SOURCE_NYQUIST_RATIO
    rates = [r for r in STANDARD_RATES
             if low <= onset / (r / 2) <= high or low <= mid / (r / 2) <= high
             or (width > EDGE_MAX_WIDTH_HZ and onset - width <= r / 2 <= onset)]
    return sorted(rates, key=lambda r: abs(mid / (r / 2) - SOURCE_TYPICAL_RATIO))


def binned_level(freq: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dB of mean power in ~EDGE_BIN_HZ bins; averaging first keeps FFT variance out of edges."""
    per = max(1, round(EDGE_BIN_HZ / (freq[1] - freq[0])))
    n = (len(freq) - 1) // per
    f = freq[1:1 + n * per].reshape(n, per).mean(axis=1)
    return f, 10 * np.log10(np.maximum(psd[1:1 + n * per].reshape(n, per).mean(axis=1), 1e-30))


def mirror_score(f: np.ndarray, level: np.ndarray, rate: int, top: float, nyquist: float):
    """Correlation of the passband below rate/2 with the spectrum mirrored above it.

    Upsampling images content at m to rate - m. Natural spectra fall with
    frequency, so they correlate negatively with their own mirror image.
    """
    m = f[(f >= max(1000, rate / 2 - 10000)) & (f <= top - 300)]
    m = m[rate - m <= 0.98 * nyquist]
    if len(m) < 10:
        return None, None
    passband, mirrored = np.interp(m, f, level), np.interp(rate - m, f, level)
    if np.std(passband) < IMAGE_MIN_SHAPE_DB or np.std(mirrored) == 0:
        return None, None
    return float(np.corrcoef(passband, mirrored)[0, 1]), float(np.median(passband - mirrored))


def find_imaging(freq: np.ndarray, psd: np.ndarray, fs: int) -> list[dict]:
    """Resampler images without a cliff: cheap (e.g. linear) interpolation mirrors the
    passband almost unattenuated, so the spectrum folds back up at the old Nyquist."""
    f, level = binned_level(freq, psd)
    found = []
    for rate in STANDARD_RATES:
        if rate >= fs:
            break
        corr, atten = mirror_score(f, level, rate, rate / 2, fs / 2)
        if corr is not None and corr >= IMAGE_STANDALONE_CORRELATION:
            found.append({"source_rate_hz": rate, "correlation": corr, "attenuation_db": atten})
    return found


def find_band_limits(freq: np.ndarray, psd: np.ndarray, fs: int, check_images: bool = True) -> list[dict]:
    """Steep, sustained drops in the averaged spectrum and the source rates each allows.

    A resampler or codec low-pass leaves a step from the content (or the old
    quantization/dither floor, so dark masters still show it) down to a lower
    floor. The step is a band limit when nothing above it comes back near the
    pre-edge level, when what comes back mirrors the passband (resampler
    imaging), or when a clear gap is followed only by rising noise (noise
    shaping, e.g. a PCM master converted through DSD).
    """
    f, level = binned_level(freq, psd)
    n = len(f)
    if n < 64:
        return []
    step = f[1] - f[0]
    nyquist = fs / 2

    # Two scales: steep filters at 0.3/1.5 kHz, lax ones (slow skirts) at 1/3 kHz.
    candidates = []
    for gap_hz, span_hz, max_width in [(300, 1500, EDGE_MAX_WIDTH_HZ), (1000, 3000, 2.5 * EDGE_MAX_WIDTH_HZ)]:
        gap, span = max(1, round(gap_hz / step)), max(2, round(span_hz / step))
        if n < 2 * (gap + span) + 2:
            continue
        # Median of each span-wide run; compare the run ending `gap` below a bin with the one starting above.
        run = np.median(np.lib.stride_tricks.sliding_window_view(level, span), axis=1)
        centers = np.arange(gap + span, n - gap - span)
        below, above = run[centers - gap - span], run[centers + gap + 1]
        for k in np.nonzero((below - above >= EDGE_BRICKWALL_DROP_DB) & (f[centers] >= EDGE_SEARCH_FROM_HZ))[0]:
            candidates.append((float(below[k] - above[k]), int(centers[k]), float(below[k]),
                               float(above[k]), gap + span, max_width))
    picked = []
    for cand in sorted(candidates, reverse=True):
        if all(abs(f[cand[1]] - f[p[1]]) > 2000 for p in picked):
            picked.append(cand)

    edges = []
    for d, c, b, a, reach, max_width in picked:
        lo, hi = c - reach, min(c + reach, n - 1)
        under = np.nonzero(level[lo:hi + 1] < (a + b) / 2)[0]
        if not len(under) or under[0] == 0 or b < EDGE_MIN_LEVEL_DB:
            continue  # Already below the midpoint at the window start: not a content-to-floor step.
        j = lo + under[0]
        # Midpoint of the transition, interpolated in dB between neighbouring bins.
        y0, y1 = level[j - 1], level[j]
        mid = f[j - 1] + float(np.clip((y0 - (a + b) / 2) / max(y0 - y1, 1e-9), 0, 1)) * step
        top = j
        while top > lo and level[top] < b - 0.1 * d:
            top -= 1
        bottom = j
        while bottom < hi and level[bottom] > a + 0.1 * d:
            bottom += 1
        width = float(f[bottom] - f[top])
        slope = d / math.log2(f[bottom] / f[top]) if bottom > top else math.inf
        if width > max_width or slope < EDGE_MIN_SLOPE_DB_PER_OCTAVE:
            continue  # A gentle analog or mastering roll-off, not a digital filter.
        core_top, core_bottom = j, j
        while core_top > top and level[core_top - 1] < b - 0.2 * d:
            core_top -= 1
        while core_bottom < bottom and level[core_bottom] > a + 0.2 * d:
            core_bottom += 1
        core = float(f[core_bottom] - f[core_top])
        rest_mask = (f >= f[bottom] + 500) & (f <= 0.98 * nyquist)
        rest = level[rest_mask]
        margin = float(b - np.percentile(rest, 90)) if len(rest) else None
        gap_region = level[rest_mask & (f < f[bottom] + 8500)]
        gap_margin = float(b - np.percentile(gap_region, 90)) if len(gap_region) >= 10 else None
        brickwall = (core <= EDGE_BRICKWALL_CORE_HZ and gap_margin is not None
                     and np.std(gap_region) <= EDGE_BRICKWALL_FLOOR_STD_DB)
        if d < EDGE_MIN_DROP_DB and not brickwall:
            continue
        # Margins are judged against the drop a brick wall needs, not the general one.
        need = EDGE_BRICKWALL_DROP_DB if brickwall else EDGE_EMPTY_MARGIN_DB
        rising = (gap_margin is not None and len(rest) >= 3 * len(gap_region)
                  and np.median(rest[-len(rest) // 3:]) - np.median(gap_region) >= 15)
        rates = source_rates(float(f[top]), float(mid), width)
        image_corr = image_atten = None
        if rates and check_images:
            image_corr, image_atten = mirror_score(f, level, rates[0], float(f[top]), nyquist)
        imaged = (image_corr is not None and image_corr >= IMAGE_MIN_CORRELATION
                  and image_atten >= EDGE_IMAGED_MARGIN_DB)
        noise_above = bool(rising and gap_margin >= need and (margin is None or margin < need))
        standard_cliff = bool(rates) and d >= EDGE_STANDARD_CLIFF_DB and margin is not None and margin >= EDGE_IMAGED_MARGIN_DB
        limit = margin is None or margin >= need or imaged or noise_above or standard_cliff
        edges.append({
            "frequency_hz": float(mid), "onset_hz": float(f[top]), "stop_hz": float(f[bottom]),
            "drop_db": d, "transition_width_hz": width, "core_transition_hz": core,
            "brick_wall": bool(brickwall), "slope_db_per_octave": float(slope),
            "pre_edge_density_db_per_hz": b, "empty_margin_db": margin, "gap_margin_db": gap_margin,
            "band_limit": bool(limit), "possible_source_rates_hz": rates,
            "image_correlation": image_corr, "image_attenuation_db": image_atten,
            "imaging_detected": bool(imaged), "rising_noise_above": noise_above,
            "weak_energy_above": bool(standard_cliff and not (margin >= need or imaged or noise_above)),
            "role": None,
        })
    # Both scales can see one transition; keep the deeper reading of overlapping ones.
    kept: list[dict] = []
    for e in sorted(edges, key=lambda e: -e["drop_db"]):
        if all(e["stop_hz"] + 300 < k["onset_hz"] or k["stop_hz"] + 300 < e["onset_hz"] for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e["frequency_hz"])
    # Above the source limit only a step at a higher standard rate is a further
    # band limit (an intermediate conversion); others end image bands or floors.
    limits = [e for e in kept if e["band_limit"]]
    ceiling = max(limits[0]["possible_source_rates_hz"], default=0) if limits else 0
    for e in kept:
        if not e["band_limit"]:
            e["role"] = "not_band_limit"
        elif e is limits[0]:
            e["role"] = "source_limit"
        elif (e["possible_source_rates_hz"] and min(e["possible_source_rates_hz"]) > ceiling
              and e["transition_width_hz"] <= EDGE_MAX_WIDTH_HZ):  # A resampler stage is steep.
            e["role"] = "intermediate_limit"
        else:
            e["role"] = "secondary_step"
            e["band_limit"] = False
    return kept


def dsd_label(rate: int) -> str:
    for base, family in [(44100, ""), (48000, tr("（48k 系）", " (48k family)"))]:
        if rate % base == 0:
            return f"DSD{rate // base}{family}"
    return f"DSD {rate / 1e6:g}MHz"


def dsd_noise_profile(freq: np.ndarray, psd: np.ndarray, limit: float) -> dict | None:
    """Shape of the modulator noise above the audio band, on a 1/6-octave grid.

    Noise shaping lifts quantization noise with frequency. A PCM band limit is
    only visible while the content still stands above that noise, so the
    profile also gives the highest frequency where an edge could be seen.
    A second hump (rise, fall by DSD_HUMP_DB, rise again) means two modulators:
    a lower-rate DSD converted through PCM and remodulated at this rate.
    """
    if limit <= 20000 * 2 ** (1 / 3):
        return None
    centers = 20000 * 2 ** (np.arange(int(math.log2(limit / 20000) * 6) + 1) / 6)
    levels = []
    for c in centers:
        mask = (freq >= c * 2 ** (-1 / 12)) & (freq < min(c * 2 ** (1 / 12), limit))
        levels.append(db(float(np.median(psd[mask]))) if mask.sum() >= 3 else math.nan)
    levels = np.array(levels)
    valid = ~np.isnan(levels)
    centers, levels = centers[valid], levels[valid]
    # Walk up in frequency: trough (content gives way to noise), rise, peak, dip, rise.
    # The first trough, not the global minimum, starts the noise: with two
    # modulators the dip between their humps can be the lowest point.
    trough, peak, dip, hump = 0, None, None, None
    for k in range(1, len(levels)):
        if peak is None:
            if levels[k] < levels[trough]:
                trough = k
            elif levels[k] - levels[trough] >= DSD_HUMP_DB:
                peak = k
        elif dip is None:
            if levels[k] > levels[peak]:
                peak = k
            elif levels[peak] - levels[k] >= DSD_HUMP_DB:
                dip = k
        elif levels[k] < levels[dip]:
            dip = k
        elif levels[k] - levels[dip] >= DSD_HUMP_DB:
            hump = {"peak_hz": float(centers[peak]), "dip_hz": float(centers[dip]),
                    "depth_db": float(levels[peak] - levels[dip])}
            break
    low = trough if peak is not None else int(np.argmin(levels))
    risen = np.nonzero(levels[low:] >= levels[low] + DSD_OBSERVABLE_RISE_DB)[0]
    observable = float(centers[low + risen[0]]) if len(risen) else float(limit)
    return {
        "floor_hz": float(centers[low]), "floor_db_per_hz": float(levels[low]),
        "observable_limit_hz": observable, "second_hump": hump,
        "profile": [[float(c), float(v)] for c, v in zip(centers, levels)],
    }


def measure_bands(info: dict, data: dict) -> dict:
    nyquist = info["analysis_nyquist_hz"]
    keep = data["freq"] <= nyquist
    freq, psd = data["freq"][keep], data["psd"][keep]
    psd_channels = data["psd_channels"][:, keep]
    dsd = info["dsd_rate_hz"] is not None
    audible = band_power(freq, psd, 20, min(20000, nyquist))
    total = band_power(freq, psd, 20, nyquist)
    bands = []
    for lo, hi in [(20, 20000), (20000, 22050), (22050, 24000),
                   (24000, 26000), (26000, 40000), (40000, 80000),
                   (80000, nyquist)]:
        hi = min(hi, nyquist)
        if hi <= lo:
            continue
        power = band_power(freq, psd, lo, hi)
        bands.append({"low_hz": lo, "high_hz": hi, "rms_dbfs": db(power),
                      "relative_to_20hz_20khz_db": relative_db(power, audible)})

    high = band_power(freq, psd, np.nextafter(24000.0, np.inf), nyquist) if nyquist > 24000 else 0.0
    high_edge_guard = band_power(freq, psd, 24500, nyquist) if nyquist > 24500 else 0.0
    high_guard = band_power(freq, psd, 26000, nyquist) if nyquist > 26000 else 0.0
    # 1 kHz pooled density reduces dependence on individual FFT-bin variance.
    densities = []
    for lo in np.arange(26000, nyquist - 1000, 1000):
        mask = (freq >= lo) & (freq < lo + 1000)
        densities.append(float(np.mean(psd[mask])))
    flatness = None
    if len(densities) >= 4 and high_guard > 1e-20:
        d = np.maximum(densities, FLOOR)
        flatness = float(np.exp(np.mean(np.log(d))) / np.mean(d))
    channel_high = [
        relative_db(band_power(freq, p, np.nextafter(24000.0, np.inf), nyquist),
                    band_power(freq, p, 20, min(20000, nyquist)))
        if nyquist > 24000 else None for p in psd_channels
    ]
    return {
        "bands": bands, "total_rms_dbfs": db(total),
        "above_24khz_rms_dbfs": db(high),
        "above_24khz_relative_db": relative_db(high, audible),
        "above_24_5khz_rms_dbfs": db(high_edge_guard),
        "above_24_5khz_relative_db": relative_db(high_edge_guard, audible),
        "above_26khz_relative_db": relative_db(high_guard, audible),
        "above_24khz_channel_relative_db": channel_high,
        # Images need a PCM upsampler; a DSD modulator leaves none, and its rising
        # noise would only fake a mirror of the falling treble.
        "band_edges": find_band_limits(freq, psd, int(2 * nyquist), check_images=not dsd),
        "imaging": [] if dsd else find_imaging(freq, psd, info["sample_rate_hz"]),
        "dsd_noise": dsd_noise_profile(freq, psd, nyquist) if dsd else None,
        "lossy": data["lossy"],
        "ultrasonic_spectral_flatness": flatness,
        "ultrasonic_audible_envelope_correlation": data["envelope_correlation"],
    }


def resampling_filter(fs: int) -> tuple[int, int, np.ndarray]:
    """Symmetric FIR: 20 kHz passband, 24 kHz stopband, ~110 dB rejection.

    The same common-rate FIR is used in both directions. scipy.resample_poly
    compensates its group delay; no loudness normalization is performed.
    """
    divisor = math.gcd(fs, REFERENCE_RATE)
    up, down = REFERENCE_RATE // divisor, fs // divisor
    common_rate = fs * up
    taps, beta = signal.kaiserord(110, 4000 / (common_rate / 2))
    taps += 1 - taps % 2
    fir = signal.firwin(taps, 22000, fs=common_rate, window=("kaiser", beta))
    return up, down, fir


def compare_roundtrip(path: Path, info: dict, seconds: float) -> dict:
    fs = info["sample_rate_hz"]
    if info["dsd_rate_hz"]:
        return {"performed": False, "reason": tr(
            "DSD 文件不做 48kHz 差分：超声区主要是调制噪声，差分结果没有意义。",
            "No 48 kHz round trip for DSD: its ultrasonic band is mostly modulator noise, so the difference means nothing.")}
    if fs <= REFERENCE_RATE or seconds <= 0 or info["analyzed_seconds"] < 0.25:
        return {"performed": False, "reason": tr(
            "采样率不高于 48kHz、片段过短，或已关闭差分。",
            "Sample rate is 48 kHz or lower, the excerpt is too short, or the round trip is turned off.")}
    up, down, fir = resampling_filter(fs)
    total_seconds = min(seconds, info["analyzed_seconds"])
    count = min(3, max(1, int(total_seconds // 2)))
    length = int(total_seconds / count * fs)
    max_start = max(0, int(info["analyzed_seconds"] * fs) - length)
    starts = np.linspace(0, max_start, count).astype(int) if count > 1 else np.array([max_start // 2])
    totals = np.zeros(4)
    per_segment = []
    trim = int(0.05 * fs)  # Remove independently processed excerpt boundaries.
    for start in starts:
        with PcmStream(path, info["channels"], start=int(start) / fs, seconds=length / fs) as stream:
            original = stream.read(length)
        if len(original) <= 2 * trim + 256:
            continue
        lower = signal.resample_poly(original, up, down, axis=0, window=fir, padtype="line")
        restored = signal.resample_poly(lower, down, up, axis=0, window=fir, padtype="line")
        restored = restored[:len(original)]
        original = original[trim:-trim]
        residual = original - restored[trim:-trim]
        n = min(32768, len(original))
        f, p = signal.welch(original, fs, nperseg=n, axis=0)
        _, r = signal.welch(residual, fs, nperseg=n, axis=0)
        original_power = float(np.mean(original**2))
        error_power = float(np.mean(residual**2))
        audible_power = band_power(f, p.T, 20, 20000)
        audible_error = band_power(f, r.T, 20, 20000)
        weight = len(original)
        totals += np.array([original_power, error_power, audible_power, audible_error]) * weight
        per_segment.append({
            "start_seconds": int(start) / fs, "measured_seconds": weight / fs,
            "residual_relative_db": relative_db(error_power, original_power),
            "audible_residual_relative_db": relative_db(audible_error, audible_power),
        })
    if not per_segment:
        return {"performed": False, "reason": tr("选段解码后过短，未执行差分。",
                                                  "The decoded excerpts were too short; no round trip.")}
    measured = sum(s["measured_seconds"] for s in per_segment)
    samples = measured * fs
    return {
        "performed": True, "reference_rate_hz": REFERENCE_RATE,
        "filter": "linear-phase Kaiser FIR; 20 kHz passband / 24 kHz stopband; target 110 dB rejection",
        "normalization": "none", "boundary_trim_seconds_each_end": 0.05,
        "measured_seconds": measured, "segments": per_segment,
        "residual_relative_db": relative_db(totals[1], totals[0]),
        "residual_rms_dbfs": db(totals[1] / samples),
        "audible_residual_relative_db": relative_db(totals[3], totals[2]),
        "audible_residual_rms_dbfs": db(totals[3] / samples),
    }


def khz(hz: float) -> str:
    return f"{hz / 1000:g}kHz"


def rates_text(rates: list[int]) -> str:
    others = "/".join(khz(r) for r in rates[1:])
    return khz(rates[0]) + (tr(f"（也可能 {others}）", f" (or {others})") if len(rates) > 1 else "")


def above_edge(edge: dict) -> str:
    if edge["rising_noise_above"]:
        return tr("截止以上先空出一段，之后只有随频率上升的噪声（噪声整形特征，例如经 DSD 转换）",
                  "above the cut there is a gap, then only noise rising with frequency (noise shaping, e.g. from a DSD conversion)")
    if edge["imaging_detected"]:
        return tr("截止以上的残留与通带呈镜像，像是升频滤波器留下的镜像",
                  "what remains above the cut mirrors the passband, like images left by an upsampling filter")
    if edge["weak_energy_above"]:
        return tr("截止以上仍有较弱的能量（可能是升频镜像或后加的噪声）",
                  "some weak energy remains above the cut (upsampling images or noise added later)")
    return tr("截止以上几乎为空", "almost nothing above the cut")


def dsd_verdict(info: dict, metrics: dict) -> tuple[str, str]:
    """PCM origin shows as a band limit below the noise-shaping rise; DSD-rate
    upsampling as a second noise hump left by the first modulator."""
    label = dsd_label(info["dsd_rate_hz"])
    noise = metrics["dsd_noise"]
    hump = noise and noise["second_hump"]
    source = next((e for e in metrics["band_edges"] if e["role"] == "source_limit"), None)
    if info["active_window_count"] < 3 or info["analyzed_seconds"] < 0.5:
        return "insufficient_signal", tr("有效信号过少或过短，无法可靠判断。",
                                         "Too little or too short a signal for a reliable verdict.")
    if source is not None:
        where = tr(f"内容约在 {source['frequency_hz'] / 1000:.1f}kHz 陡降，{above_edge(source)}；",
                   f"Content drops steeply at about {source['frequency_hz'] / 1000:.1f} kHz; {above_edge(source)}. ")
        if source["frequency_hz"] < LOSSY_EDGE_BELOW_HZ:
            code, text = "dsd_lossy_origin", where + tr(f"疑似由有损或严重限带的来源转为 {label}。",
                                                        f"Probably converted to {label} from a lossy or heavily band-limited source.")
        elif source["possible_source_rates_hz"]:
            code = "dsd_pcm_origin"
            rates = rates_text(source["possible_source_rates_hz"])
            text = where + tr(f"与 {rates} PCM 来源转为 {label} 一致。",
                              f"Consistent with {rates} PCM converted to {label}.")
        else:
            code, text = "dsd_band_limited_nonstandard", where + tr(
                "该位置不对应常见采样率，可能是母带低通。",
                "The position matches no common sample rate; possibly a mastering low-pass.")
    elif hump:
        code = "dsd_remodulated"
        text = ""
    else:
        seen = noise["observable_limit_hz"] if noise else info["analysis_nyquist_hz"]
        higher = next((r for r in STANDARD_RATES if r / 2 >= seen), None)
        code = "dsd_no_pcm_edge"
        floor_rate = khz(higher) if higher else tr("该频率两倍", "twice that frequency")
        text = tr(f"在约 {seen / 1000:.0f}kHz 以下未见 PCM 截止；可能是原生 DSD，"
                  f"也可能来自不低于 {floor_rate} 的 PCM（例如 DXD 352.8kHz），这超出了可检出范围。",
                  f"No PCM band limit below about {seen / 1000:.0f} kHz: native DSD, or PCM at {floor_rate} "
                  "or higher (e.g. DXD 352.8 kHz), which is beyond what can be seen.")
    if hump:
        also = tr("另：", "Also: ") if text else ""
        text += (" " if text and LANG != "zh" else "") + also + tr(
            f"调制噪声在约 {hump['peak_hz'] / 1000:.0f}kHz 先隆起，回落 {hump['depth_db']:.0f}dB 后再上升，"
            f"像是两级调制：低倍率 DSD 经 PCM 中转后重新调制为 {label}（DSD 升频）。",
            f"the modulator noise humps at about {hump['peak_hz'] / 1000:.0f} kHz, falls {hump['depth_db']:.0f} dB, "
            f"then rises again, like two modulators: a lower-rate DSD re-modulated to {label} via PCM (DSD upsampling).")
    return code, text


def codec_hint(cutoff: float) -> str:
    """Encoders whose default low-pass lands near `cutoff` (LAME, FFmpeg AAC, Opus, Vorbis)."""
    for limit, zh, en in [(16700, "MP3 128kbps 及以下", "MP3 at 128 kbps or less"),
                          (17600, "MP3/AAC 约 128–160kbps", "MP3/AAC at about 128-160 kbps"),
                          (19000, "MP3 约 192kbps 或 VBR V2", "MP3 at about 192 kbps or VBR V2"),
                          (19700, "MP3 约 256kbps", "MP3 at about 256 kbps"),
                          (LOSSY_MAX_CUTOFF_HZ, "MP3 320kbps、Opus 或高码率 Vorbis", "MP3 320 kbps, Opus or high-rate Vorbis")]:
        if cutoff < limit:
            return tr(zh, en)
    return tr("高码率有损编码", "a high-rate lossy codec")


def lossy_findings(lossy: dict | None) -> tuple[bool, str]:
    """Whether the short-window checks point at a lossy source, and a sentence on why."""
    if not lossy:
        return False, ""
    parts = []
    if lossy["cliff_detected"]:
        parts.append(tr(f"平均谱在约 {lossy['cliff_hz'] / 1000:.1f}kHz 陡降 {lossy['cliff_drop_db']:.0f}dB，"
                        f"位置接近 {codec_hint(lossy['cliff_hz'])} 的低通",
                        f"the average spectrum drops {lossy['cliff_drop_db']:.0f} dB at about "
                        f"{lossy['cliff_hz'] / 1000:.1f} kHz, near the low-pass of {codec_hint(lossy['cliff_hz'])}"))
    if lossy["holes_detected"]:
        parts.append(tr(f"16kHz 以上有 {lossy['holes_percent']:.1f}% 的时频点出现空洞"
                        "（校准用的 CD 原盘都低于 0.1%），像 MP3/Vorbis 的频谱空洞",
                        f"{lossy['holes_percent']:.1f}% of the time-frequency cells above 16 kHz are holes "
                        "(every CD rip used for calibration was below 0.1%), like MP3/Vorbis spectral holes"))
    return bool(parts), sep().join(parts)


def format_label(info: dict) -> str:
    rate = info["dsd_rate_hz"]
    if rate:
        return dsd_label(rate)
    bits = f"/{info['bits_per_sample']}bit" if info["bits_per_sample"] else ""
    return f"{info['sample_rate_hz'] / 1000:g}kHz{bits}"


def verdict(info: dict, metrics: dict) -> dict:
    common = tr("单文件分析无法证明原生录音采样率，也无法证明听感明显更好。",
                "A single file cannot prove the native recording rate, nor that it sounds better.")
    if info["dsd_rate_hz"]:
        code, text = dsd_verdict(info, metrics)
        return {"code": code, "summary": text,
                "limitation": tr("单文件分析无法证明录音的原始格式；许多正规 SACD 也经过 PCM（如 DXD）剪辑。",
                                 "A single file cannot prove the original recording format; many legitimate "
                                 "SACDs are edited in PCM (e.g. DXD).")}
    fs = info["sample_rate_hz"]
    source = next((e for e in metrics["band_edges"] if e["role"] == "source_limit"), None)
    # An edge implies a lower source only if no plausible rate reaches the file's own rate.
    def below_own(e: dict) -> bool:
        return bool(e["possible_source_rates_hz"]) and max(e["possible_source_rates_hz"]) < fs
    lossy = source is not None and source["frequency_hz"] < LOSSY_EDGE_BELOW_HZ
    mirror = metrics["imaging"][0] if metrics["imaging"] else None
    above = above_edge(source) if source is not None else ""
    lossy_hit, lossy_why = lossy_findings(metrics["lossy"])

    def drop_at(edge: dict) -> str:
        return tr(f"内容约在 {edge['frequency_hz'] / 1000:.1f}kHz 陡降，{above}；",
                  f"Content drops steeply at about {edge['frequency_hz'] / 1000:.1f} kHz; {above}. ")

    if info["active_window_count"] < 3 or info["analyzed_seconds"] < 0.5:
        code, text = "insufficient_signal", tr("有效信号过少或过短，无法可靠判断。",
                                               "Too little or too short a signal for a reliable verdict.")
    elif fs <= REFERENCE_RATE:
        found = metrics["lossy"]
        if lossy_hit:
            code = "lossy_suspect"
            text = tr(f"{lossy_why}。疑似由有损文件转成无损格式；也可能是母带主动低通或降噪处理。",
                      f"{lossy_why[:1].upper()}{lossy_why[1:]}. Probably converted from a lossy file to a lossless "
                      "format; a deliberate mastering low-pass or noise reduction can look the same.")
        elif found is None:
            code, text = "insufficient_signal", tr("响亮的片段太少，无法做有损检测。",
                                                   "Too few loud passages for the lossy check.")
        else:
            code = "no_lossy_sign"
            share, top = found["loud_frames_above_20khz_percent"], found["frame_cutoff_p95_hz"] / 1000
            text = tr(f"未见有损编码的截止或频谱空洞（{share:.0f}% 的响亮片段有 20kHz 以上的内容，最高约 {top:.1f}kHz）。",
                      f"No lossy-codec cut or spectral holes ({share:.0f}% of loud passages reach above 20 kHz, "
                      f"up to about {top:.1f} kHz). ")
            if found["holes_percent"] is not None and found["holes_percent"] >= LOSSY_HOLES_NOTE_PERCENT:
                text += tr(f"16kHz 以上有少量空洞（{found['holes_percent']:.2f}%），接近 {LOSSY_HOLES_PERCENT:g}% 的门槛，"
                           "可以看看频谱图里是否有成块缺失的高频。",
                           f"A few holes above 16 kHz ({found['holes_percent']:.2f}%), close to the "
                           f"{LOSSY_HOLES_PERCENT:g}% threshold; check the spectrogram for missing blocks of treble. ")
            text += tr("高码率 AAC（256kbps 以上）和 Opus 的频谱可能与无损相同，无法据此排除。",
                       "High-rate AAC (256 kbps and up) and Opus can look lossless, so they cannot be ruled out.")
    elif lossy:
        code = "band_limited_low"
        text = drop_at(source) + tr(f"疑似有损来源或严重限带的母带，后被转为 {khz(fs)}。",
                                    f"Probably a lossy or heavily band-limited source, later converted to {khz(fs)}.")
    elif (source is not None and below_own(source)) or mirror:
        code = "upsampled"
        if mirror and not (source is not None and below_own(source)
                           and source["possible_source_rates_hz"][0] <= mirror["source_rate_hz"]):
            origin = mirror["source_rate_hz"]
            text = tr(f"{khz(origin / 2)} 以上的频谱与其下方呈镜像（相关 {mirror['correlation']:.2f}，"
                      f"约低 {mirror['attenuation_db']:.0f} dB），是升频滤波不足留下的镜像；"
                      f"与 {khz(origin)} 来源升频到 {khz(fs)} 一致。",
                      f"The spectrum above {khz(origin / 2)} mirrors the one below (correlation "
                      f"{mirror['correlation']:.2f}, about {mirror['attenuation_db']:.0f} dB lower): images left by a "
                      f"weak upsampling filter. Consistent with a {khz(origin)} source upsampled to {khz(fs)}.")
        else:
            origin = source["possible_source_rates_hz"][0]
            rates = rates_text(source["possible_source_rates_hz"])
            text = drop_at(source) + tr(f"与 {rates} 来源升频到 {khz(fs)} 一致。",
                                        f"Consistent with a {rates} source upsampled to {khz(fs)}.")
        for stage in metrics["band_edges"]:
            if (stage["role"] in ("source_limit", "intermediate_limit") and below_own(stage)
                    and min(stage["possible_source_rates_hz"]) > origin):
                rates = rates_text(stage["possible_source_rates_hz"])
                text += tr(f"约 {stage['frequency_hz'] / 1000:.1f}kHz 处另有截止，可能经过 {rates} 中间转换。",
                           f" Another cut at about {stage['frequency_hz'] / 1000:.1f} kHz suggests an "
                           f"intermediate {rates} step.")
    elif source is not None and not source["possible_source_rates_hz"] and source["frequency_hz"] < 0.9 * fs / 2:
        code = "band_limited_nonstandard"
        text = drop_at(source) + tr("该位置不对应常见采样率，可能是母带低通或非常规转换。",
                                    "The position matches no common sample rate; possibly a mastering low-pass "
                                    "or an unusual conversion.")
    else:
        relative = metrics["above_24khz_relative_db"]
        negligible = relative is None or relative < -75 or metrics["above_24khz_rms_dbfs"] < -95
        flatness = metrics["ultrasonic_spectral_flatness"]
        corr = metrics["ultrasonic_audible_envelope_correlation"]
        guarded_relative = metrics["above_24_5khz_relative_db"]
        clear_above_edge = (guarded_relative is not None and guarded_relative > -65
                            and metrics["above_24_5khz_rms_dbfs"] > -95)
        if negligible:
            code = "little_ultrasonic_energy"
            text = tr("未检出显著的 >24kHz 能量，也未见明确的截止；未见 192kHz 的额外频宽价值。",
                      "No significant energy above 24 kHz and no clear cut; the high rate adds no bandwidth.")
        elif flatness is not None and flatness > 0.5 and (corr is None or abs(corr) < 0.25):
            code = "ultrasonic_noise_like"
            text = tr("存在超声能量，呈宽带噪声特征；不能据此认定有更多音乐信息。",
                      "There is ultrasonic energy, but it looks like broadband noise, not more music.")
        elif clear_above_edge:
            code = "ultrasonic_content"
            text = tr("检测到超出 48kHz 频宽的能量，且未见升频截止；其音乐意义及可听优势仍不确定。",
                      "Energy beyond the 48 kHz bandwidth and no upsampling cut; its musical value and "
                      "audibility are still uncertain.")
        else:
            code = "uncertain"
            text = tr("超声能量较弱或证据混合，无法可靠判断是否升采样。",
                      "Weak ultrasonic energy or mixed evidence; upsampling cannot be judged reliably.")
    if fs > REFERENCE_RATE and lossy_hit and code in ("upsampled", "band_limited_low", "band_limited_nonstandard"):
        text += tr(f"另：{lossy_why}，来源可能是有损文件。", f" Also, {lossy_why}: the source may be a lossy file.")
    if fs <= REFERENCE_RATE:
        common = tr("单文件分析只能找有损编码留下的痕迹，不能证明来源一定是无损母带。",
                    "A single file only shows traces a lossy codec leaves; it cannot prove a lossless master.")
    chain = (metrics.get("chain_check") or {}).get("best")
    if chain:
        text += tr(f"超声指纹与加工链「{chain['label']}」吻合（{chain_evidence(chain)}），超声部分可能不是原始录音。",
                   f" The ultrasonic fingerprint matches processing chain \"{chain['label']}\" "
                   f"({chain_evidence(chain)}); the ultrasonic band may not come from the original recording.")
    return {"code": code, "summary": text, "limitation": common, "chain": chain["name"] if chain else None}


def chain_evidence(match: dict) -> str:
    parts = [tr(f"{t['hz'] / 1000:g}kHz 尖峰高出 {t['prominence_db']:.1f}dB",
                f"{t['hz'] / 1000:g} kHz tone {t['prominence_db']:.1f} dB above its surroundings")
             for t in match["tones"] if t["present"]]
    if match["shape_correlation"] is not None:
        parts.append(tr(f"超声细节形状相关 {match['shape_correlation']:.2f}",
                        f"fine ultrasonic shape correlation {match['shape_correlation']:.2f}"))
    return tr("，", ", ").join(parts)


# Verdicts whose "looks real" reading a chain match overturns.
CHAIN_OVERRIDES = {"ultrasonic_content", "ultrasonic_noise_like", "uncertain", "little_ultrasonic_energy"}


def number(value: float | None, digits: int = 1) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def edge_line(e: dict, fs: int) -> str:
    f, drop, width = e["frequency_hz"] / 1000, e["drop_db"], e["transition_width_hz"] / 1000
    head = tr(f"  约 {f:.2f} kHz：下降 {drop:.0f} dB，过渡带 {width:.1f} kHz，",
              f"  About {f:.2f} kHz: drops {drop:.0f} dB over {width:.1f} kHz; ")
    if e["role"] == "secondary_step":
        return head + tr("位于来源截止以上，不对应常见采样率，视为镜像带边缘或底噪变化。",
                         "above the source limit and at no common rate: an image-band edge or a floor change.")
    margin = number(e["empty_margin_db"], 0)
    if not e["band_limit"]:
        return head + tr(f"以上仍有接近的能量（余量 {margin} dB），不视为频宽上限。",
                         f"similar energy continues above (margin {margin} dB), so not a band limit.")
    if e["rising_noise_above"]:
        head += tr(f"其上 8kHz 内比截止前低 ≥{number(e['gap_margin_db'], 0)} dB，再往上是上升的噪声；",
                   f"the 8 kHz above is >= {number(e['gap_margin_db'], 0)} dB lower, then noise rises; ")
    elif e["empty_margin_db"] is None:
        head += tr("其上已到分析上限；", "the analysis range ends above it; ")
    elif e["weak_energy_above"]:
        head += tr(f"以上仍有较弱的能量（低 {margin} dB），截止足够深，仍视为频宽上限；",
                   f"weaker energy remains above ({margin} dB lower), but the cut is deep enough to be a band limit; ")
    else:
        head += tr(f"以上比截止前低 ≥{margin} dB；", f">= {margin} dB lower above; ")
    rates = e["possible_source_rates_hz"]
    if e["frequency_hz"] < LOSSY_EDGE_BELOW_HZ:
        tail = tr("低于常见采样率滤波位置，疑似有损或限带来源", "below any common rate's filter: a lossy or band-limited source")
    elif not rates:
        tail = tr("不对应常见采样率", "matches no common sample rate")
    elif max(rates) < fs:
        tail = tr(f"对应来源 {rates_text(rates)}", f"points to a {rates_text(rates)} source")
    else:
        tail = tr(f"与 {rates_text(rates)} 相容，可能是本采样率自身的滤波",
                  f"compatible with {rates_text(rates)}, possibly this rate's own filter")
    if e["image_correlation"] is not None:
        tail += tr(f"；镜像相关 {e['image_correlation']:.2f}", f"; image correlation {e['image_correlation']:.2f}")
        if e["imaging_detected"]:
            tail += tr(f"（检出镜像，约低 {e['image_attenuation_db']:.0f} dB）",
                       f" (images found, about {e['image_attenuation_db']:.0f} dB lower)")
    return head + tail + tr("。", ".")


# Plain-language grades. They rank the strength of the evidence; they are not
# measured probabilities (no labelled real-world corpus exists to calibrate those).
# Codes are stored in report.json; level_text() gives the words.
VERY_HIGH, HIGH, UNSURE, LOW, VERY_LOW, UNKNOWN = "very_high", "high", "unsure", "low", "very_low", "unknown"
LEVEL_WORDS = {VERY_HIGH: ("很高", "very high"), HIGH: ("较高", "high"), UNSURE: ("说不准", "unsure"),
               LOW: ("较低", "low"), VERY_LOW: ("很低", "very low"), UNKNOWN: ("无法判断", "cannot tell")}


def level_text(level: str | None) -> str:
    return tr(*LEVEL_WORDS[level]) if level in LEVEL_WORDS else ""


def plain_summary(info: dict, metrics: dict, verdict_: dict) -> dict:
    """A headline and a few graded answers for readers who are not audio engineers."""
    code, fs = verdict_["code"], info["sample_rate_hz"]
    dsd = info["dsd_rate_hz"] is not None
    label = format_label(info)
    lossy = metrics["lossy"]
    lossy_hit, _ = lossy_findings(lossy)
    edges = metrics["band_edges"]
    source = next((e for e in edges if e["role"] == "source_limit"), None)
    mirror = metrics["imaging"][0] if metrics["imaging"] else None
    corr = metrics["ultrasonic_audible_envelope_correlation"]
    chain = (metrics.get("chain_check") or {}).get("best")
    items: list[dict] = []

    def add(question: str, level: str, why: str) -> None:
        items.append({"question": question, "level": level, "level_text": level_text(level), "why": why})

    def rate_words(rates: list[int]) -> str:
        return khz(rates[0]) if len(rates) == 1 else tr(f"{khz(rates[0])} 或 {khz(rates[1])}",
                                                        f"{khz(rates[0])} or {khz(rates[1])}")

    def clean_cut(edge: dict) -> bool:
        return edge["drop_db"] >= 60 and (edge["empty_margin_db"] or 0) >= 40

    cut = f"{source['frequency_hz'] / 1000:.1f}" if source else ""

    # 1) Lossy compression (MP3, AAC ...)
    q_lossy = tr("经过有损压缩（MP3/AAC 等）", "Made from a lossy file (MP3, AAC ...)")
    if code in ("band_limited_low", "dsd_lossy_origin"):
        add(q_lossy, HIGH, tr(f"声音在约 {cut}kHz 以上被整齐地切掉了，这是有损格式为了省空间的常见做法。",
                              f"Everything above about {cut} kHz is cut off cleanly, which lossy formats do to save space."))
    elif lossy_hit:
        reasons = []
        if lossy["cliff_detected"]:
            at = f"{lossy['cliff_hz'] / 1000:.1f}"
            reasons.append(tr(f"高频在约 {at}kHz 处被一刀切断，位置和 {codec_hint(lossy['cliff_hz'])} 的习惯一致",
                              f"the treble is cut off sharply at about {at} kHz, where {codec_hint(lossy['cliff_hz'])} usually cuts"))
        if lossy["holes_detected"]:
            reasons.append(tr("高频里有很多一小块一小块的缺口，这是有损编码丢掉细节时留下的",
                              "the treble is full of small gaps, left where a lossy codec threw detail away"))
        why = sep().join(reasons) + tr("。", ".")
        add(q_lossy, VERY_HIGH if len(reasons) == 2 else HIGH, why[:1].upper() + why[1:])
    elif lossy is None or dsd:
        if not dsd:
            add(q_lossy, UNKNOWN, tr("能用来判断的响亮片段太少。", "Too few loud passages to judge."))
    elif code == "ultrasonic_content" and corr is not None and corr >= 0.5 and not chain:
        add(q_lossy, VERY_LOW, tr("有损格式会把 20kHz 左右以上的声音切掉，而这个文件的高频一直延伸到 24kHz 以上，还跟着音乐起伏。",
                                  "Lossy formats cut everything above about 20 kHz; this file's treble goes past "
                                  "24 kHz and moves with the music."))
    elif lossy["holes_percent"] is not None and lossy["holes_percent"] >= LOSSY_HOLES_NOTE_PERCENT:
        add(q_lossy, UNSURE, tr("没有明显的切断，但高频里有少量缺口，接近判断门槛，建议看一下频谱图。",
                                "No clear cut, but a few gaps in the treble, close to the threshold; have a look at the spectrogram."))
    else:
        add(q_lossy, LOW, tr("没找到有损格式常见的高频切断和缺口。256kbps 以上的高码率 AAC 用这种方法看不出来，所以不能说完全没有。",
                             "No sign of the treble cut or gaps lossy formats leave. AAC at 256 kbps and above cannot "
                             "be seen this way, so it cannot be ruled out completely."))

    # 2) Was the sample rate raised?
    q_up = tr(f"由低采样率升频到 {khz(fs)}", f"Upsampled to {khz(fs)} from a lower rate")
    if dsd:
        pass
    elif fs <= REFERENCE_RATE:
        add(tr("采样率被提升过", "Sample rate was raised"), VERY_LOW,
            tr(f"{khz(fs)} 本身就是 CD/普通无损的规格，不存在“假高解析”的问题。",
               f"{khz(fs)} is the ordinary CD/lossless rate, so there is no \"fake hi-res\" question."))
    elif code == "upsampled":
        if mirror and not (source and source["possible_source_rates_hz"]):
            add(q_up, HIGH, tr(f"高频是把 {khz(mirror['source_rate_hz'] / 2)} 以下的声音“镜像”复制上去的，这是廉价升频留下的痕迹；"
                               f"原始采样率约 {khz(mirror['source_rate_hz'])}。",
                               f"The treble is a mirror copy of what lies below {khz(mirror['source_rate_hz'] / 2)}, a trace "
                               f"of cheap upsampling; the original rate is about {khz(mirror['source_rate_hz'])}."))
        else:
            rates = source["possible_source_rates_hz"]
            both = tr("；两者分不清", "; the two cannot be told apart") if len(rates) > 1 else ""
            add(q_up, VERY_HIGH if clean_cut(source) else HIGH,
                tr(f"声音只到约 {cut}kHz，再往上几乎是空的，正好是 {rate_words(rates)} 录音能记录的上限；"
                   f"原始采样率约 {rate_words(rates)}{both}。",
                   f"The sound stops at about {cut} kHz and is nearly empty above, exactly the limit of a "
                   f"{rate_words(rates)} recording; the original rate is about {rate_words(rates)}{both}."))
    elif code == "band_limited_low":
        add(tr(f"由低规格来源提升到 {khz(fs)}", f"Raised to {khz(fs)} from a low-quality source"), HIGH,
            tr(f"声音只到约 {cut}kHz，连 CD 的水平都不到。", f"The sound stops at about {cut} kHz, below even CD."))
    elif code == "ultrasonic_content":
        level = HIGH if corr is not None and corr >= 0.5 else UNSURE
        follow = (tr("，而且跟着音乐一起起伏，像是真的乐器高频", ", and it moves with the music like real instrument overtones")
                  if level == HIGH else
                  tr("，但看不出它是否跟着音乐变化，也可能只是噪声", ", but it is unclear whether it follows the music; it may be noise"))
        add(q_up, LOW if level == HIGH else UNSURE,
            tr(f"文件里有 24kHz 以上的声音（CD 和 48kHz 录音记录不到这一段）{follow}。"
               "原始采样率应高于 48kHz，但是 96kHz 还是 192kHz 无法细分。",
               f"There is sound above 24 kHz (CD and 48 kHz recordings cannot hold it){follow}. "
               "The original rate should be above 48 kHz; 96 vs 192 kHz cannot be told."))
    elif code == "ultrasonic_noise_like":
        add(q_up, UNSURE, tr("24kHz 以上有声音，但更像均匀的噪声而不是音乐，无法说明是真高解析。",
                             "There is sound above 24 kHz, but it is more like even noise than music, so it proves nothing."))
    elif code == "little_ultrasonic_energy":
        add(tr("实际内容超过 48kHz 规格", "Holds more than a 48 kHz file can"), LOW,
            tr("24kHz 以上几乎没有声音，高采样率并没有带来额外的内容；但也没看到明显的升频痕迹。",
               "There is almost nothing above 24 kHz, so the high rate adds nothing; but no clear sign of upsampling either."))
    elif code == "band_limited_nonstandard":
        add(q_up, UNSURE, tr(f"声音在约 {cut}kHz 被切断，这个位置不对应常见的采样率，可能是母带处理。",
                             f"The sound is cut at about {cut} kHz, which matches no common rate; perhaps mastering."))
    else:
        add(q_up, UNKNOWN, tr("证据不足。", "Not enough evidence."))

    # 2b) Known processing chain (--check-sr): the ultrasonic band was made, not recorded.
    if chain and not dsd:
        if code in CHAIN_OVERRIDES:
            items.pop()
            limit = chain["source_max_hz"]
            if limit:
                add(tr(f"实际是 {khz(limit)} 及以下的音源，被加工成 {khz(fs)}",
                       f"Really a source of {khz(limit)} or less, processed into {khz(fs)}"), chain["level"],
                    tr(f"超声部分像是加工时加上的，不是原始录音里的；这条加工链的原始音源大多不高于 {khz(limit)}。",
                       "The ultrasonic band looks added during processing rather than recorded; this chain's "
                       f"sources are mostly {khz(limit)} or lower."))
            else:
                add(q_up, UNSURE, tr("超声部分可能是加工出来的，不能再当作真高解析的证据；原始采样率看不出来。",
                                     "The ultrasonic band may have been added in processing, so it no longer counts as "
                                     "evidence of real hi-res; the original rate cannot be seen."))
        why = sep().join(filter(None, [chain_plain(chain), chain["note"]])) + tr("。", ".")
        add(tr(f"经过已知加工链「{chain['label']}」", f"Went through known processing chain \"{chain['label']}\""),
            chain["level"], why[:1].upper() + why[1:])

    # 3) DSD origin
    if dsd:
        q_pcm = tr("由普通 PCM 音频转成 DSD", "Converted to DSD from ordinary PCM")
        if code == "dsd_pcm_origin":
            rates = source["possible_source_rates_hz"]
            cd = tr("（多半是 CD 母带）", " (most likely a CD master)") if rates[0] == 44100 else ""
            add(q_pcm, VERY_HIGH if source["rising_noise_above"] and clean_cut(source) else HIGH,
                tr(f"声音只到约 {cut}kHz，正好是 {rate_words(rates)} PCM 的上限；来源约为 {rate_words(rates)}{cd}。",
                   f"The sound stops at about {cut} kHz, exactly the limit of {rate_words(rates)} PCM; "
                   f"the source is about {rate_words(rates)}{cd}."))
        elif code == "dsd_lossy_origin":
            add(q_pcm, HIGH, tr("而且来源的频宽比 CD 还低。", "And the source has less bandwidth than a CD."))
        elif code == "dsd_no_pcm_edge":
            noise = metrics["dsd_noise"]
            seen = noise["observable_limit_hz"] if noise else info["analysis_nyquist_hz"]
            higher = next((r for r in STANDARD_RATES if r / 2 >= seen), None)
            ruled = [r for r in STANDARD_RATES if r / 2 < seen]
            checked = (tr(f"{khz(ruled[-1])} 及以下", f"{khz(ruled[-1])} or lower") if ruled
                       else tr("常见规格", "common-rate"))
            top = khz(higher) if higher else tr("更高规格", "higher-rate")
            add(q_pcm, UNSURE, tr(f"没发现 {checked} 的 PCM 留下的切断；可能是原生 DSD，"
                                  f"也可能来自 {top} 及以上的高解析 PCM（这种看不出来）。",
                                  f"No cut left by {checked} PCM; it may be native DSD, or come from hi-res PCM at "
                                  f"{top} or above (which cannot be seen)."))
        elif code == "dsd_band_limited_nonstandard":
            add(q_pcm, UNSURE, tr(f"声音在约 {cut}kHz 被切断，位置不常见。", f"The sound is cut at an unusual {cut} kHz."))
        else:
            add(q_pcm, UNKNOWN, tr("证据不足。", "Not enough evidence."))
        hump = metrics["dsd_noise"] and metrics["dsd_noise"]["second_hump"]
        add(tr(f"由低规格 DSD 提升到 {dsd_label(info['dsd_rate_hz'])}",
               f"Raised to {dsd_label(info['dsd_rate_hz'])} from a lower DSD rate"), HIGH if hump else LOW,
            tr("噪声曲线上有两次隆起，像是低规格 DSD 转换后重新生成的。",
               "The noise curve humps twice, as if a lower-rate DSD was converted and re-modulated.") if hump else
            tr("没看到这种提升通常会留下的两段式噪声。", "None of the two-stage noise such a conversion usually leaves."))

    headlines = {
        "lossy_suspect": tr("这首很可能是用 MP3/AAC 等有损文件转成的“假无损”。",
                            "Probably a \"fake lossless\" file made from an MP3, AAC or other lossy file."),
        "no_lossy_sign": tr(f"没发现有损压缩的痕迹，像是正常的 {label} 无损音乐。",
                            f"No sign of lossy compression; looks like normal {label} lossless music."),
        "band_limited_low": tr(f"“假高解析”：内容连 CD 水平都不到，很可能来自有损文件，再被提升成 {label}。",
                               f"\"Fake hi-res\": the content is below CD quality, probably from a lossy file, raised to {label}."),
        "band_limited_nonstandard": tr("高频被限制在一个不常见的位置，来源不明。",
                                       "The treble is limited at an unusual frequency; the source is unclear."),
        "little_ultrasonic_energy": tr(f"虽然是 {label}，但几乎没有 CD/48kHz 之外的高频内容，高采样率意义不大。",
                                       f"Although {label}, there is almost nothing beyond what CD/48 kHz holds; "
                                       "the high rate adds little."),
        "ultrasonic_noise_like": tr("高采样率多出来的部分主要是噪声，难以说明是真高解析。",
                                    "What the high rate adds is mostly noise; it does not show real hi-res."),
        "uncertain": tr("证据混合，无法给出明确结论。", "Mixed evidence; no clear conclusion."),
        "insufficient_signal": tr("有效声音太少或太短，无法判断。", "Too little or too short a sound to judge."),
        "dsd_lossy_origin": tr("这个 DSD 的来源很可能是有损或低规格文件。",
                               "This DSD probably comes from a lossy or low-quality file."),
        "dsd_band_limited_nonstandard": tr("这个 DSD 的高频被限制在一个不常见的位置，来源不明。",
                                           "This DSD's treble is limited at an unusual frequency; the source is unclear."),
        "dsd_remodulated": tr("这个 DSD 像是从更低规格的 DSD 提升上来的。",
                              "This DSD looks raised from a lower DSD rate."),
        "dsd_no_pcm_edge": tr("没发现由 CD 等常见规格转制的痕迹；可能是原生 DSD，也可能来自高规格 PCM。",
                              "No sign of conversion from CD or other common rates; native DSD or hi-res PCM."),
    }
    if code == "upsampled":
        rates = (source["possible_source_rates_hz"] if source and source["possible_source_rates_hz"]
                 else [mirror["source_rate_hz"]])
        head = tr(f"“假高解析”：实际只相当于 {rate_words(rates)} 的水平，被提升成了 {label}。",
                  f"\"Fake hi-res\": really only {rate_words(rates)} quality, raised to {label}.")
        if lossy_hit:
            head += tr("而且来源还可能是有损文件。", " The source may also be a lossy file.")
    elif code == "ultrasonic_content":
        head = (tr("像是真正的高解析录音：含有 CD/48kHz 记录不到的高频。",
                   "Looks like a real hi-res recording: it holds treble CD/48 kHz cannot.")
                if corr is not None and corr >= 0.5 else
                tr("可能是真高解析，但多出来的高频也可能只是噪声。",
                   "May be real hi-res, but the extra treble may also be just noise."))
    elif code == "dsd_pcm_origin":
        rates = source["possible_source_rates_hz"]
        head = tr(f"不是原生 DSD 录音：是由 {rate_words(rates)} 的普通音频"
                  f"{'（多半是 CD 母带）' if rates[0] == 44100 else ''}转成的。",
                  f"Not a native DSD recording: converted from ordinary {rate_words(rates)} audio"
                  f"{' (most likely a CD master)' if rates[0] == 44100 else ''}.")
    else:
        head = headlines.get(code, verdict_["summary"])
    if chain and not dsd:
        if code in CHAIN_OVERRIDES:
            limit = chain["source_max_hz"]
            head = (tr(f"超声部分可能不是原始录音：和已知加工链「{chain['label']}」的指纹吻合，像是加工时加上的。",
                       f"The ultrasonic band may not be from the original recording: it matches known processing "
                       f"chain \"{chain['label']}\" and looks added in processing.")
                    + (tr(f"原始音源大概不高于 {khz(limit)}。", f" The source is probably {khz(limit)} or lower.")
                       if limit else ""))
        else:
            head += tr(f"另外，超声部分和已知加工链「{chain['label']}」的指纹吻合。",
                       f" The ultrasonic band also matches known processing chain \"{chain['label']}\".")
    return {"headline": head, "items": items,
            "note": tr("“可能性”是按证据强弱分的等级，不是统计出来的概率。",
                       "Likelihoods are grades of evidence strength, not measured probabilities.")}


def chain_plain(match: dict) -> str:
    tones = [f"{t['hz'] / 1000:g}kHz" for t in match["tones"] if t["present"]]
    parts = []
    if tones:
        parts.append(tr(f"在 {'、'.join(tones)} 有和这条链一样的固定尖峰（转换设备的杂音，音乐里不会有）",
                        f"the same fixed tones as this chain at {', '.join(tones)} (converter noise, never part of music)"))
    r = match["shape_correlation"]
    if r is not None and r >= match["shape_strong"]:
        parts.append(tr("超声部分的细节形状也和这条链的其他文件对得上",
                        "the fine shape of the ultrasonic band also matches this chain's other files"))
    return tr("，", ", and ").join(parts)


def format_plain(plain: dict) -> list[str]:
    lines = [tr("结论", "Conclusion"), "-" * 54, plain["headline"], ""]
    for item in plain["items"]:
        answer = (tr(f"可能性{item['level_text']}", f"likelihood {item['level_text']}")
                  if item["level"] != UNKNOWN else item["level_text"])
        lines.append(f"  {item['question']}{tr('：', ': ')}{answer}")
        lines.append(f"      {item['why']}")
    lines += ["", f"  {tr('（', '(')}{plain['note']}{tr('）', ')')}", ""]
    return lines


def format_report(report: dict) -> str:
    i, m, r, v = (report[k] for k in ["file", "spectral_metrics", "roundtrip_48khz", "verdict"])
    bits = f"{i['bits_per_sample']} bit" if i["bits_per_sample"] else i["sample_format"]
    dsd = i["dsd_rate_hz"] is not None
    colon = tr("：", ": ")
    if dsd:
        layout = tr(f"{dsd_label(i['dsd_rate_hz'])}（{i['dsd_rate_hz'] / 1e6:g} MHz）/ {i['channels']} 声道；"
                    f"由 FFmpeg 解码为 {i['sample_rate_hz']} Hz PCM，分析至 {i['analysis_nyquist_hz'] / 1000:g} kHz",
                    f"{dsd_label(i['dsd_rate_hz'])} ({i['dsd_rate_hz'] / 1e6:g} MHz) / {i['channels']} channels; "
                    f"decoded by FFmpeg to {i['sample_rate_hz']} Hz PCM, analysed up to {i['analysis_nyquist_hz'] / 1000:g} kHz")
    else:
        layout = tr(f"{i['sample_rate_hz']} Hz / {i['channels']} 声道 / {bits}",
                    f"{i['sample_rate_hz']} Hz / {i['channels']} channels / {bits}")
    lines = [
        tr("DSD 来源检查", "DSD source check") if dsd else tr("PCM 采样率 / 频宽检查", "PCM sample rate / bandwidth check"),
        "=" * 54,
        tr("文件：", "File: ") + i["path"],
        tr(f"格式：{format_label(i)}（{i['codec']}）", f"Format: {format_label(i)} ({i['codec']})"), "",
        *format_plain(report["plain"]),
        tr("技术细节", "Technical details"), "=" * 54,
        tr(f"格式：{i['codec']}（{i['container']}）/ {layout}", f"Format: {i['codec']} ({i['container']}) / {layout}"),
        tr(f"文件时长：{i['duration_seconds']:.2f} 秒；扫描：{i['analyzed_seconds']:.2f} 秒",
           f"Duration: {i['duration_seconds']:.2f} s; scanned: {i['analyzed_seconds']:.2f} s"),
        tr("扫描范围：", "Scan range: ") + (tr("全曲", "whole file") if i["full_file_scanned"]
                                         else tr("仅文件开头，后续未分析", "start of the file only; the rest was not analysed")),
        tr(f"频率分辨率：{i['frequency_resolution_hz']:.2f} Hz；有效窗：{i['active_window_count']}/{i['window_count']}；"
           f"静音门限：{i['rms_gate_dbfs']:.0f} dBFS",
           f"Frequency resolution: {i['frequency_resolution_hz']:.2f} Hz; active windows: "
           f"{i['active_window_count']}/{i['window_count']}; silence gate: {i['rms_gate_dbfs']:.0f} dBFS"),
    ]
    if i["decoder_warnings"]:
        lines.append(tr("解码警告：", "Decoder warning: ") + i["decoder_warnings"].splitlines()[0])
    lines += ["", tr(f"技术判断（{v['code']}）：", f"Technical verdict ({v['code']}): ") + v["summary"], v["limitation"], ""]
    rel = tr("相对 20Hz–20kHz", "relative to 20 Hz-20 kHz")
    if dsd:
        lines.append(tr("DSD 的超声区主要是调制噪声，超声能量、平坦度等指标不说明内容，见下方噪声分析。",
                        "DSD's ultrasonic band is mostly modulator noise; ultrasonic energy and flatness say nothing "
                        "about content here. See the noise analysis below."))
    else:
        lines += [
            tr(">24kHz 能量：", "Energy >24 kHz: ") + f"{m['above_24khz_rms_dbfs']:.1f} dBFS{tr('；', '; ')}"
            f"{rel}{colon}{number(m['above_24khz_relative_db'])} dB",
            tr(">26kHz 能量（避开转换过渡带）：", "Energy >26 kHz (clear of the conversion transition band): ")
            + f"{number(m['above_26khz_relative_db'])} dB{tr('，', ', ')}{rel}",
            tr(">24.5kHz 能量（自动判断的截止边缘保护区）：", "Energy >24.5 kHz (guard band for the automatic edge check): ")
            + f"{number(m['above_24_5khz_relative_db'])} dB{tr('，', ', ')}{rel}",
            tr("每声道 >24kHz 相对能量：", "Per-channel relative energy >24 kHz: ")
            + ", ".join(number(x) + " dB" for x in m["above_24khz_channel_relative_db"]),
            tr("超声谱平坦度：", "Ultrasonic spectral flatness: ") + number(m["ultrasonic_spectral_flatness"], 3)
            + tr("（接近 1 较平坦；不能单独证明是噪声）", " (near 1 is flat; this alone does not prove noise)"),
            tr("超声/可听频段电平相关性：", "Ultrasonic/audible level correlation: ")
            + number(m["ultrasonic_audible_envelope_correlation"], 3),
        ]
    lines += ["", tr("频段                    RMS dBFS      相对 20Hz–20kHz dB",
                     "Band                    RMS dBFS      rel. 20 Hz-20 kHz dB")]
    for b in m["bands"]:
        label = f"{b['low_hz']/1000:g}–{b['high_hz']/1000:g} kHz"
        lines.append(f"{label:<24}{b['rms_dbfs']:>9.1f}      {number(b['relative_to_20hz_20khz_db']):>10}")
    lines += ["", tr("频宽截止（全曲平均谱中陡峭、持续的下降，含量化/抖动底噪的台阶）：",
                     "Band limits (steep, sustained drops in the whole-file average spectrum, including "
                     "quantisation/dither floor steps):")]
    if m["band_edges"]:
        lines += [edge_line(e, i["dsd_rate_hz"] or i["sample_rate_hz"]) for e in m["band_edges"]]
    else:
        lines.append(tr("  未发现陡峭且持续的截止；这不代表一定是原生高采样率。",
                        "  No steep, sustained cut found; that does not prove a native high rate."))
    for mirror in m["imaging"]:
        half, src = khz(mirror["source_rate_hz"] / 2), khz(mirror["source_rate_hz"])
        lines.append(tr(f"  镜像：{half} 两侧频谱对称（相关 {mirror['correlation']:.2f}，"
                        f"镜像约低 {mirror['attenuation_db']:.0f} dB），提示由 {src} 升频。",
                        f"  Images: the spectrum is symmetric around {half} (correlation {mirror['correlation']:.2f}, "
                        f"images about {mirror['attenuation_db']:.0f} dB lower), pointing to upsampling from {src}."))
    found = m["lossy"]
    if found:
        lines += ["", tr(f"有损编码检查（{found['frame_seconds'] * 1000:.0f}ms 短窗，{found['frames_analyzed']} 个响亮片段）：",
                         f"Lossy-codec check ({found['frame_seconds'] * 1000:.0f} ms windows, "
                         f"{found['frames_analyzed']} loud frames):")]
        at, drop = found["cliff_hz"] / 1000, found["cliff_drop_db"]
        if found["cliff_detected"]:
            lines.append(tr(f"  平均谱约 {at:.2f}kHz 处陡降 {drop:.0f}dB，接近 {codec_hint(found['cliff_hz'])} 的低通位置。",
                            f"  The average spectrum drops {drop:.0f} dB at about {at:.2f} kHz, near the low-pass "
                            f"of {codec_hint(found['cliff_hz'])}."))
        else:
            lines.append(tr(f"  14–{LOSSY_MAX_CUTOFF_HZ / 1000:g}kHz 内未见 ≥{LOSSY_STEP_DB}dB 的陡降"
                            f"（最陡处约 {at:.2f}kHz，{drop:.0f}dB）。",
                            f"  No drop of >= {LOSSY_STEP_DB} dB within 14-{LOSSY_MAX_CUTOFF_HZ / 1000:g} kHz "
                            f"(steepest: about {at:.2f} kHz, {drop:.0f} dB)."))
        p95, share = found["frame_cutoff_p95_hz"] / 1000, found["loud_frames_above_20khz_percent"]
        lines.append(tr(f"  响亮片段的内容上限：95% 分位约 {p95:.1f}kHz；{share:.0f}% 的片段延伸到 20kHz 以上。",
                        f"  Upper limit of loud frames: 95th percentile about {p95:.1f} kHz; {share:.0f}% reach above 20 kHz."))
        if found["holes_percent"] is None:
            holes = tr("频段太窄，未统计。", "band too narrow, not counted.")
        else:
            holes = tr(f"{found['holes_percent']:.2f}%（≥{LOSSY_HOLES_PERCENT:g}% 视为有损迹象；校准用的 CD 原盘都低于 0.1%）；"
                       f"另有 {found['full_band_gap_frames_percent']:.1f}% 的片段整段高频停顿，视为音乐本身，不计入。",
                       f"{found['holes_percent']:.2f}% (>= {LOSSY_HOLES_PERCENT:g}% points to lossy; every CD rip used "
                       f"for calibration was below 0.1%); another {found['full_band_gap_frames_percent']:.1f}% of frames "
                       "have the whole treble pause, taken as the music itself and not counted.")
        lines.append(tr("  16kHz 以上频谱空洞：", "  Spectral holes above 16 kHz: ") + holes)
    check = m.get("chain_check")
    if check is not None:
        lines += ["", tr("加工链检查（--check-sr）：", "Processing-chain check (--check-sr):")]
        if not check["performed"]:
            lines.append("  " + check["reason"])
        for c in check["results"]:
            tones = tr("，", ", ").join(
                tr(f"{t['hz'] / 1000:g}kHz 尖峰 {number(t['prominence_db'])}dB（需 ≥{t['minimum_db']:g}）",
                   f"{t['hz'] / 1000:g} kHz tone {number(t['prominence_db'])} dB (needs >= {t['minimum_db']:g})")
                for t in c["tones"]) or tr("无尖峰特征", "no tones")
            state = (tr(f"吻合（可能性{level_text(c['level'])}）", f"match (likelihood {level_text(c['level'])})")
                     if c["matched"] else
                     tr("部分特征吻合，不足以下结论", "partial match, not enough for a conclusion") if c["partial"]
                     else tr("不吻合", "no match"))
            alone = (tr(f"≥{c['shape_only']:g} 可单独成立", f">= {c['shape_only']:g} matches on its own")
                     if c["shape_only"] is not None else
                     tr("这条链不能只凭形状判定", "this chain cannot match on shape alone"))
            lines.append(tr(f"  「{c['label']}」：{state}；{tones}；超声细节形状相关 {number(c['shape_correlation'], 2)}"
                            f"（≥{c['shape_strong']:g} 加强尖峰证据；{alone}）",
                            f"  \"{c['label']}\": {state}; {tones}; fine ultrasonic shape correlation "
                            f"{number(c['shape_correlation'], 2)} (>= {c['shape_strong']:g} strengthens the tones; {alone})"))
    noise = m["dsd_noise"]
    if noise:
        lines += [
            "", tr("DSD 调制噪声（1/6 倍频程中位数）：", "DSD modulator noise (1/6-octave medians):"),
            tr(f"  最低点约 {noise['floor_hz'] / 1000:.0f} kHz，{noise['floor_db_per_hz']:.0f} dBFS/Hz；"
               f"约 {noise['observable_limit_hz'] / 1000:.0f} kHz 起高出 {DSD_OBSERVABLE_RISE_DB} dB，"
               "更高频的 PCM 截止会被噪声淹没，无法检出。",
               f"  Lowest at about {noise['floor_hz'] / 1000:.0f} kHz, {noise['floor_db_per_hz']:.0f} dBFS/Hz; "
               f"{DSD_OBSERVABLE_RISE_DB} dB higher from about {noise['observable_limit_hz'] / 1000:.0f} kHz, so a PCM "
               "cut above that is buried in noise and cannot be seen."),
        ]
        hump = noise["second_hump"]
        lines.append(
            tr(f"  约 {hump['peak_hz'] / 1000:.0f} kHz 处有第二个噪声隆起（回落 {hump['depth_db']:.0f} dB 后再上升），提示两级调制。",
               f"  A second noise hump at about {hump['peak_hz'] / 1000:.0f} kHz (falls {hump['depth_db']:.0f} dB, "
               "then rises again) points to two modulators.")
            if hump else tr("  噪声随频率单调上升或持平，未见两级调制的隆起。",
                            "  The noise rises steadily or stays flat; no two-modulator hump."))
    lines += ["", tr("原采样率 → 48kHz → 原采样率差分：", "Round trip: original rate -> 48 kHz -> original rate:")]
    if r["performed"]:
        lines += [
            tr(f"  均匀选取 {len(r['segments'])} 个片段，共测量 {r['measured_seconds']:.2f} 秒。",
               f"  {len(r['segments'])} evenly spread excerpts, {r['measured_seconds']:.2f} s measured."),
            tr("  转换滤波器通带至 20kHz、阻带从 24kHz 起；保留电平，补偿群延迟，并排除片段边缘。",
               "  Filter passband to 20 kHz, stopband from 24 kHz; level kept, group delay compensated, excerpt edges excluded."),
            tr(f"  全频段残差：{number(r['residual_relative_db'])} dB（相对原信号），{r['residual_rms_dbfs']:.1f} dBFS。",
               f"  Full-band residual: {number(r['residual_relative_db'])} dB (relative to the original), "
               f"{r['residual_rms_dbfs']:.1f} dBFS."),
            tr(f"  20Hz–20kHz 残差：{number(r['audible_residual_relative_db'])} dB（相对同频段原信号），"
               f"{r['audible_residual_rms_dbfs']:.1f} dBFS。",
               f"  20 Hz-20 kHz residual: {number(r['audible_residual_relative_db'])} dB (relative to the original "
               f"in that band), {r['audible_residual_rms_dbfs']:.1f} dBFS."),
            tr("  例如 -60 dB 的残差能量约为原信号的百万分之一；差分也包含滤波器误差。",
               "  A -60 dB residual holds about a millionth of the original's energy; it includes filter error."),
            tr("  差分只覆盖这些片段；20–24kHz 过渡带的损失也会计入全频段残差。",
               "  Only these excerpts are covered; losses in the 20-24 kHz transition band count in the full-band residual."),
        ]
    else:
        lines.append("  " + r["reason"])
    lines += ["", tr("解释与边界：", "Interpretation and limits:")] + ["  • " + note for note in report["notes"]]
    return "\n".join(lines) + "\n"


def save_plot(path: Path, info: dict, data: dict, metrics: dict) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    freq = data["freq"] / 1000
    top_khz = info["analysis_nyquist_hz"] / 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), constrained_layout=True)
    top, middle, bottom = axes
    for channel, psd in enumerate(data["psd_channels"]):
        top.plot(freq, 10 * np.log10(np.maximum(psd, 1e-20)), lw=0.6, alpha=0.45,
                 label=f"Channel {channel + 1}")
    top.plot(freq, 10 * np.log10(np.maximum(data["psd"], 1e-20)), color="#136f63", lw=1.2,
             label="Mean channel power (active frames)")
    for cutoff, color in [(20, "#9b6500"), (24, "#cf4138")]:
        if cutoff < top_khz:
            top.axvline(cutoff, color=color, ls="--", label=f"{cutoff} kHz")
    for n, edge in enumerate(e for e in metrics["band_edges"] if e["band_limit"]):
        top.axvline(edge["frequency_hz"] / 1000, color="#5b3cc4", ls=":", lw=1.2,
                    label="Detected band limit" if n == 0 else None)
    top.set(xlim=(0, top_khz), ylim=(-190, 0), xlabel="Frequency (kHz)",
            ylabel="PSD (dBFS/Hz)",
            title=("DSD source check - decoded by FFmpeg" if info["dsd_rate_hz"] else "PCM bandwidth check")
            + " - signal evidence, not an audibility test")
    top.grid(alpha=0.2)
    top.legend(loc="upper right", fontsize=8)
    heat = 10 * np.log10(np.maximum(data["spectrogram"].T, 1e-20))
    im = middle.imshow(heat, origin="lower", aspect="auto", cmap="magma", vmin=-160, vmax=-50,
                       extent=(0, data["time_extent"], 0, freq[-1]))
    middle.set_ylim(0, top_khz)
    if top_khz > 24:
        middle.axhline(24, color="cyan", ls="--", lw=0.8)
    middle.set(xlabel="Time (s)", ylabel="Frequency (kHz)", title="Power spectrogram (all frames)")
    fig.colorbar(im, ax=middle, label="dBFS/Hz", pad=0.01)
    for label, key, color in [("20 Hz - 20 kHz", "time_audible", "#136f63"),
                              (">24 kHz", "time_ultra", "#cf4138")]:
        values = 10 * np.log10(np.maximum(data[key], 1e-20))
        values[~data["time_valid"]] = np.nan
        bottom.plot(data["time"], values, label=label, color=color, lw=1)
    bottom.set(xlim=(0, data["time_extent"]), ylim=(-160, 5),
               xlabel="Time (s)", ylabel="Band RMS (dBFS)",
               title=f"Band levels - >24 kHz relative energy: {number(metrics['above_24khz_relative_db'])} dB")
    bottom.legend(fontsize=9)
    bottom.grid(alpha=0.2)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def positive(value: str) -> float:
    number_ = float(value)
    if not math.isfinite(number_) or number_ <= 0:
        raise argparse.ArgumentTypeError(tr("请输入大于 0 的有限数值。", "enter a finite number above 0"))
    return number_


def nonnegative(value: str) -> float:
    number_ = float(value)
    if not math.isfinite(number_) or number_ < 0:
        raise argparse.ArgumentTypeError(tr("请输入不小于 0 的有限数值。", "enter a finite number of 0 or more"))
    return number_


REPORT_FILES = ("report.txt", "report.json", "spectrum.csv", "spectrum.png")
# Extensions picked up when scanning folders; single files are probed whatever their name.
AUDIO_EXTENSIONS = {
    ".flac", ".wav", ".wave", ".w64", ".rf64", ".aif", ".aiff", ".aifc", ".caf", ".m4a", ".alac",
    ".mka", ".ape", ".wv", ".tta", ".tak", ".dsf", ".dff",
}
# Lossy codecs are skipped in folder scans: the checks target lossless and hi-res files.
LOSSY_CODECS = {
    "mp1", "mp2", "mp3", "aac", "aac_latm", "opus", "vorbis", "wmav1", "wmav2", "wmapro", "wmavoice",
    "ac3", "eac3", "dts", "atrac3", "atrac3p", "cook", "speex", "amr_nb", "amr_wb", "musepack7", "musepack8",
}


def build_report(info: dict, metrics: dict, roundtrip: dict) -> dict:
    report = {
        "schema_version": VERSION, "file": info, "spectral_metrics": metrics,
        "roundtrip_48khz": roundtrip, "verdict": verdict(info, metrics),
        "heuristics": {
            "edge_bin_hz": EDGE_BIN_HZ, "edge_search_from_hz": EDGE_SEARCH_FROM_HZ,
            "edge_minimum_drop_db": EDGE_MIN_DROP_DB,
            "edge_brickwall_minimum_drop_and_margin_db": EDGE_BRICKWALL_DROP_DB,
            "edge_brickwall_maximum_core_transition_hz": EDGE_BRICKWALL_CORE_HZ,
            "edge_brickwall_maximum_floor_std_db": EDGE_BRICKWALL_FLOOR_STD_DB, "edge_maximum_transition_hz": EDGE_MAX_WIDTH_HZ,
            "edge_minimum_slope_db_per_octave": EDGE_MIN_SLOPE_DB_PER_OCTAVE,
            "edge_minimum_pre_edge_density_db_per_hz": EDGE_MIN_LEVEL_DB,
            "edge_empty_margin_db": EDGE_EMPTY_MARGIN_DB,
            "edge_empty_margin_with_imaging_db": EDGE_IMAGED_MARGIN_DB,
            "image_minimum_correlation": IMAGE_MIN_CORRELATION,
            "image_standalone_minimum_correlation": IMAGE_STANDALONE_CORRELATION,
            "image_minimum_passband_shape_db": IMAGE_MIN_SHAPE_DB,
            "lossy_edge_below_hz": LOSSY_EDGE_BELOW_HZ,
            "lossy_max_cutoff_hz": LOSSY_MAX_CUTOFF_HZ, "lossy_search_from_hz": LOSSY_SEARCH_FROM_HZ,
            "lossy_minimum_cliff_db": LOSSY_STEP_DB, "lossy_hole_depth_db": LOSSY_HOLE_DB,
            "lossy_minimum_holes_percent": LOSSY_HOLES_PERCENT, "lossy_frame_seconds": LOSSY_FRAME_SECONDS,
            "lossy_holes_note_percent": LOSSY_HOLES_NOTE_PERCENT, "lossy_gap_frame_share": LOSSY_GAP_SHARE,
            "dsd_analysis_share_of_decoded_nyquist": DSD_ANALYSIS_SHARE,
            "dsd_observable_rise_db": DSD_OBSERVABLE_RISE_DB, "dsd_second_hump_db": DSD_HUMP_DB,
            "source_nyquist_ratio_range": list(SOURCE_NYQUIST_RATIO),
            "source_typical_ratio": SOURCE_TYPICAL_RATIO, "edge_standard_cliff_db": EDGE_STANDARD_CLIFF_DB,
            "standard_rates_hz": list(STANDARD_RATES),
            "negligible_ultrasonic_relative_db": -75, "negligible_ultrasonic_rms_dbfs": -95,
            "clear_ultrasonic_relative_db": -65, "clear_ultrasonic_rms_dbfs": -95,
            "clear_ultrasonic_edge_guard_hz": 24500,
            "noise_like_minimum_flatness": 0.5, "noise_like_maximum_absolute_correlation": 0.25,
        },
        "notes": [tr(zh, en) for zh, en in [
            ("48kHz 的奈奎斯特频率为 24kHz；常见可听频率上限约 20kHz，且因人而异。",
             "48 kHz has a 24 kHz Nyquist frequency; the usual upper limit of hearing is about 20 kHz and varies by person."),
            (">24kHz 能量可能来自乐器、话筒噪声、噪声整形、DSD 转 PCM 或后期非线性处理。能量多不等于音质好。",
             "Energy above 24 kHz can come from instruments, microphone noise, noise shaping, DSD-to-PCM conversion or "
             "non-linear processing. More energy does not mean better sound."),
            ("截止检测在全曲平均谱上连续扫描两种尺度：截止前后中位数相差 ≥25dB、斜率 ≥120dB/倍频程（模拟滚降通常只有 12–48），"
             "且截止以上的 90 分位电平比截止前低 ≥20dB，才算频宽上限。量化/抖动底噪的台阶同样计入，所以高频很弱的录音也能检出。",
             "Cuts are searched in the whole-file average spectrum at two scales: a band limit needs >= 25 dB between the "
             "medians on either side, a slope of >= 120 dB/octave (analogue roll-offs are 12-48), and a 90th-percentile level "
             "above the cut >= 20 dB below the level before it. Quantisation/dither floor steps count too, so recordings "
             "with very weak treble are still caught."),
            ("截止以上若是与通带对称的镜像（升频滤波不足），或先空出一段再出现上升的噪声（噪声整形，如经 DSD 转换），也视为频宽上限。"
             "线性插值等廉价升频几乎不衰减镜像、没有陡降，另由镜像对称性单独检测。",
             "A cut still counts as a band limit when what lies above mirrors the passband (a weak upsampling filter), or "
             "when a gap is followed by rising noise (noise shaping, e.g. via DSD). Cheap upsampling such as linear "
             "interpolation barely attenuates images and has no steep drop; it is caught separately by image symmetry."),
            ("来源采样率按截止起点或中点推断：须落在该采样率奈奎斯特频率的 80%–103%。滤波器较缓时 44.1k 与 48k 无法区分，报告会同时列出。",
             "The source rate is inferred from the start or midpoint of the cut, which must lie within 80-103% of that "
             "rate's Nyquist frequency. With gentle filters 44.1k and 48k cannot be told apart, and both are listed."),
            ("多级升频只有在中间步骤留下痕迹（镜像、噪声、第二个截止）时才能看出；转换干净时，与一次直接升频在数学上无法区分。",
             "Multi-stage upsampling shows only when a middle step leaves traces (images, noise, a second cut); clean "
             "conversions are mathematically identical to one direct upsampling."),
            ("截止也可能来自录音设备、母带处理或主动低通，不能证明有人伪造采样率。",
             "A cut can also come from recording equipment, mastering or a deliberate low-pass; it does not prove "
             "anyone faked the sample rate."),
            ("升采样后加入噪声/失真能重新产生超声能量，所以本脚本也无法证明没有升采样。",
             "Noise or distortion added after upsampling can recreate ultrasonic energy, so this tool cannot prove "
             "that no upsampling happened."),
            ("判定门限仅为可复查的经验规则，不是统计置信度或听觉阈值。极弱高频、窄带音乐、短片段可能误判。",
             "Thresholds are checkable rules of thumb, not statistical confidence or hearing thresholds. Very weak "
             "treble, narrow-band music and short excerpts can be misjudged."),
            ("频谱按声道功率平均，避免反相声道抵消；剔除极静窗。PSD 去除每窗直流；FFT 泄漏和量化噪声仍会影响结果。",
             "Channels are averaged by power so out-of-phase channels cannot cancel; near-silent windows are dropped. "
             "Each window's DC is removed; FFT leakage and quantisation noise still affect the result."),
            ("dBFS 以样本幅度 1 为参考：满幅正弦的 RMS 为 -3.01 dBFS；dB 相对值表示能量比。",
             "dBFS is relative to a sample value of 1: a full-scale sine has an RMS of -3.01 dBFS; relative dB are energy ratios."),
            ("滤波后可听频段残差很小，支持这次转换对该频段影响很小；不等于已经完成听觉测试。",
             "A small audible-band residual after filtering supports that the conversion barely changes that band; "
             "it is not a listening test."),
            ("需要比较听感时，应使用同一母带、电平匹配且转换质量足够的版本，做随机双盲 ABX。",
             "To compare how they sound, use the same master, level-matched, with good conversion, in a randomised "
             "blind ABX test."),
        ]],
    }
    if (metrics.get("chain_check") or {}).get("performed"):
        report["notes"] += [tr(zh, en) for zh, en in [
            ("加工链检查（--check-sr）比对两类指纹：一是对准采样率分数的窄带尖峰（如 192kHz 的 48k/72kHz，转换设备的固有杂音），"
             f"二是 {CHAIN_BAND_LOW_HZ / 1000:g}kHz 以上频谱的细节形状。真实录音之间这两者互不相关；同一条加工链做出来的文件则一致。",
             "The chain check (--check-sr) compares two fingerprints: narrow tones at simple fractions of the sample rate "
             "(e.g. 48k/72 kHz at 192 kHz, a converter's own noise), and the fine shape of the spectrum above "
             f"{CHAIN_BAND_LOW_HZ / 1000:g} kHz. Unrelated recordings do not share them; files from one chain do."),
            ("尖峰必须全部出现才算吻合，单独一根尖峰在原生录音里也见过（2L 的 192k 录音曾出现 7.9dB 的 48kHz 尖峰）。",
             "All of a chain's tones must be present: a single tone has been seen in native recordings (a 2L 192k "
             "recording showed a 7.9 dB tone at 48 kHz)."),
            ("吻合只说明和学习用的那批文件经过同一种加工，说不出加工方式（回录、AI 或其他）；同一套设备做的真录音也可能被误认。",
             "A match only says the file went through the same processing as the files the chain was learned from; it "
             "cannot say how (re-recording, AI or anything else), and real recordings made on the same equipment may match too."),
        ]]
        report["heuristics"].update({
            "chain_band_low_hz": CHAIN_BAND_LOW_HZ, "chain_pool_bins": CHAIN_POOL_BINS,
            "chain_smooth_bins": CHAIN_SMOOTH_BINS, "chain_tone_half_width_hz": CHAIN_TONE_HALF_HZ,
            "chain_tone_side_hz": list(CHAIN_TONE_SIDE_HZ), "chain_silence_dbfs": CHAIN_SILENCE_DBFS,
        })
    if not info["dsd_rate_hz"] and info["sample_rate_hz"] <= REFERENCE_RATE:
        report["notes"] = [tr(zh, en) for zh, en in [
            ("44.1/48kHz 文件只做有损检测：有损编码器会在 16–20.4kHz 加陡峭低通，心理声学模型还会让 16kHz 以上的频段时有时无，形成频谱空洞。",
             "44.1/48 kHz files only get the lossy check: lossy encoders apply a steep low-pass at 16-20.4 kHz, and "
             "their psychoacoustic model makes bands above 16 kHz come and go, leaving spectral holes."),
            (f"判为疑似有损的条件：平均谱在 {LOSSY_SEARCH_FROM_HZ / 1000:g}–{LOSSY_MAX_CUTOFF_HZ / 1000:g}kHz 内有 ≥{LOSSY_STEP_DB}dB 的陡降，"
             f"或 16kHz 以上响亮片段中 ≥{LOSSY_HOLES_PERCENT:g}% 的时频点比该频点中位数低 {LOSSY_HOLE_DB}dB 以上（整段高频停顿的片段不计入）。"
             "门限用 CD 原盘与其 MP3/AAC/Opus/Vorbis 转码校准，属经验规则。",
             f"Suspected lossy when the average spectrum drops >= {LOSSY_STEP_DB} dB within "
             f"{LOSSY_SEARCH_FROM_HZ / 1000:g}-{LOSSY_MAX_CUTOFF_HZ / 1000:g} kHz, or when >= {LOSSY_HOLES_PERCENT:g}% of "
             f"time-frequency cells above 16 kHz in loud frames are {LOSSY_HOLE_DB} dB or more below that bin's median "
             "(frames where the whole treble pauses are left out). Thresholds were calibrated on CD rips and their "
             "MP3/AAC/Opus/Vorbis transcodes; they are rules of thumb."),
            ("CD 播放链路的抗混叠滤波器通常在 20.75kHz 以上，所以 20.5kHz 以上的截止不算有损迹象；这也意味着截止高于 20.5kHz、空洞又少的高码率编码检不出来。",
             "CD anti-alias filters usually sit at 20.75 kHz or higher, so a cut above 20.5 kHz is not taken as lossy; "
             "it also means high-rate encodes that cut above 20.5 kHz with few holes cannot be caught."),
            ("实测检不出：FFmpeg AAC 256/320kbps；容易检出：MP3（含 320kbps 与 VBR V0）、128kbps 级别的 AAC、Opus、Vorbis。",
             "Not caught in tests: FFmpeg AAC 256/320 kbps. Easily caught: MP3 (including 320 kbps and VBR V0), "
             "AAC around 128 kbps, Opus, Vorbis."),
            ("母带主动低通、老录音、降噪处理也会产生截止；“疑似有损”需要结合来源判断，“未见有损迹象”也不能证明一定是无损母带。",
             "Mastering low-passes, old recordings and noise reduction also make cuts; weigh \"suspected lossy\" against "
             "the source, and \"no lossy sign\" does not prove a lossless master."),
            ("dBFS 以样本幅度 1 为参考；频谱按声道功率平均，并剔除极静窗。",
             "dBFS is relative to a sample value of 1; channels are averaged by power and near-silent windows dropped."),
        ]]
    if info["dsd_rate_hz"]:
        report["notes"] = [tr(zh, en) for zh, en in [
            ("DSD 由 sigma-delta 调制产生，噪声整形把量化噪声推到超声区；无论来源是不是 PCM，都会有这段上升的噪声，它本身不能证明真假。",
             "DSD comes from sigma-delta modulation, whose noise shaping pushes quantisation noise into the ultrasonic "
             "band; every DSD has this rising noise whatever its source, so it proves nothing by itself."),
            ("PCM 来源只能在调制噪声抬起之前看到：DSD64 通常只能分辨 44.1/48kHz 及部分 88.2/96kHz 来源，DSD128/256 能看得更高；DXD（352.8kHz）来源一般无法检出。",
             "A PCM source can only be seen below where the modulator noise rises: DSD64 usually shows 44.1/48 kHz and "
             "some 88.2/96 kHz sources, DSD128/256 higher; DXD (352.8 kHz) sources generally cannot be seen."),
            ("许多正规 SACD 也在 PCM/DXD 中剪辑或做母带，“PCM 来源”不等于伪造；但 22.05kHz 附近的硬截止通常说明来自 CD 规格的母带。",
             "Many legitimate SACDs are edited or mastered in PCM/DXD, so \"PCM source\" is not forgery; but a hard cut "
             "near 22.05 kHz usually means a CD-resolution master."),
            ("两级噪声隆起提示 DSD 升频（低倍率 DSD 经 PCM 中转后重新调制）。不同调制器的噪声曲线差异很大，这一项只是线索，不是定论。",
             "Two noise humps suggest DSD upsampling (a lower-rate DSD re-modulated via PCM). Modulator noise curves "
             "vary a lot, so this is a clue, not proof."),
            ("只分析 FFmpeg 解码结果中可信的频段（解码后奈奎斯特频率的一半，DSD64 为 88.2kHz）；截止门限与 PCM 检测相同，均为经验规则。",
             "Only the trustworthy part of FFmpeg's decode is analysed (half the decoded Nyquist frequency, 88.2 kHz for "
             "DSD64); the cut thresholds are the same rules of thumb as for PCM."),
            ("dBFS 以样本幅度 1 为参考；DSD 的 0dB 参考电平按 FFmpeg 解码结果计。",
             "dBFS is relative to a sample value of 1; DSD's 0 dB reference follows FFmpeg's decoder."),
        ]]
    return report


def check_file(path: Path, out: Path, max_seconds: float | None, compare_seconds: float,
               no_plot: bool, progress=None, check_sr: bool = False) -> tuple[dict, str]:
    """Analyse one file and write its four report files into `out`."""
    step = progress or (lambda message: None)
    input_path = path.resolve()
    # Never overwrite the input, even if it happens to use a report filename.
    if any((out / name).resolve() == input_path for name in REPORT_FILES):
        raise ValueError(tr("输出文件会覆盖输入音频，请更换 --out 目录。",
                            "A report file would overwrite the input audio; choose another --out folder."))
    step(tr("[1/3] 流式扫描频谱…", "[1/3] Scanning the spectrum..."))
    info, data = analyze_spectrum(path, max_seconds, check_sr=check_sr)
    metrics = measure_bands(info, data)
    if check_sr:
        metrics["chain_check"] = chain_check(info, data)
    step(tr("[2/3] 测量 48kHz 转换差分…", "[2/3] Measuring the 48 kHz round trip..."))
    roundtrip = compare_roundtrip(path, info, compare_seconds)
    report = build_report(info, metrics, roundtrip)
    report["language"] = LANG
    report["plain"] = plain_summary(info, metrics, report["verdict"])
    out.mkdir(parents=True, exist_ok=True)
    step(tr("[3/3] 写入报告…", "[3/3] Writing the report..."))
    plotted = False if no_plot else save_plot(out / "spectrum.png", info, data, metrics)
    report["plot_generated"] = plotted
    report["plot_status"] = "created" if plotted else ("disabled" if no_plot else "matplotlib_unavailable")
    text = format_report(report)
    (out / "report.txt").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (out / "spectrum.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frequency_hz", "mean_psd_dbfs_per_hz"] +
                        [f"channel_{c + 1}_psd_dbfs_per_hz" for c in range(info["channels"])])
        for index, frequency in enumerate(data["freq"]):
            writer.writerow([f"{frequency:.6f}", f"{db(data['psd'][index]):.6f}"] +
                            [f"{db(p[index]):.6f}" for p in data["psd_channels"]])
    return report, text


def find_audio(paths: list[Path], skip: Path) -> list[tuple[Path, Path]]:
    """(file, path relative to the report root) for every input; folders are walked recursively.

    Hidden entries and macOS "._" resource files are ignored, and symlinked
    folders are not followed, so a link back up the tree cannot loop.
    """
    found = []
    for given in paths:
        if given.is_file():
            found.append((given, Path(given.name)))
            continue
        root = given.resolve()
        # With several inputs, each folder keeps its own name under the report root.
        prefix = Path(root.name) if len(paths) > 1 else Path()
        for folder, dirs, files in os.walk(root):
            here = Path(folder)
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and (here / d).resolve() != skip)
            for name in sorted(files):
                if name.startswith(".") or Path(name).suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                found.append((here / name, prefix / (here / name).relative_to(root)))
    return found


def report_dirs(found: list[tuple[Path, Path]], root: Path) -> list[Path]:
    """One report folder per file, mirroring the input tree; same-stem files keep their extension."""
    stems: dict[Path, int] = {}
    for _, rel in found:
        stems[rel.with_suffix("")] = stems.get(rel.with_suffix(""), 0) + 1
    dirs, used = [], set()
    for _, rel in found:
        target = rel.with_suffix("")
        if stems[target] > 1:
            target = rel.with_name(f"{rel.stem}_{rel.suffix.lstrip('.').lower()}")
        candidate, n = target, 2
        while candidate in used:
            candidate, n = target.with_name(f"{target.name} ({n})"), n + 1
        used.add(candidate)
        dirs.append(root / candidate)
    return dirs


def source_rate(report: dict) -> int | None:
    """The lowest source rate the verdict relies on: a band limit or resampler images."""
    metrics = report["spectral_metrics"]
    source = next((e for e in metrics["band_edges"] if e["role"] == "source_limit"), None)
    rates = [source["possible_source_rates_hz"][0]] if source and source["possible_source_rates_hz"] else []
    rates += [m["source_rate_hz"] for m in metrics["imaging"]]
    return min(rates) if rates else None


VERDICT_TAGS = {
    "lossy_suspect": ("疑似有损转无损", "probably from lossy"), "no_lossy_sign": ("未见有损痕迹", "no lossy sign"),
    "upsampled": ("假高解析（升频）", "fake hi-res (upsampled)"),
    "band_limited_low": ("假高解析（来源频宽很低）", "fake hi-res (low-bandwidth source)"),
    "band_limited_nonstandard": ("来源不明", "unclear source"),
    "little_ultrasonic_energy": ("高解析意义不大", "hi-res adds little"),
    "ultrasonic_content": ("像真高解析", "looks like real hi-res"),
    "ultrasonic_noise_like": ("高频像噪声", "treble looks like noise"),
    "uncertain": ("无法判断", "cannot tell"), "insufficient_signal": ("无法判断", "cannot tell"),
    "dsd_pcm_origin": ("PCM 转制的 DSD", "DSD made from PCM"),
    "dsd_lossy_origin": ("低规格来源转制的 DSD", "DSD from a low-quality source"),
    "dsd_remodulated": ("DSD 升频", "upsampled DSD"), "dsd_no_pcm_edge": ("未见转制痕迹", "no conversion sign"),
    "dsd_band_limited_nonstandard": ("来源不明", "unclear source"),
}


def row_tag(row: dict) -> str:
    if row["status"] != "ok":
        return {"skipped": tr("已跳过", "skipped"), "error": tr("出错", "error")}[row["status"]]
    if row.get("chain") and row["verdict"] in CHAIN_OVERRIDES:
        return tr(f"疑似加工链：{row['chain']}", f"processing chain: {row['chain']}")
    return tr(*VERDICT_TAGS[row["verdict"]]) if row["verdict"] in VERDICT_TAGS else row["verdict"]


# Verdicts whose text names a source rate (the summary table repeats it).
SOURCE_VERDICTS = {"upsampled", "dsd_pcm_origin"}


def batch_one(path: str, rel: str, out: str, max_seconds: float | None,
              compare_seconds: float, no_plot: bool, check_sr: bool = False, lang: str | None = None) -> dict:
    """Summary row for one file of a folder scan; runs in a worker process with -j."""
    if lang:
        set_lang(lang)  # worker processes started by "spawn" (macOS, Windows) do not inherit it
    row = {"file": rel, "status": "ok", "verdict": "", "conclusion": "", "grades": "", "summary": "",
           "format": "", "sample_rate_hz": "", "source_rate_hz": "", "chain": "", "report_dir": out, "error": ""}
    try:
        meta = probe(Path(path))
        row["format"] = f"{meta['codec']} {format_label(meta)}"
        row["sample_rate_hz"] = meta["dsd_rate_hz"] or meta["sample_rate_hz"]
        if meta["codec"] in LOSSY_CODECS:
            row.update(status="skipped", report_dir="", conclusion=tr(
                f"本身就是有损格式（{meta['codec']}），已跳过。", f"Already a lossy format ({meta['codec']}); skipped."))
            return row
        report, _ = check_file(Path(path), Path(out), max_seconds, compare_seconds, no_plot, check_sr=check_sr)
        best = (report["spectral_metrics"].get("chain_check") or {}).get("best")
        row["chain"] = best["label"] if best else ""
        rate = source_rate(report)
        plain = report["plain"]
        row.update(verdict=report["verdict"]["code"], summary=report["verdict"]["summary"],
                   conclusion=plain["headline"],
                   grades=sep().join(f"{i['question']}{tr('：', ': ')}{i['level_text']}" for i in plain["items"]),
                   source_rate_hz=rate if rate and report["verdict"]["code"] in SOURCE_VERDICTS else "")
    except (OSError, RuntimeError, ValueError) as exc:
        row.update(status="error", error=str(exc), report_dir="")
    except Exception as exc:  # One unexpected failure must not end a library scan.
        row.update(status="error", error=tr("内部错误 ", "Internal error ") + f"{type(exc).__name__}: {exc}",
                   report_dir="")
    return row



def write_summary(root: Path, rows: list[dict]) -> None:
    fields = ["file", "status", "conclusion", "grades", "verdict", "source_rate_hz", "chain", "format",
              "sample_rate_hz", "summary", "report_dir", "error"]
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:  # BOM so Excel reads UTF-8
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row_tag(row)] = counts.get(row_tag(row), 0) + 1
    lines = [tr(f"hirescheck {VERSION} 批量检查：共 {len(rows)} 个文件", f"hirescheck {VERSION} batch check: {len(rows)} files"), "",
             tr("（“可能性”是按证据强弱分的等级，不是统计出来的概率；每首的技术细节见各自的 report.txt。）",
                "(Likelihoods are grades of evidence strength, not measured probabilities; technical details are in "
                "each file's report.txt.)"), ""]
    lines += [f"  {key:<30}{count:>5}" for key, count in sorted(counts.items(), key=lambda kv: -kv[1])]
    lines.append("")
    for row in rows:
        lines.append(f"[{row_tag(row)}] {row['file']}" + (tr(f"（{row['format']}）", f" ({row['format']})")
                                                         if row["format"] else ""))
        lines.append(f"    {row['conclusion'] or row['error']}")
        if row["grades"]:
            lines.append(f"    {row['grades']}")
    (root / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(paths: list[Path], root: Path, args: argparse.Namespace) -> int:
    found = find_audio(paths, root.resolve())
    if not found:
        print(tr("错误：没有找到可检查的音频文件。", "Error: no audio files to check were found."), file=sys.stderr)
        return 1
    outs = report_dirs(found, root)
    root.mkdir(parents=True, exist_ok=True)
    print(tr(f"找到 {len(found)} 个音频文件；报告写入 {root.resolve()}",
             f"Found {len(found)} audio files; reports go to {root.resolve()}"), file=sys.stderr, flush=True)
    jobs = [(str(f), str(rel), str(out), args.max_seconds, args.compare_seconds, args.no_plot, args.check_sr, LANG)
            for (f, rel), out in zip(found, outs)]
    rows: list[dict | None] = [None] * len(jobs)
    done = 0

    def show(index: int, row: dict) -> None:
        nonlocal done
        done += 1
        rows[index] = row
        fmt = tr(f"（{row['format']}）", f" ({row['format']})") if row["format"] else ""
        print(f"[{done}/{len(jobs)}] {row['file']}{fmt}{tr('：', ': ')}{row['conclusion'] or row['error']}", flush=True)

    if args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(batch_one, *job): i for i, job in enumerate(jobs)}
            for future in as_completed(futures):
                show(futures[future], future.result())
    else:
        for i, job in enumerate(jobs):
            show(i, batch_one(*job))
    write_summary(root, rows)
    errors = sum(row["status"] == "error" for row in rows)
    print(tr(f"完成：{len(rows) - errors} 个已处理，{errors} 个出错；汇总见 {(root / 'summary.txt').resolve()}",
             f"Done: {len(rows) - errors} processed, {errors} failed; summary in {(root / 'summary.txt').resolve()}"))
    return 1 if errors else 0


def chain_spectrum_job(path: str, max_seconds: float | None) -> tuple:
    try:
        meta, spectrum = chain_spectrum(Path(path), max_seconds)
        return meta, spectrum, ""
    except (OSError, RuntimeError, ValueError) as exc:
        return None, None, str(exc)


def tone_curve(freq: np.ndarray, psd: np.ndarray, fs: int) -> np.ndarray:
    """Prominence of every bin over a 2.4 kHz running median, for finding learned tones."""
    df = freq[1] - freq[0]
    level = 10 * np.log10(np.maximum(psd, 1e-30))
    floor = ndimage.median_filter(level, size=int(2400 / df) | 1, mode="nearest")
    peak = ndimage.maximum_filter1d(level, size=int(2 * CHAIN_TONE_HALF_HZ / df) | 1, mode="nearest")
    curve = peak - floor
    curve[(freq < 20000) | (freq > 0.49 * fs)] = -np.inf
    return curve


def chain_slug(name: str) -> str:
    return re.sub(r"[^\w.-]+", "_", name).strip("._") or "chain"


def learn_chain(name: str, paths: list[Path], out: Path | None, note: str | None,
                max_seconds: float | None, jobs: int) -> int:
    """Build a chain fingerprint from files known to share one production path."""
    builtin = {c["name"] for c in load_chains() if c["origin"] == "builtin"}
    if name in builtin and out is None:
        print(tr(f"错误：「{name}」是内置加工链的名字，请换一个。",
                 f"Error: \"{name}\" is the name of a built-in chain; choose another."), file=sys.stderr)
        return 1
    files = [f for f, _ in find_audio(paths, Path("/"))]
    if len(files) < CHAIN_MIN_FILES:
        print(tr(f"错误：学习加工链至少需要 {CHAIN_MIN_FILES} 个文件（来自至少 {CHAIN_MIN_FOLDERS} 个不同文件夹），"
                 f"现在只有 {len(files)} 个。",
                 f"Error: learning a chain needs at least {CHAIN_MIN_FILES} files (from at least {CHAIN_MIN_FOLDERS} "
                 f"different folders); got {len(files)}."), file=sys.stderr)
        return 1
    print(tr(f"读取 {len(files)} 个文件的长窗频谱…", f"Reading long-window spectra of {len(files)} files..."),
          file=sys.stderr, flush=True)
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs, initializer=set_lang, initargs=(LANG,)) as pool:
            results = list(pool.map(chain_spectrum_job, [str(f) for f in files], [max_seconds] * len(files)))
    else:
        results = [chain_spectrum_job(str(f), max_seconds) for f in files]
    problems, rates = [], {}
    for f, (meta, spectrum, error) in zip(files, results):
        if error:
            problems.append(f"{f}：{error}")
        elif meta["dsd_rate_hz"] or meta["codec"] in LOSSY_CODECS:
            problems.append(tr(f"{f}：只能用 PCM 无损文件学习（这是 {meta['codec']}）",
                               f"{f}: only lossless PCM files can be used (this is {meta['codec']})"))
        elif spectrum is None:
            problems.append(tr(f"{f}：有效声音太少", f"{f}: too little signal"))
        else:
            rates.setdefault(meta["sample_rate_hz"], []).append((f, spectrum))
    if problems:
        print(tr("错误：以下文件不能用来学习：", "Error: these files cannot be used for learning:")
              + "\n  " + "\n  ".join(problems), file=sys.stderr)
        return 1
    if len(rates) > 1:
        detail = tr("，", ", ").join(tr(f"{khz(r)} {len(v)} 个", f"{len(v)} at {khz(r)}") for r, v in sorted(rates.items()))
        print(tr(f"错误：一条加工链只对应一种采样率，这批文件混有 {detail}。请分开学习。",
                 f"Error: a chain covers one sample rate, but these files mix {detail}. Learn them separately."),
              file=sys.stderr)
        return 1
    fs, members = next(iter(rates.items()))
    if fs < CHAIN_MIN_RATE:
        print(tr(f"错误：{khz(fs)} 没有超声频段，无法提取加工链指纹（至少 {khz(CHAIN_MIN_RATE)}）。",
                 f"Error: {khz(fs)} has no ultrasonic band to fingerprint (needs {khz(CHAIN_MIN_RATE)} or more)."),
              file=sys.stderr)
        return 1
    folders = {f.resolve().parent for f, _ in members}
    if len(folders) < CHAIN_MIN_FOLDERS:
        print(tr(f"错误：文件只来自 {len(folders)} 个文件夹。至少需要 {CHAIN_MIN_FOLDERS} 个不同文件夹（专辑），"
                 "否则学到的可能只是某张专辑自己的母带特征。",
                 f"Error: the files come from only {len(folders)} folder(s). At least {CHAIN_MIN_FOLDERS} different "
                 "folders (albums) are needed, or the fingerprint may just be one album's mastering."), file=sys.stderr)
        return 1

    # Tones: bins that stand out in every file.
    curves = np.array([tone_curve(fr, ps, fs) for _, (fr, ps) in members])
    freq = members[0][1][0]
    weakest = curves.min(axis=0)
    weakest[~np.isfinite(weakest)] = -100
    peaks, props = signal.find_peaks(weakest, height=CHAIN_LEARN_TONE_DB,
                                     distance=max(1, int(CHAIN_TONE_NOTCH_HZ / (freq[1] - freq[0]))))
    # Refine each candidate to the peak of the files' mean level (the max filter smears it).
    mean_level = np.mean([10 * np.log10(np.maximum(ps, 1e-30)) for _, (_, ps) in members], axis=0)
    picked = []
    for i in peaks[np.argsort(-props["peak_heights"])][:CHAIN_MAX_TONES]:
        near = np.nonzero(np.abs(freq - freq[i]) <= 2 * CHAIN_TONE_HALF_HZ)[0]
        picked.append(float(round(freq[near[np.argmax(mean_level[near])]])))
    tones = []
    for hz in sorted(picked):
        values = [tone_prominence(fr, 10 * np.log10(np.maximum(ps, 1e-30)), hz) for _, (fr, ps) in members]
        if all(v is not None for v in values) and min(values) >= CHAIN_LEARN_TONE_DB:
            low, high = CHAIN_TONE_DB_RANGE
            tones.append({"hz": hz, "min_db": round(float(min(max(min(values) - CHAIN_TONE_MARGIN_DB, low), high)), 1),
                          "learned_min_db": round(min(values), 1), "learned_values_db": [round(v, 1) for v in values]})
    notch = [t["hz"] for t in tones]
    shapes = np.array([chain_shape(fr, ps, fs, notch) for _, (fr, ps) in members])
    if shapes.shape[1] == 0:
        print(tr("错误：超声频段太窄，无法提取形状指纹。", "Error: the ultrasonic band is too narrow for a shape fingerprint."),
              file=sys.stderr)
        return 1

    def template_of(rows: np.ndarray) -> np.ndarray:
        t = rows.mean(axis=0)
        keep = t != 0
        out_ = np.zeros(len(t))
        out_[keep] = (t[keep] - t[keep].mean()) / max(t[keep].std(), 1e-9)
        return out_

    loo = []
    for i in range(len(shapes)):
        t = template_of(np.delete(shapes, i, axis=0))
        loo.append(float(shapes[i] @ t / len(t)))
    print(tr(f"\n{len(members)} 个文件，{len(folders)} 个文件夹，{khz(fs)}：",
             f"\n{len(members)} files, {len(folders)} folders, {khz(fs)}:"))
    for (f, _), r, i in zip(members, loo, range(len(members))):
        tone_text = ", ".join(f"{t['hz'] / 1000:g}k {t['learned_values_db'][i]:.1f}dB" for t in tones)
        print(f"  {f.name[:40]:<40} {tr('留一法形状相关', 'leave-one-out shape r')} {r:5.2f}  {tone_text}")
    shape_only_ok = min(loo) >= CHAIN_SHAPE_ONLY
    sys.stdout.flush()
    if len(tones) < 2 and not shape_only_ok:
        print(tr(f"\n错误：没找到这批文件的共同指纹：共同尖峰 {len(tones)} 根（至少要 2 根），"
                 f"留一法形状相关最低 {min(loo):.2f}（只靠形状需要每个文件都 ≥{CHAIN_SHAPE_ONLY:g}）。"
                 "它们可能并非出自同一条加工链。",
                 f"\nError: these files share no fingerprint: {len(tones)} common tone(s) (at least 2 needed), "
                 f"lowest leave-one-out shape correlation {min(loo):.2f} (shape alone needs every file >= "
                 f"{CHAIN_SHAPE_ONLY:g}). They may not come from one processing chain."), file=sys.stderr)
        return 1
    chain = {
        "format": CHAIN_FORMAT, "name": name, "label": name, "sample_rate_hz": fs,
        "note": note or f"和你用 --learn-chain 学习的那 {len(members)} 个文件经过同一种加工",
        "note_en": None if note else f"processed the same way as the {len(members)} files you learned it from",
        "source_max_hz": None,
        "tones": [{k: t[k] for k in ("hz", "min_db", "learned_min_db")} for t in tones],
        "shape_strong": CHAIN_SHAPE_STRONG, "shape_only": CHAIN_SHAPE_ONLY if shape_only_ok else None,
        "learned": {"files": len(members), "folders": len(folders), "program_version": VERSION,
                    "loo_shape_correlation_min": round(min(loo), 3),
                    "loo_shape_correlation_median": round(float(np.median(loo)), 3)},
        "shape_template": [round(float(v), 3) for v in template_of(shapes)],
    }
    target = out or CHAIN_USER_DIR / f"{chain_slug(name)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    existed = target.exists()
    target.write_text(json.dumps(chain, ensure_ascii=False) + "\n", encoding="utf-8")
    found_tones = tr("、", ", ").join(tr(f"{t['hz'] / 1000:g}kHz（匹配门槛 {t['min_db']:g}dB）",
                                         f"{t['hz'] / 1000:g} kHz (match threshold {t['min_db']:g} dB)")
                                      for t in tones) or tr("无", "none")
    print(tr("\n共同尖峰：", "\nCommon tones: ") + found_tones)
    if np.median(loo) < CHAIN_SHAPE_STRONG:
        print(tr(f"注意：形状相关的中位数只有 {np.median(loo):.2f}，形状指纹较弱，匹配主要靠尖峰。",
                 f"Note: the median shape correlation is only {np.median(loo):.2f}; the shape fingerprint is weak "
                 "and matching relies mostly on the tones."))
    print(tr(f"{'已覆盖' if existed else '已保存'}加工链指纹：{target}",
             f"Chain fingerprint {'overwritten' if existed else 'saved'}: {target}"))
    print(tr("以后加 --check-sr 检查时会自动比对。建议先用几首确定不是这条链的文件试一下，确认不会误报。",
             "It will be compared whenever you check with --check-sr. Try a few files you know are not from this "
             "chain first, to make sure it does not give false matches."))
    return 0


def list_chains() -> int:
    chains = load_chains()
    if not chains:
        print(tr("没有可用的加工链指纹。", "No chain fingerprints available."))
        return 0
    for c in chains:
        tones = tr("、", ", ").join(f"{t['hz'] / 1000:g}kHz" for t in c["tones"]) or tr("无", "none")
        origin = tr("内置", "built-in") if c["origin"] == "builtin" else tr("自己学习", "learned")
        label = chain_field(c, "label") or c["name"]
        print(tr(f"「{label}」（{origin}）：{khz(c['sample_rate_hz'])}，尖峰 {tones}，由 {c['learned']['files']} 个文件学习",
                 f"\"{label}\" ({origin}): {khz(c['sample_rate_hz'])}, tones {tones}, learned from "
                 f"{c['learned']['files']} files") + f"\n    {c['file']}")
    return 0


def pick_lang(argv: list[str]) -> str | None:
    """--lang value from the raw arguments, read before the parser so --help is translated too."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    lang = pick_lang(argv)
    if lang in ("zh", "en"):
        set_lang(lang)
    parser = argparse.ArgumentParser(
        prog="hirescheck", formatter_class=argparse.RawDescriptionHelpFormatter,
        description=tr("检查 PCM / DSD 文件的真实规格：44.1/48k 是否由有损文件转来，高采样率是否升频而来、"
                       "原始采样率大概是多少，DSD 是否由 PCM 转制。结果是证据强弱的分级，不是鉴定结论。",
                       "Check what a PCM or DSD file really holds: whether 44.1/48k came from a lossy file, whether a "
                       "high-rate file was upsampled and from what rate, and whether a DSD was made from PCM. Results "
                       "are graded evidence, not proof."))
    parser.add_argument("paths", nargs="*", type=Path, metavar=tr("路径", "PATH"),
                        help=tr("待检查的音频文件或文件夹（FLAC、WAV、AIFF、ALAC 等 PCM，或 DSF/DFF）；文件夹会递归扫描",
                                "audio files or folders to check (FLAC, WAV, AIFF, ALAC and other PCM, or DSF/DFF); "
                                "folders are scanned recursively"))
    parser.add_argument("--out", type=Path,
                        help=tr("报告目录；单个文件默认为 ./<文件名>_hirescheck，文件夹默认为 ./<文件夹名>_hirescheck",
                                "report folder; defaults to ./<file name>_hirescheck for a file and "
                                "./<folder name>_hirescheck for a folder"))
    parser.add_argument("--max-seconds", type=positive,
                        help=tr("仅扫描开头 N 秒；默认扫描全曲", "scan only the first N seconds; default: the whole file"))
    parser.add_argument("--compare-seconds", type=nonnegative, default=24,
                        help=tr("差分选段总时长，默认 24 秒；0 为关闭",
                                "total length of the round-trip excerpts, default 24 s; 0 turns it off"))
    parser.add_argument("--no-plot", action="store_true",
                        help=tr("不生成 PNG，也不需要 matplotlib", "skip the PNG; matplotlib is then not needed"))
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help=tr("批量检查时同时分析的文件数，默认 1", "files analysed in parallel in a batch, default 1"))
    parser.add_argument("--lang", choices=("zh", "en"),
                        help=tr("输出语言：zh 中文，en 英文；默认跟随系统语言",
                                "output language: zh Chinese, en English; default follows the system language"))
    chain = parser.add_argument_group(
        tr("加工链检查（检查超声部分是不是由某条已知的加工链做出来的，只对 88.2kHz 及以上的 PCM 有效）",
           "processing-chain check (whether the ultrasonic band was made by a known processing chain; "
           "PCM at 88.2 kHz and above only)"))
    chain.add_argument("--check-sr", action="store_true",
                       help=tr("检查时比对内置和自己学习的加工链指纹；不加则不做这项检查",
                               "compare against the built-in and learned chain fingerprints; off by default"))
    chain.add_argument("--learn-chain", metavar=tr("名字", "NAME"),
                       help=tr(f"用给出的文件学习一条新的加工链指纹，不做常规检查。至少 {CHAIN_MIN_FILES} 个文件，"
                               f"来自至少 {CHAIN_MIN_FOLDERS} 个不同文件夹（专辑），采样率必须相同（≥{khz(CHAIN_MIN_RATE)}）；"
                               f"建议 10 个以上。默认保存到 {CHAIN_USER_DIR}，可用 --out 指定文件",
                               f"learn a new chain fingerprint from the given files instead of checking them. Needs at "
                               f"least {CHAIN_MIN_FILES} files from at least {CHAIN_MIN_FOLDERS} different folders "
                               f"(albums), all at one sample rate (>= {khz(CHAIN_MIN_RATE)}); 10 or more recommended. "
                               f"Saved to {CHAIN_USER_DIR} unless --out names a file"))
    chain.add_argument("--chain-note", metavar=tr("说明", "TEXT"),
                       help=tr("配合 --learn-chain：匹配时显示的说明文字", "with --learn-chain: text shown on a match"))
    chain.add_argument("--list-chains", action="store_true",
                       help=tr("列出可用的加工链指纹", "list the available chain fingerprints"))
    parser.add_argument("--version", action="version", version=f"hirescheck {VERSION}")
    args = parser.parse_args(argv)
    if args.list_chains:
        return list_chains()
    if not args.paths:
        parser.error(tr("请给出要检查的文件或文件夹。", "give the files or folders to check"))
    for path in args.paths:
        if not path.exists():
            parser.error(tr(f"找不到文件或文件夹：{path}", f"no such file or folder: {path}"))
    if args.jobs < 1:
        parser.error(tr("-j 必须是正整数。", "-j must be a positive integer"))
    if args.learn_chain is not None:
        if not args.learn_chain.strip():
            parser.error(tr("--learn-chain 需要一个名字。", "--learn-chain needs a name"))
        return learn_chain(args.learn_chain.strip(), args.paths, args.out, args.chain_note,
                           args.max_seconds, args.jobs)

    if len(args.paths) == 1 and args.paths[0].is_file():
        path = args.paths[0]
        out = args.out or Path.cwd() / (path.stem + "_hirescheck")
        try:
            report, text = check_file(path, out, args.max_seconds, args.compare_seconds, args.no_plot,
                                      progress=lambda message: print(message, file=sys.stderr, flush=True),
                                      check_sr=args.check_sr)
        except (OSError, RuntimeError, ValueError) as exc:
            print(tr("错误：", "Error: ") + str(exc), file=sys.stderr)
            return 1
        print(text)
        print(tr("报告目录：", "Report folder: ") + str(out.resolve()))
        if not args.no_plot and not report["plot_generated"]:
            print(tr("未安装 matplotlib，已跳过 PNG；其余报告已生成。",
                     "matplotlib is not installed, so no PNG; the rest of the report was written."), file=sys.stderr)
        return 0

    if args.out:
        root = args.out
    elif len(args.paths) == 1:
        root = Path.cwd() / (args.paths[0].resolve().name + "_hirescheck")
    else:
        root = Path.cwd() / "hirescheck_reports"
    try:
        return run_batch(args.paths, root, args)
    except KeyboardInterrupt:
        print(tr("\n已中断；已完成的报告保留在各自目录中，未生成汇总。",
                 "\nInterrupted; finished reports are kept in their folders, but no summary was written."), file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
