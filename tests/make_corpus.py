#!/usr/bin/env python3
"""Build the labelled synthetic corpus used by run_tests.py.

PCM files are written through FFmpeg; DSD files need a C compiler for the
1-bit sigma-delta modulator in sdm.c. Usage: python make_corpus.py OUT_DIR
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from math import gcd
from pathlib import Path

import numpy as np
from scipy import optimize, signal

HERE = Path(__file__).resolve().parent
SEC = 20
DSD64 = 2822400


def music(rng, fs, top, order=4, seconds=SEC):
    """Pink-ish noise with a music-like envelope, rolled off by a Butterworth at `top`."""
    n = int(fs * seconds)
    w = rng.standard_normal(n)
    b, a = signal.butter(1, 1500, fs=fs)
    x = signal.lfilter(b, a, w) + 0.03 * w
    x = signal.sosfilt(signal.butter(order, top, fs=fs, output="sos"), x)
    return 0.3 * x / np.max(np.abs(x)) * (0.4 + 0.6 * np.abs(np.sin(np.arange(n) / fs * 1.3)))


def dither(rng, x, bits):
    lsb = 2.0 ** -(bits - 1)
    return np.round(x / lsb + rng.uniform(-.5, .5, x.shape) + rng.uniform(-.5, .5, x.shape)) * lsb


def steep(x, a, b):
    """Steep resampler: passband to 91% of the lower Nyquist, 120 dB stop at that Nyquist."""
    g = gcd(a, b)
    up, down = b // g, a // g
    ny = min(a, b) / 2
    taps, beta = signal.kaiserord(120, 0.09 * ny / (a * up / 2))
    fir = signal.firwin(taps | 1, 0.955 * ny, fs=a * up, window=("kaiser", beta))
    return signal.resample_poly(x, up, down, window=fir)  # resample_poly applies the gain


def lax(x, a, b):
    """scipy's default polyphase filter: a slow skirt around the lower Nyquist."""
    g = gcd(a, b)
    return signal.resample_poly(x, b // g, a // g)


def linear(x, a, b):
    """Cheap linear interpolation: strong images."""
    t = np.arange(int(len(x) * b / a)) * a / b
    return np.interp(t, np.arange(len(x)), x)


def write_pcm(path: Path, x, fs, bits=24):
    stereo = np.stack([x, np.roll(x, 37) * 0.9], axis=1)
    codec = (["-c:a", "flac", "-sample_fmt", "s32" if bits == 24 else "s16"] if path.suffix == ".flac"
             else ["-c:a", f"pcm_s{bits}le"])
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f64le", "-ar", str(fs), "-ac", "2", "-i", "pipe:0",
                    *codec, str(path)], input=np.ascontiguousarray(stereo, "<f8").tobytes(), check=True)


def build_pcm(out: Path) -> None:
    rng = np.random.default_rng(7)
    def save(name, x, fs, bits=24):
        return write_pcm(out / name, dither(rng, x, bits), fs, bits)
    full44 = dither(rng, music(rng, 44100, 20000), 16)
    full48 = dither(rng, music(rng, 48000, 21000), 16)
    save("up44_to_192.flac", steep(full44, 44100, 192000), 192000)
    save("up48_to_192.flac", steep(full48, 48000, 192000), 192000)
    save("up44_to_88.flac", steep(full44, 44100, 88200), 88200)
    save("up44_to_96_linear.flac", linear(full44, 44100, 96000), 96000)
    save("up44_to_192_lax.flac", lax(full44, 44100, 192000), 192000)
    save("dark44_to_192_lax.flac", lax(dither(rng, music(rng, 44100, 6000), 16), 44100, 192000), 192000)
    save("chain44_96linear_192.flac", steep(linear(full44, 44100, 96000), 96000, 192000), 192000)
    save("dark44_to_192.flac", steep(dither(rng, music(rng, 44100, 6000), 16), 44100, 192000), 192000)
    save("dark48_to_192.flac", steep(dither(rng, music(rng, 48000, 6000), 16), 48000, 192000), 192000)
    save("up96_to_192.flac", steep(music(rng, 96000, 40000), 96000, 192000), 192000)
    # Native-like: analog roll-off, ADC noise continuing to Nyquist.
    def adc(fs, dbhz):
        return rng.standard_normal(fs * SEC) * 10 ** (dbhz / 20) * np.sqrt(fs / 2)
    save("native192_analog.flac", music(rng, 192000, 40000, order=2) + adc(192000, -150), 192000)
    save("native192_dark.flac", music(rng, 192000, 8000) + adc(192000, -150), 192000)
    save("native96_analog.wav", music(rng, 96000, 30000, order=2) + adc(96000, -145), 96000)
    for cut in (16000, 19000):  # Lossy-style low-pass, then upsampled.
        lp = signal.sosfiltfilt(signal.ellip(12, 0.1, 100, cut, fs=44100, output="sos"), full44)
        save(f"lp{cut // 1000}k_up192.flac", steep(lp, 44100, 192000), 192000)
    save("plain44_16.wav", full44, 44100, bits=16)
    # Lossy round trips of the 44.1/16 file, decoded back to 16-bit FLAC.
    for name, codec in [("mp3_320", ["-c:a", "libmp3lame", "-b:a", "320k"]),
                        ("mp3_128", ["-c:a", "libmp3lame", "-b:a", "128k"]),
                        ("aac_128", ["-c:a", "aac", "-b:a", "128k"])]:
        lossy = out / f"{name}.{'mp3' if name.startswith('mp3') else 'm4a'}"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "plain44_16.wav"), *codec, str(lossy)], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(lossy), "-c:a", "flac", "-sample_fmt", "s16",
                        str(out / f"{name}_44k16.flac")], check=True)
        lossy.unlink()


def ntf(fs, hinf=1.5):
    """5th-order Butterworth-highpass noise transfer function with |NTF| = hinf at Nyquist."""
    def coeffs(fc):
        _, a = signal.butter(5, fc, "high", fs=fs)
        return np.poly(np.ones(5)), a / a[0]
    def err(fc):
        b, a = coeffs(fc)
        return abs(np.polyval(b, -1) / np.polyval(a, -1)) - hinf
    return coeffs(optimize.brentq(err, 1e3, fs / 2.5))


def modulate(sdm: Path, x, fs):
    b, a = ntf(fs)
    p = subprocess.run([str(sdm)] + [repr(float(v)) for v in (*b, *a)],
                       input=np.ascontiguousarray(x, "<f8").tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8)


def write_dsf(path: Path, packed, fs):
    """Mono DSF, LSB-first, 4096-byte blocks."""
    block = 4096
    data = np.concatenate([packed, np.zeros((-len(packed)) % block, np.uint8)]).tobytes()
    fmt = struct.pack("<4sQIIIIIIQII", b"fmt ", 52, 1, 0, 1, 1, fs, 1, len(packed) * 8, block, 0)
    head = struct.pack("<4sQQQ", b"DSD ", 28, 28 + 52 + 12 + len(data), 0)
    path.write_bytes(head + fmt + struct.pack("<4sQ", b"data", 12 + len(data)) + data)


def read_dsf_bits(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count, = struct.unpack_from("<Q", raw, 28 + 36)
    data = np.frombuffer(raw, np.uint8, offset=28 + 52 + 12)
    return np.unpackbits(data, bitorder="little")[:count].astype(np.float64) * 2 - 1


def build_dsd(out: Path) -> None:
    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not compiler:
        print("未找到 C 编译器，跳过 DSD 样本。")
        return
    sdm = out / "sdm"
    subprocess.run([compiler, "-O3", "-o", str(sdm), str(HERE / "sdm.c")], check=True)
    rng = np.random.default_rng(5)
    base = 352800
    # 'Analog' programme with a gentle 2-pole roll-off at 40 kHz.
    x = signal.sosfilt(signal.butter(2, 40000, fs=base, output="sos"), music(rng, base, 150000, order=1, seconds=8))
    def via(rate):
        return steep(steep(x, base, rate), rate, base)[:len(x)]
    lossy = signal.sosfiltfilt(signal.ellip(12, 0.1, 100, 16000, fs=base, output="sos"), via(44100))
    scale = 0.35 / np.max(np.abs(x))
    variants = {"native": x, "pcm44k": via(44100), "pcm48k": via(48000), "pcm96k": via(96000),
                "pcm192k": via(192000), "lossy16k": lossy}
    for name, y in variants.items():
        for mult, label in [(8, "dsd64"), (16, "dsd128")]:
            up = signal.resample_poly(y * scale, mult, 1, window=("kaiser", 12.0))
            write_dsf(out / f"{name}_{label}.dsf", modulate(sdm, up, base * mult), base * mult)
    # DSD64 -> PCM -> DSD128, through a 100 kHz low-pass and through a wider DXD-like one.
    bits = read_dsf_bits(out / "native_dsd64.dsf")
    for tag, cut in [("lp100k", 100000), ("dxd", 150000)]:
        pcm = signal.sosfiltfilt(signal.butter(8, cut, fs=DSD64, output="sos"), bits)
        write_dsf(out / f"remod64to128_{tag}.dsf", modulate(sdm, signal.resample_poly(pcm, 2, 1), 2 * DSD64), 2 * DSD64)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    build_pcm(target)
    build_dsd(target)
    print(f"样本已写入 {target}")
