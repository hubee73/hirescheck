#!/usr/bin/env python3
"""Derive labelled fakes from a real high-rate source that is known not to be upsampled.

Usage: python derive_from_source.py SOURCE OUT_DIR [--start 60] [--seconds 60]

Writes an excerpt of the source plus versions passed through lower PCM rates,
an MP3, DSD64/DSD128, a 44.1/16 copy with lossy round trips of it, and an
expected.json that run_tests.py picks up.
All conversions use FFmpeg; DSD needs a C compiler for sdm.c.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # Shares the DSD helpers with make_corpus.py.
from make_corpus import DSD64, ntf, write_dsf  # noqa: E402
# swr settings matching a careful mastering resampler; quantize with TPDF dither.
HQ = "filter_size=256:phase_shift=12:cutoff=0.97:kaiser_beta=14:dither_method=triangular"
NOT_UPSAMPLED = "NOT:upsampled,band_limited_low,band_limited_nonstandard"


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args], check=True)


def flac24(src: Path, out: Path, chain: str) -> None:
    ffmpeg("-i", f"file:{src}", "-af", chain, "-c:a", "flac", "-sample_fmt", "s32",
           "-bits_per_raw_sample", "24", f"file:{out}")


def via(rate: int, bits: int) -> str:
    """Resample to `rate` and quantize to `bits`, then back up to 192 kHz / 24 bit."""
    fmt = "s16" if bits == 16 else "s32"
    return (f"aresample={rate}:{HQ}:output_sample_bits={bits},aformat=sample_fmts={fmt},"
            f"aresample=192000:{HQ}:output_sample_bits=24")


def to_dsd(sdm: Path, src: Path, out: Path, rate: int | None, dsd_rate: int) -> None:
    """Optionally pass through PCM `rate`, then FFmpeg upsamples and the modulator runs (left channel)."""
    b, a = ntf(dsd_rate)
    pre = f"aresample={rate}:{HQ}:output_sample_bits=24,aformat=sample_fmts=s32," if rate else ""
    chain = f"pan=mono|c0=c0,{pre}aresample={dsd_rate}:{HQ},volume=0.7"
    decoder = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", f"file:{src}", "-af", chain,
                                "-f", "f64le", "-c:a", "pcm_f64le", "pipe:1"], stdout=subprocess.PIPE)
    modulated = subprocess.run([str(sdm)] + [repr(float(v)) for v in (*b, *a)],
                               stdin=decoder.stdout, capture_output=True, check=True)
    decoder.stdout.close()
    if decoder.wait():
        raise RuntimeError("FFmpeg 转换失败")
    write_dsf(out, np.frombuffer(modulated.stdout, np.uint8), dsd_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--start", type=float, default=60)
    parser.add_argument("--seconds", type=float, default=60)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = args.out / "src_192k.flac"
    ffmpeg("-ss", str(args.start), "-t", str(args.seconds), "-i", f"file:{args.source.resolve()}", "-map", "0:a:0",
           "-af", f"aresample=192000:{HQ}:output_sample_bits=24", "-c:a", "flac", "-sample_fmt", "s32",
           "-bits_per_raw_sample", "24", f"file:{src}")
    expected = {src.name: [NOT_UPSAMPLED, None, None]}
    for rate, bits, name in [(44100, 16, "via44k16_192k.flac"), (48000, 24, "via48k24_192k.flac"),
                             (88200, 24, "via88k24_192k.flac"), (96000, 24, "via96k24_192k.flac")]:
        flac24(src, args.out / name, via(rate, bits))
        expected[name] = ["upsampled", rate, None]
    flac24(src, args.out / "chain44k_96k_192k.flac",
           f"aresample=44100:{HQ}:output_sample_bits=16,aformat=sample_fmts=s16,"
           f"aresample=96000:filter_size=8:cutoff=0.97:output_sample_bits=24,aformat=sample_fmts=s32,"
           f"aresample=192000:{HQ}:output_sample_bits=24")
    expected["chain44k_96k_192k.flac"] = ["upsampled", 44100, None]
    mp3 = args.out / "mp3_320.mp3"
    ffmpeg("-i", f"file:{src}", "-ar", "44100", "-c:a", "libmp3lame", "-b:a", "320k", f"file:{mp3}")
    flac24(mp3, args.out / "mp3_320_192k.flac", f"aresample=192000:{HQ}:output_sample_bits=24")
    expected["mp3_320_192k.flac"] = ["upsampled", 44100, None]
    mp3.unlink()

    # CD-format copies: a clean 44.1/16 version and lossy round trips of it.
    cd = args.out / "cd44k16.flac"
    ffmpeg("-i", f"file:{src}", "-af", f"aresample=44100:{HQ}:output_sample_bits=16", "-c:a", "flac",
           "-sample_fmt", "s16", f"file:{cd}")
    expected[cd.name] = ["no_lossy_sign", None, None]
    for name, ext, codec, want in [
        ("cd_mp3_320", "mp3", ["libmp3lame", "-b:a", "320k"], "lossy_suspect"),
        # V0 low-passes near 21 kHz; with weak treble it leaves few holes and matches the CD filter.
        ("cd_mp3_v0", "mp3", ["libmp3lame", "-q:a", "0"], "KNOWN"),
        ("cd_aac_128", "m4a", ["aac", "-b:a", "128k"], "lossy_suspect"),
        ("cd_opus_160", "opus", ["libopus", "-b:a", "160k"], "lossy_suspect"),
        ("cd_aac_256", "m4a", ["aac", "-b:a", "256k"], "KNOWN"),
    ]:
        lossy = args.out / f"{name}.{ext}"
        ffmpeg("-i", f"file:{cd}", "-c:a", *codec, f"file:{lossy}")
        ffmpeg("-i", f"file:{lossy}", "-ar", "44100", "-c:a", "flac", "-sample_fmt", "s16",
               f"file:{args.out / (name + '.flac')}")
        expected[name + ".flac"] = [want, None, None]
        lossy.unlink()

    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler:
        sdm = args.out / "sdm"
        subprocess.run([compiler, "-O3", "-o", str(sdm), str(HERE / "sdm.c")], check=True)
        for name, rate, dsd_rate, want in [
            ("dsd64_from44k.dsf", 44100, DSD64, ["dsd_pcm_origin", 44100, None]),
            ("dsd64_from192k.dsf", None, DSD64, ["dsd_no_pcm_edge", None, None]),
            ("dsd128_from96k.dsf", 96000, 2 * DSD64, ["KNOWN", None, None]),
        ]:
            to_dsd(sdm, src, args.out / name, rate, dsd_rate)
            expected[name] = want
        sdm.unlink()
    else:
        print("未找到 C 编译器，跳过 DSD 派生样本。")
    (args.out / "expected.json").write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"派生样本已写入 {args.out}")


if __name__ == "__main__":
    main()
