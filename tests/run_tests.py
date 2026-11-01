#!/usr/bin/env python3
"""Run hirescheck over the corpus from make_corpus.py and compare verdicts.

Usage: python run_tests.py CORPUS_DIR [--real-dsd DIR] [-v]

An expected.json in CORPUS_DIR (written by derive_from_source.py) adds its
cases; an expected code "NOT:a,b" passes for any verdict other than a or b.
--real-dsd DIR adds every .dsf/.dff in DIR as a known CD-sourced (44.1 kHz) DSD;
point it only at albums verified to be CD-sourced.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Run the package from this checkout; -I keeps the current directory and PYTHON* variables out.
COMMAND = [sys.executable, "-I", "-c",
           f"import sys; sys.path.insert(0, {str(ROOT)!r}); from hirescheck import main; sys.exit(main())"]
# file: (expected verdict code, expected source rate or None, expected intermediate rate or None)
# KNOWN = physically hard to see; reported but not counted as a failure.
CASES = {
    "up44_to_192.flac": ("upsampled", 44100, None),
    "up48_to_192.flac": ("upsampled", 48000, None),
    "up44_to_88.flac": ("upsampled", 44100, None),
    "up44_to_96_linear.flac": ("upsampled", 44100, None),
    "up44_to_192_lax.flac": ("upsampled", 44100, None),
    "dark44_to_192_lax.flac": ("KNOWN", None, None),
    "chain44_96linear_192.flac": ("upsampled", 44100, 96000),
    "dark44_to_192.flac": ("upsampled", 44100, None),
    "dark48_to_192.flac": ("upsampled", 48000, None),
    "up96_to_192.flac": ("upsampled", 96000, None),
    "native192_analog.flac": ("ultrasonic_content", None, None),
    "native192_dark.flac": ("ultrasonic_content", None, None),
    "native96_analog.wav": ("ultrasonic_content", None, None),
    "lp16k_up192.flac": ("band_limited_low", None, None),
    "lp19k_up192.flac": ("upsampled", 44100, None),
    "plain44_16.wav": ("no_lossy_sign", None, None),
    "mp3_320_44k16.flac": ("lossy_suspect", None, None),
    "mp3_128_44k16.flac": ("lossy_suspect", None, None),
    "aac_128_44k16.flac": ("lossy_suspect", None, None),
    "native_dsd64.dsf": ("dsd_no_pcm_edge", None, None),
    "native_dsd128.dsf": ("dsd_no_pcm_edge", None, None),
    "pcm44k_dsd64.dsf": ("dsd_pcm_origin", 44100, None),
    "pcm44k_dsd128.dsf": ("dsd_pcm_origin", 44100, None),
    "pcm48k_dsd64.dsf": ("dsd_pcm_origin", 48000, None),
    "pcm48k_dsd128.dsf": ("dsd_pcm_origin", 48000, None),
    "pcm96k_dsd64.dsf": ("KNOWN", None, None),
    "pcm96k_dsd128.dsf": ("dsd_pcm_origin", 96000, None),
    "pcm192k_dsd64.dsf": ("dsd_no_pcm_edge", None, None),
    "pcm192k_dsd128.dsf": ("KNOWN", None, None),
    "lossy16k_dsd64.dsf": ("dsd_lossy_origin", None, None),
    "lossy16k_dsd128.dsf": ("dsd_lossy_origin", None, None),
    "remod64to128_lp100k.dsf": ("dsd_remodulated", None, None),
    "remod64to128_dxd.dsf": ("KNOWN", None, None),
}


def source_rates(report: dict) -> tuple[int | None, int | None]:
    """The lowest source rate the report implies, and the first intermediate stage above it."""
    metrics = report["spectral_metrics"]
    edges = metrics["band_edges"]
    source = next((e for e in edges if e["role"] == "source_limit"), None)
    rates = [source["possible_source_rates_hz"][0]] if source and source["possible_source_rates_hz"] else []
    rates += [m["source_rate_hz"] for m in metrics["imaging"]]
    origin = min(rates) if rates else None
    later = [e["possible_source_rates_hz"][0] for e in edges
             if e["role"] in ("source_limit", "intermediate_limit") and e["possible_source_rates_hz"]
             and origin and min(e["possible_source_rates_hz"]) > origin]
    return origin, later[0] if later else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--real-dsd", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    cases = {args.corpus / name: expected for name, expected in CASES.items() if (args.corpus / name).exists()}
    listed = args.corpus / "expected.json"
    if listed.exists():
        cases.update({args.corpus / name: tuple(v) for name, v in json.loads(listed.read_text()).items()})
    if args.real_dsd:
        for path in sorted(args.real_dsd.iterdir()):
            if path.suffix.lower() in (".dsf", ".dff"):
                cases[path] = ("dsd_pcm_origin", 44100, None)
    missing = [name for name in CASES if not (args.corpus / name).exists()]
    if missing and not listed.exists():
        print(f"跳过 {len(missing)} 个不存在的样本：{', '.join(missing)}")
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path, (code, rate, stage) in cases.items():
            out = Path(tmp) / path.name
            run = subprocess.run([*COMMAND, str(path), "--out", str(out),
                                  "--compare-seconds", "0", "--no-plot"], capture_output=True, text=True)
            if run.returncode:
                failures += 1
                print("ERROR", path.name, run.stderr.strip()[-300:])
                continue
            report = json.loads((out / "report.json").read_text(encoding="utf-8"))
            got_code = report["verdict"]["code"]
            got_rate, got_stage = source_rates(report)
            code_ok = (got_code not in code[4:].split(",")) if code.startswith("NOT:") else got_code == code
            ok = code_ok and rate in (None, got_rate) and stage in (None, got_stage)
            known = code == "KNOWN"
            failures += not (ok or known)
            print("KNOWN" if known else "PASS " if ok else "FAIL ", path.name[:28].ljust(29),
                  got_code.ljust(26), report["verdict"]["summary"])
            if args.verbose or not (ok or known):
                for e in report["spectral_metrics"]["band_edges"]:
                    print(f"        {e['frequency_hz']:8.0f} Hz  drop {e['drop_db']:4.0f} dB  "
                          f"width {e['transition_width_hz']:5.0f}  role {e['role']}  rates {e['possible_source_rates_hz']}")
    print(f"{len(cases) - failures}/{len(cases)} 通过（KNOWN 为已知局限，不计失败）")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
