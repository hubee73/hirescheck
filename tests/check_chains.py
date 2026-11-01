#!/usr/bin/env python3
"""Regression check for --check-sr: known chain members must match, other files must not.

Usage: python check_chains.py --chain NAME --match LIST [--no-match LIST] [-v]

LIST is a text file with one audio path per line (blank lines and # comments
are ignored). Files under --match must match chain NAME (e.g. chain-192k-a);
files under --no-match (native hi-res such as the 2L test bench, plainly
upsampled files) must not match any chain. The audio itself is not shipped
with the repository; keep the lists next to your own library.
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


def read_list(path: str | None) -> list[str]:
    if not path:
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]


def check(path: str, work: Path) -> dict:
    out = work / str(abs(hash(path)))
    subprocess.run([*COMMAND, path, "--check-sr", "--no-plot",
                    "--compare-seconds", "0", "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.loads((out / "report.json").read_text(encoding="utf-8"))["spectral_metrics"]["chain_check"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--match", required=True)
    parser.add_argument("--no-match")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    cases = [(p, True) for p in read_list(args.match)] + [(p, False) for p in read_list(args.no_match)]
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path, expect in cases:
            result = check(path, Path(tmp))
            best = result.get("best")
            mine = next((r for r in result["results"] if r["name"] == args.chain), None)
            ok = (best is not None and best["name"] == args.chain) if expect else best is None
            failures += not ok
            if mine:
                tones = " ".join(f"{t['hz'] / 1000:g}k={t['prominence_db']:.1f}" for t in mine["tones"])
                detail = f"{tones} r={mine['shape_correlation']:.2f} level={mine['level'] or ('partial' if mine['partial'] else '-')}"
            else:
                detail = result["reason"]
            if args.verbose or not ok:
                print(f"{'PASS' if ok else 'FAIL'} {'match   ' if expect else 'no-match'} {Path(path).name}: {detail}")
    print(f"{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
