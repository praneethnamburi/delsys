"""Trim a real Delsys CSV down to a small fixture suitable for tests.

The trimmer keeps the full header block plus ``--rows`` lines of data
(default 1000). For Trigno Discover files it also rewrites
``Collection Length (seconds):`` so the per-format parser's
``round(duration * max(sr)) == len(df)`` invariant still holds.

Usage::

    python scripts/make_fixture.py <input.csv> <output.csv> [--rows N]

Run with no arguments for an interactive default that produces all
seven canonical fixtures from ``C:/dev/immersionToolbox/_data``.
"""
import argparse
import csv
import os
import re
import sys
from pathlib import Path

# Allow `python scripts/make_fixture.py` to find the package without install.
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

from delsys._parse import _parse_hdr  # noqa: E402


def _trimmed_duration(rows_kept: int, max_sr_hz: float) -> float:
    """Find a duration value such that ``round(duration * max_sr) == rows_kept``."""
    # Start at the obvious choice, nudge up if rounding lands short.
    duration = rows_kept / max_sr_hz
    for _ in range(5):
        if round(duration * max_sr_hz) == rows_kept:
            return duration
        duration += 1e-9
    raise RuntimeError(
        f"Could not find a duration matching {rows_kept} rows at max_sr={max_sr_hz}"
    )


def _max_sr_from_discover_hdr(hdr) -> float:
    """Pull the largest finite sampling rate out of a Discover header dict.

    Works for both 1.4.2 (rate embedded in column-name parens) and 1.5.0+
    (separate sampling-rate row).
    """
    pattern = re.compile(r"\((-?[^)]+)Hz")
    rates = []
    for col in hdr["sensor_signal_names"]:
        for m in pattern.findall(col):
            try:
                rates.append(float(m.strip()))
            except ValueError:
                continue
    finite_positive = [r for r in rates if r > 0]
    if not finite_positive:
        raise RuntimeError("No positive sampling rates found in header")
    return max(finite_positive)


def trim(input_path: str, output_path: str, rows: int = 1000) -> None:
    """Trim ``input_path`` to ``output_path`` keeping ``rows`` data rows."""
    hdr = _parse_hdr(input_path)

    with open(input_path, "r", newline="") as f_in:
        all_lines = f_in.readlines()

    if hdr["application"] == "EMGworks":
        # First row is the column header; keep it + ``rows`` data lines.
        kept = all_lines[: 1 + rows]
        with open(output_path, "w", newline="") as f_out:
            f_out.writelines(kept)
        return

    # Trigno Discover
    skiprows = hdr["skiprows"]
    header_block = all_lines[:skiprows]
    data_block = all_lines[skiprows : skiprows + rows]
    rows_kept = len(data_block)
    if rows_kept == 0:
        raise RuntimeError(f"No data rows found in {input_path} after skiprows={skiprows}")

    max_sr = _max_sr_from_discover_hdr(hdr)
    new_duration = _trimmed_duration(rows_kept, max_sr)

    # Rewrite the duration line. Use csv to preserve the trailing whitespace
    # / comma layout.
    rewritten_header = []
    for line in header_block:
        if line.strip().startswith("Collection Length"):
            # Format: "Collection Length (seconds):, <value>\n"
            parts = list(csv.reader([line]))[0]
            parts[-1] = f" {new_duration:.6f}"
            rewritten_header.append(",".join(parts) + "\n")
        else:
            rewritten_header.append(line)

    with open(output_path, "w", newline="") as f_out:
        f_out.writelines(rewritten_header)
        f_out.writelines(data_block)


# Per-fixture row counts. EMGworks uses time-stamped per-channel pairs that
# pandas reads independently, so 200 rows is enough; Discover is a fixed grid
# at the highest sampling rate, so we keep enough rows for ~0.1 s of data.
_DEFAULT_BUILDS = [
    ("emgworks.csv",
     "C:/dev/immersionToolbox/_data/emgworks/mrs01_s009_g01_delsys_01_Rep_4.3.csv", 200),
    ("discover142.csv",
     "C:/dev/immersionToolbox/_data/discover142/Trial_5.csv", 250),
    ("discover150.csv",
     "C:/dev/immersionToolbox/_data/discover150/Trial_1.csv", 250),
    ("discover164_link.csv",
     "C:/dev/immersionToolbox/_data/discover164/Trial_10_ts.csv", 250),
    ("discover164_basic.csv",
     "C:/dev/immersionToolbox/_data/discover164/Trial_10_removeVo2.csv", 250),
    ("discover164_mvc.csv",
     "C:/dev/immersionToolbox/_data/discover164/MVC_1.csv", 500),
    ("discover170.csv",
     "C:/dev/immersionToolbox/_data/discover170/Trial_5.csv", 200),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="Path to source CSV (omit for default batch)")
    parser.add_argument("output", nargs="?", help="Path to write trimmed CSV")
    parser.add_argument("--rows", type=int, default=1000, help="Number of data rows to keep")
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPTS_DIR.parent / "tests" / "fixtures"),
        help="Output directory for the default batch",
    )
    args = parser.parse_args()

    if args.input and args.output:
        trim(args.input, args.output, rows=args.rows)
        print(f"wrote {args.output}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, src, rows in _DEFAULT_BUILDS:
        dst = out_dir / name
        if not os.path.exists(src):
            print(f"skip {name}: source not found ({src})")
            continue
        trim(src, str(dst), rows=rows)
        size_kb = os.path.getsize(dst) / 1024
        print(f"wrote {dst} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
