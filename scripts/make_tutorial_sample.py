"""Build the tutorial sample CSV from a real TaiChi recording.

Trims a Trigno Discover CSV to 6 seconds of data (keeping every sensor)
and runs the cleaner on the result so the bundled reference PDF stays
in sync with what the tutorial walks through.

Outputs:

* ``tutorials/data/taichi_trial5_6s.csv`` — the trimmed CSV.
* ``tutorials/data/taichi_trial5_6s_cleaning_report.pdf`` — generated
  by ``Log.clean_emg_ekg_artifact()`` against the trimmed CSV.

Usage::

    python scripts/make_tutorial_sample.py            # uses the canonical S:/ path
    python scripts/make_tutorial_sample.py <input>    # override source
"""

import argparse
import os
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))

# Reuse the shared time-only trimmer used to build the test fixtures.
from make_fixture import _max_sr_from_discover_hdr, trim  # noqa: E402

from delsys._parse import _parse_hdr  # noqa: E402

DEFAULT_SOURCE = "S:/2210000787 - TaiChi/data/005/delsys/Trial_5.csv"
DEFAULT_DURATION_S = 6.0


def _rows_for_duration(input_path: Path, duration_s: float) -> int:
    hdr = _parse_hdr(str(input_path))
    if hdr["application"] != "Trigno Discover":
        raise RuntimeError(
            f"make_tutorial_sample expects a Trigno Discover CSV; got {hdr['application']!r}"
        )
    max_sr = _max_sr_from_discover_hdr(hdr)
    return max(1, int(round(duration_s * max_sr)))


def _build_report(csv_path: Path) -> Path:
    """Run the cleaner on ``csv_path``; return the generated PDF path."""
    import delsys

    lf = delsys.Log(str(csv_path))
    if lf.emg is None:
        raise RuntimeError(f"trimmed sample {csv_path} has no EMG bundle")
    lf.clean_emg_ekg_artifact(in_place=False)
    return csv_path.parent / f"{csv_path.stem}_cleaning_report.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", default=DEFAULT_SOURCE,
        help=f"Source CSV (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(SCRIPTS_DIR.parent / "tutorials" / "data"),
        help="Output directory for the trimmed sample + report",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_S,
        help="How many seconds of data to keep",
    )
    args = parser.parse_args()

    src = Path(args.input)
    if not src.exists():
        print(
            f"skip make_tutorial_sample: source not found ({src}). "
            "The committed sample is what's checked in — no regeneration needed."
        )
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    duration_tag = re.sub(r"\.0$", "", f"{args.duration:g}")
    out_csv = out_dir / f"taichi_trial5_{duration_tag}s.csv"

    rows = _rows_for_duration(src, args.duration)
    print(f"trimming {src} -> {out_csv} ({rows} rows ≈ {args.duration} s)")
    trim(str(src), str(out_csv), rows=rows)
    size_kb = os.path.getsize(out_csv) / 1024
    print(f"  wrote {out_csv} ({size_kb:.1f} KB)")

    print("running clean_emg_ekg_artifact() to generate reference report...")
    report_path = _build_report(out_csv)
    print(f"  wrote {report_path}")


if __name__ == "__main__":
    main()
