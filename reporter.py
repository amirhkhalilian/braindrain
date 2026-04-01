#!/usr/bin/env python3
"""
EDF File Reporter
-----------------
Scans a directory for .EDF files, extracts metadata (measurement date, duration,
channel count), sorts by measurement date, and checks recording continuity.
"""

import argparse
import sys
from datetime import timedelta
from pathlib import Path

import mne

mne.set_log_level("WARNING")  # suppress verbose MNE output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_edf_files(directory: Path) -> list[Path]:
    """Return all .EDF / .edf files under *directory* (recursive)."""
    return sorted(directory.rglob("*.[Ee][Dd][Ff]"))


def extract_metadata(edf_path: Path) -> dict:
    """Read an EDF file and return its key metadata."""
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        meas_date = raw.info.get("meas_date", None)
        duration_s = raw.n_times / raw.info["sfreq"]          # seconds (float)
        n_channels = len(raw.ch_names)
        return {
            "file": edf_path,
            "meas_date": meas_date,
            "duration_s": duration_s,
            "n_channels": n_channels,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "file": edf_path,
            "meas_date": None,
            "duration_s": None,
            "n_channels": None,
            "error": str(exc),
        }


def fmt_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm"""
    td = timedelta(seconds=seconds)
    total_s = int(td.total_seconds())
    h, rem = divmod(total_s, 3600)
    m, s = divmod(rem, 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def check_continuity(records: list[dict], gap_tolerance_s: float = 1.0) -> list[dict]:
    """
    Compare each recording's end time (meas_date + duration) with the next
    recording's start time. Records that lack a meas_date are skipped.

    Returns a list of gap dictionaries for every consecutive pair.
    """
    # Keep only records that have both meas_date and duration
    dated = [r for r in records if r["meas_date"] is not None and r["duration_s"] is not None]

    gaps = []
    for i in range(len(dated) - 1):
        curr = dated[i]
        nxt = dated[i + 1]

        end_time = curr["meas_date"] + timedelta(seconds=curr["duration_s"])
        gap_s = (nxt["meas_date"] - end_time).total_seconds()
        status = "OK" if abs(gap_s) <= gap_tolerance_s else ("OVERLAP" if gap_s < 0 else "GAP")

        gaps.append({
            "from_file": curr["file"].name,
            "to_file": nxt["file"].name,
            "expected_next": end_time,
            "actual_next": nxt["meas_date"],
            "gap_s": gap_s,
            "status": status,
        })
    return gaps


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------

COL_W = {
    "file": 40,
    "meas_date": 32,
    "duration": 16,
    "channels": 10,
    "error": 30,
}

SEP = "-" * (sum(COL_W.values()) + len(COL_W) * 3 + 1)


def print_report(records: list[dict]) -> None:
    header = (
        f"{'Filename':<{COL_W['file']}} | "
        f"{'Measurement Date':<{COL_W['meas_date']}} | "
        f"{'Duration':<{COL_W['duration']}} | "
        f"{'Channels':<{COL_W['channels']}} | "
        f"{'Error':<{COL_W['error']}}"
    )
    print("\n" + "=" * len(SEP))
    print("EDF FILE REPORT")
    print("=" * len(SEP))
    print(header)
    print(SEP)

    for r in records:
        fname = r["file"].name
        if len(fname) > COL_W["file"]:
            fname = "..." + fname[-(COL_W["file"] - 1):]

        meas_str = str(r["meas_date"]) if r["meas_date"] else "N/A"
        dur_str = fmt_duration(r["duration_s"]) if r["duration_s"] is not None else "N/A"
        ch_str = str(r["n_channels"]) if r["n_channels"] is not None else "N/A"
        err_str = (r["error"] or "")[:COL_W["error"]]

        print(
            f"{fname:<{COL_W['file']}} | "
            f"{meas_str:<{COL_W['meas_date']}} | "
            f"{dur_str:<{COL_W['duration']}} | "
            f"{ch_str:<{COL_W['channels']}} | "
            f"{err_str:<{COL_W['error']}}"
        )
    print(SEP)
    print(f"Total files: {len(records)}\n")


def print_continuity(gaps: list[dict], tolerance_s: float) -> None:
    print("=" * 70)
    print(f"CONTINUITY CHECK  (tolerance +/- {tolerance_s} s)")
    print("=" * 70)

    if not gaps:
        print("  No consecutive dated recordings to compare.\n")
        return

    ok_count = sum(1 for g in gaps if g["status"] == "OK")
    issue_count = len(gaps) - ok_count

    for g in gaps:
        icon = "[OK]" if g["status"] == "OK" else "[!!]"
        sign = "+" if g["gap_s"] >= 0 else ""
        print(
            f"  {icon} {g['from_file']} -> {g['to_file']}\n"
            f"       Expected next start : {g['expected_next']}\n"
            f"       Actual   next start : {g['actual_next']}\n"
            f"       Gap                 : {sign}{g['gap_s']:.3f} s  [{g['status']}]\n"
        )

    print(f"Pairs checked: {len(gaps)}  |  OK: {ok_count}  |  Issues: {issue_count}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a directory for EDF files and produce a continuity report."
    )
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search for EDF files (default: current directory)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Acceptable gap between recordings in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Search only the top-level directory, not subdirectories",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Exclude files whose duration is shorter than this value in seconds (default: 0, no filter)",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="FILENAME",
        default=[],
        help="One or more filenames to exclude (e.g. --skip bad1.edf bad2.edf)",
    )
    args = parser.parse_args()

    search_dir = Path(args.directory).resolve()
    if not search_dir.is_dir():
        print(f"ERROR: '{search_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSearching for EDF files in: {search_dir}")

    if args.no_recurse:
        edf_files = sorted(search_dir.glob("*.[Ee][Dd][Ff]"))
    else:
        edf_files = find_edf_files(search_dir)

    if not edf_files:
        print("No EDF files found.")
        sys.exit(0)

    skip_names = {name.lower() for name in args.skip}
    if skip_names:
        before = len(edf_files)
        edf_files = [f for f in edf_files if f.name.lower() not in skip_names]
        skipped = before - len(edf_files)
        print(f"Skipping {skipped} file(s): {', '.join(args.skip)}")

    print(f"Found {len(edf_files)} EDF file(s). Reading metadata ...\n")

    records = [extract_metadata(f) for f in edf_files]

    if args.min_duration > 0:
        before = len(records)
        records = [r for r in records if r["duration_s"] is None or r["duration_s"] >= args.min_duration]
        dropped = before - len(records)
        if dropped:
            print(f"Excluded {dropped} file(s) with duration < {args.min_duration} s.")

    # Sort: files without a meas_date go to the end
    records.sort(key=lambda r: (r["meas_date"] is None, r["meas_date"] or 0))

    print_report(records)

    gaps = check_continuity(records, gap_tolerance_s=args.tolerance)
    print_continuity(gaps, tolerance_s=args.tolerance)


if __name__ == "__main__":
    main()
