#!/usr/bin/env python3
"""
EDF Event Mapper
----------------
Combines EDF file scanning (from edf_report) with patient event loading
(from event_reader) to locate which EDF file each event falls inside.

An event at time T belongs to a file when:
    meas_date <= T < meas_date + duration_s

Usage
-----
    python edf_event_mapper.py /path/to/edfs events.csv PATIENT_ID [options]

Options
-------
    --tolerance SEC    Seconds of slack applied to file boundaries (default 0)
    --no-recurse       Only scan the top-level EDF directory
    --skip FILE ...    EDF filenames to exclude
    --min-duration S   Ignore EDF files shorter than S seconds
"""

import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import mne

mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# Event loading  (from event_reader.py)
# ---------------------------------------------------------------------------

def load_events(csv_path: str, patient_id: str) -> list[tuple[datetime, str]]:
    """
    Return a flat, time-sorted list of (datetime, event_type) for *patient_id*.
    datetimes are made timezone-aware (UTC) so they compare cleanly with
    meas_date values from MNE (which are always UTC-aware).
    """
    events: dict[str, list[datetime]] = defaultdict(list)

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header1 = next(reader)   # patient IDs
        header2 = next(reader)   # column names

        idx = [i for i, h in enumerate(header1) if h == patient_id]
        if not idx:
            raise ValueError(f"Patient ID '{patient_id}' not found in CSV header.")

        groups = []
        for i in idx:
            if header2[i] == "date":
                groups.append((i, i + 1, i + 2))

        for row in reader:
            for d_i, t_i, e_i in groups:
                if not row[d_i] or not row[t_i] or not row[e_i]:
                    continue
                date_str = row[d_i].strip()
                time_str = row[t_i].strip().zfill(4)
                dt = datetime.strptime(
                    f"{date_str} {time_str[:2]}:{time_str[2:]}",
                    "%m/%d/%y %H:%M",
                )
                # make UTC-aware so comparisons with meas_date work
                dt = dt.replace(tzinfo=timezone.utc)
                events[row[e_i]].append(dt)

    flat = [(dt, etype) for etype, dts in events.items() for dt in dts]
    flat.sort(key=lambda x: x[0])
    return flat


# ---------------------------------------------------------------------------
# EDF scanning  (from edf_report.py)
# ---------------------------------------------------------------------------

def find_edf_files(directory: Path, recurse: bool = True) -> list[Path]:
    pattern = "*.[Ee][Dd][Ff]"
    return sorted(directory.rglob(pattern) if recurse else directory.glob(pattern))


def extract_metadata(edf_path: Path) -> dict:
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        meas_date = raw.info.get("meas_date", None)
        duration_s = raw.n_times / raw.info["sfreq"]
        n_channels = len(raw.ch_names)
        return {
            "file": edf_path,
            "meas_date": meas_date,
            "duration_s": duration_s,
            "n_channels": n_channels,
            "error": None,
        }
    except Exception as exc:
        return {
            "file": edf_path,
            "meas_date": None,
            "duration_s": None,
            "n_channels": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Matching logic
# ---------------------------------------------------------------------------

def match_events_to_files(
    records: list[dict],
    events: list[tuple[datetime, str]],
    tolerance_s: float = 0.0,
) -> list[dict]:
    """
    For every event return a result dict with the matched EDF record (or None).

    A file matches when:
        meas_date - tolerance <= event_dt < meas_date + duration_s + tolerance
    """
    # Only consider records that have valid timing
    dated = [
        r for r in records
        if r["meas_date"] is not None and r["duration_s"] is not None
    ]

    results = []
    for event_dt, event_type in events:
        match = None
        for r in dated:
            start = r["meas_date"] - timedelta(seconds=tolerance_s)
            end   = r["meas_date"] + timedelta(seconds=r["duration_s"] + tolerance_s)
            if start <= event_dt < end:
                match = r
                break   # first (earliest) match wins; files shouldn't overlap

        results.append({
            "event_dt":   event_dt,
            "event_type": event_type,
            "match":      match,
        })
    return results


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

SEP_WIDE  = "=" * 90
SEP_THIN  = "-" * 90


def print_event_report(
    patient_id: str,
    match_results: list[dict],
    records: list[dict],
) -> None:
    matched   = [r for r in match_results if r["match"] is not None]
    unmatched = [r for r in match_results if r["match"] is None]

    # ---- per-file summary ------------------------------------------------
    file_to_events: dict[Path, list[dict]] = defaultdict(list)
    for r in matched:
        file_to_events[r["match"]["file"]].append(r)

    print(f"\n{SEP_WIDE}")
    print(f"  EDF EVENT MAPPER  —  Patient: {patient_id}")
    print(SEP_WIDE)
    print(f"  EDF files with timing info : {sum(1 for r in records if r['meas_date'] is not None)}")
    print(f"  Total events               : {len(match_results)}")
    print(f"  Matched                    : {len(matched)}")
    print(f"  Unmatched                  : {len(unmatched)}")
    print(SEP_WIDE)

    # ---- event-by-file breakdown -----------------------------------------
    print("\n[ Events grouped by EDF file ]\n")

    # iterate in file order (sorted by meas_date)
    for rec in records:
        if rec["file"] not in file_to_events:
            continue
        evts = file_to_events[rec["file"]]
        meas = rec["meas_date"]
        end  = meas + timedelta(seconds=rec["duration_s"])
        print(f"  File     : {rec['file'].name}")
        print(f"  Window   : {meas}  ->  {end}  ({rec['duration_s']:.1f} s)")
        print(f"  Events   : {len(evts)}")
        print(f"  {'Datetime (UTC)':25} {'Offset (s)':>12}  Event Type")
        print("  " + "-" * 60)
        for e in evts:
            offset_s = (e["event_dt"] - meas).total_seconds()
            print(
                f"  {e['event_dt'].strftime('%Y-%m-%d %H:%M:%S %Z'):25} "
                f"{offset_s:>12.1f}  {e['event_type']}"
            )
        print()

    # ---- flat chronological table ----------------------------------------
    print(SEP_THIN)
    print("\n[ All events — chronological ]\n")
    print(f"  {'Datetime (UTC)':25} {'Event Type':20}  {'EDF File':40}  {'Offset (s)':>10}")
    print("  " + "-" * 100)
    for r in match_results:
        dt_str    = r["event_dt"].strftime("%Y-%m-%d %H:%M:%S %Z")
        etype     = r["event_type"]
        if r["match"]:
            fname    = r["match"]["file"].name
            offset_s = (r["event_dt"] - r["match"]["meas_date"]).total_seconds()
            off_str  = f"{offset_s:.1f}"
        else:
            fname   = "*** NO MATCH ***"
            off_str = "---"
        if len(fname) > 40:
            fname = "..." + fname[-37:]
        print(f"  {dt_str:25} {etype:20}  {fname:40}  {off_str:>10}")

    # ---- unmatched events ------------------------------------------------
    if unmatched:
        print(f"\n{SEP_THIN}")
        print(f"\n[ Unmatched events ({len(unmatched)}) ]\n")
        print(f"  {'Datetime (UTC)':25}  Event Type")
        print("  " + "-" * 50)
        for r in unmatched:
            print(
                f"  {r['event_dt'].strftime('%Y-%m-%d %H:%M:%S %Z'):25}  "
                f"{r['event_type']}"
            )

    print(f"\n{SEP_WIDE}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("edf_dir",     help="Directory containing EDF files")
    parser.add_argument("csv_path",    help="Events CSV file")
    parser.add_argument("patient_id",  help="Patient ID as it appears in the CSV header")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Boundary slack in seconds when matching events to files (default: 0)",
    )
    parser.add_argument(
        "--no-recurse",
        action="store_true",
        help="Search only the top-level EDF directory",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="FILENAME",
        default=[],
        help="EDF filenames to exclude",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Ignore EDF files shorter than this duration (default: 0)",
    )
    args = parser.parse_args()

    # --- EDF files ---
    edf_dir = Path(args.edf_dir).resolve()
    if not edf_dir.is_dir():
        print(f"ERROR: '{edf_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    print(f"\nSearching for EDF files in: {edf_dir}")
    edf_files = find_edf_files(edf_dir, recurse=not args.no_recurse)

    skip_names = {n.lower() for n in args.skip}
    if skip_names:
        before = len(edf_files)
        edf_files = [f for f in edf_files if f.name.lower() not in skip_names]
        print(f"Skipped {before - len(edf_files)} file(s) by name.")

    if not edf_files:
        print("No EDF files found.")
        sys.exit(0)

    print(f"Reading metadata from {len(edf_files)} EDF file(s) ...")
    records = [extract_metadata(f) for f in edf_files]

    if args.min_duration > 0:
        before = len(records)
        records = [
            r for r in records
            if r["duration_s"] is None or r["duration_s"] >= args.min_duration
        ]
        dropped = before - len(records)
        if dropped:
            print(f"Excluded {dropped} file(s) with duration < {args.min_duration} s.")

    # sort by meas_date; undated files go to the end
    records.sort(key=lambda r: (r["meas_date"] is None, r["meas_date"] or 0))

    # --- Events ---
    print(f"Loading events for patient '{args.patient_id}' from: {args.csv_path}")
    try:
        events = load_events(args.csv_path, args.patient_id)
    except (ValueError, FileNotFoundError) as exc:
        print(f"ERROR loading events: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(events)} event(s).")

    # --- Match ---
    match_results = match_events_to_files(records, events, tolerance_s=args.tolerance)

    # --- Report ---
    print_event_report(args.patient_id, match_results, records)


if __name__ == "__main__":
    main()
