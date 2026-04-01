#!/usr/bin/env python3
"""
EDF Clip Generator
------------------
Crop an EDF file into one or more clips and save each as a new EDF.

Start times can be expressed as:
  - An absolute ISO-8601 datetime  e.g.  2019-08-03T21:00:00
    (assumed UTC if no timezone offset is given)
  - Seconds from the start of the recording  e.g.  3600

Two modes
---------
Single / explicit clips
  Supply --clip one or more times.  Each --clip takes exactly two values:
  START  DURATION_SECONDS.

      edf_clip.py rec.edf --out-dir clips/ --clip 2019-08-03T21:00:00 3600

Sequential clips
  Supply --seq-start, --seq-duration, and --n-clips.
  The first clip starts at seq-start; every following clip starts immediately
  after the previous one ends.

      edf_clip.py rec.edf --out-dir clips/ \\
          --seq-start 2019-08-03T21:00:00 --seq-duration 3600 --n-clips 8

Both modes can be combined in the same call.
"""

import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import mne

mne.set_log_level("WARNING")

from mne.io import read_raw_edf as _read_raw_edf

def edf_start_datetime(raw):
    if isinstance(raw, (str, Path)):
        raw = _read_raw_edf(raw, preload=False, verbose=False)
    dt = raw.info.get("meas_date", None)
    if dt is None:
        raise ValueError("EDF does not contain a valid start date/time")
    if isinstance(dt, (int, float)):
        dt = datetime.fromtimestamp(dt, tz=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def datetime_to_edf_seconds(raw, target_dt):
    start_dt = edf_start_datetime(raw)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=start_dt.tzinfo)
    return (target_dt - start_dt).total_seconds()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_start(value: str) -> datetime | float:
    """
    Accept either an ISO-8601 datetime string or a plain float (seconds).
    Returns a timezone-aware datetime or a float.
    """
    try:
        return float(value)
    except ValueError:
        pass
    # Try ISO-8601 with or without offset
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(value, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse start time '{value}'. "
        "Use an ISO-8601 datetime (e.g. 2019-08-03T21:00:00) "
        "or seconds from recording start (e.g. 3600)."
    )


def resolve_to_seconds(start_val: datetime | float, edf_start: datetime) -> float:
    """Convert a start value (datetime or float offset) to seconds from EDF start."""
    if isinstance(start_val, (int, float)):
        return float(start_val)
    if start_val.tzinfo is None:
        start_val = start_val.replace(tzinfo=edf_start.tzinfo)
    offset = (start_val - edf_start).total_seconds()
    if offset < 0:
        raise ValueError(
            f"Start time {start_val} is before the recording start {edf_start}."
        )
    return offset


def make_output_name(out_dir: Path, stem: str, clip_index: int,
                     t_start_abs: datetime, t_end_abs: datetime) -> Path:
    """Build an output filename: <stem>_clip<N>_<start>_<end>.edf"""
    ts_start = t_start_abs.strftime("%Y%m%d_%H%M%S")
    ts_end   = t_end_abs.strftime("%Y%m%d_%H%M%S")
    fname = f"{stem}_clip{clip_index:03d}_{ts_start}_{ts_end}.edf"
    return out_dir / fname


def export_clip(raw: mne.io.BaseRaw, edf_start: datetime,
                t_start_s: float, t_end_s: float, out_path: Path) -> None:
    """Crop *raw* to [t_start_s, t_end_s] and export to *out_path*."""
    rec_duration = raw.n_times / raw.info["sfreq"]
    if t_start_s >= rec_duration:
        raise ValueError(
            f"Clip start ({t_start_s:.1f} s) is beyond recording end "
            f"({rec_duration:.1f} s)."
        )
    t_end_s = min(t_end_s, rec_duration)

    raw_crop = raw.copy().crop(tmin=t_start_s, tmax=t_end_s)

    # Update meas_date to the actual clip start
    new_meas_date = edf_start + timedelta(seconds=t_start_s)
    raw_crop.set_meas_date(new_meas_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_crop.export(str(out_path), fmt="edf", overwrite=True)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

class _TwoValueAction(argparse.Action):
    """Collect pairs of (START, DURATION) from repeated --clip flags."""
    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest, None) or []
        items.append(values)          # values is already a 2-element list
        setattr(namespace, self.dest, items)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input",
        metavar="INPUT_EDF",
        help="Source EDF file to clip",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        metavar="DIR",
        help="Directory where clip files will be saved",
    )

    # ---- explicit clips ----
    p.add_argument(
        "--clip",
        nargs=2,
        metavar=("START", "DURATION_SEC"),
        action=_TwoValueAction,
        dest="clips",
        help=(
            "Define one clip: START (ISO datetime or seconds offset) and "
            "DURATION in seconds.  Repeat for multiple clips."
        ),
    )

    # ---- sequential clips ----
    seq = p.add_argument_group("sequential clips")
    seq.add_argument(
        "--seq-start",
        metavar="START",
        help="Start of the first sequential clip (ISO datetime or seconds offset)",
    )
    seq.add_argument(
        "--seq-duration",
        type=float,
        metavar="SECONDS",
        help="Duration of each sequential clip in seconds",
    )
    seq.add_argument(
        "--n-clips",
        type=int,
        metavar="N",
        help="Number of sequential clips to generate",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # --- validate sequential args ---
    seq_args = (args.seq_start, args.seq_duration, args.n_clips)
    if any(a is not None for a in seq_args):
        if not all(a is not None for a in seq_args):
            parser.error(
                "--seq-start, --seq-duration, and --n-clips must all be "
                "supplied together."
            )
        if args.n_clips < 1:
            parser.error("--n-clips must be >= 1.")

    if not args.clips and not args.seq_start:
        parser.error(
            "Provide at least one --clip START DURATION  "
            "or use --seq-start / --seq-duration / --n-clips."
        )

    in_path = Path(args.input).resolve()
    if not in_path.is_file():
        print(f"ERROR: Input file not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()

    # --- load EDF header once ---
    print(f"\nReading EDF header: {in_path.name}")
    raw = mne.io.read_raw_edf(str(in_path), preload=False, verbose=False)
    edf_start = edf_start_datetime(raw)
    rec_duration_s = raw.n_times / raw.info["sfreq"]
    stem = in_path.stem

    print(f"  Recording start : {edf_start}")
    print(f"  Duration        : {rec_duration_s:.1f} s  ({rec_duration_s/3600:.3f} h)")
    print(f"  Channels        : {len(raw.ch_names)}")
    print(f"  Output dir      : {out_dir}\n")

    # --- build clip list ---
    clip_jobs: list[tuple[float, float]] = []   # (t_start_s, t_end_s)

    # explicit --clip entries
    if args.clips:
        for raw_start, raw_dur in args.clips:
            start_val = parse_start(raw_start)
            t_start_s = resolve_to_seconds(start_val, edf_start)
            t_end_s   = t_start_s + float(raw_dur)
            clip_jobs.append((t_start_s, t_end_s))

    # sequential clips
    if args.seq_start:
        seq_start_val = parse_start(args.seq_start)
        t_cursor = resolve_to_seconds(seq_start_val, edf_start)
        for _ in range(args.n_clips):
            clip_jobs.append((t_cursor, t_cursor + args.seq_duration))
            t_cursor += args.seq_duration

    print(f"Clips to generate: {len(clip_jobs)}")
    print("-" * 50)

    # --- generate clips ---
    errors = []
    for idx, (t_start_s, t_end_s) in enumerate(clip_jobs, start=1):
        t_start_abs = edf_start + timedelta(seconds=t_start_s)
        t_end_abs   = edf_start + timedelta(seconds=min(t_end_s, rec_duration_s))
        out_path = make_output_name(out_dir, stem, idx, t_start_abs, t_end_abs)

        print(
            f"  [{idx:03d}/{len(clip_jobs):03d}]  "
            f"{t_start_abs.strftime('%H:%M:%S')} -> "
            f"{t_end_abs.strftime('%H:%M:%S')}  "
            f"({t_end_s - t_start_s:.1f} s)  =>  {out_path.name}"
        )
        try:
            export_clip(raw, edf_start, t_start_s, t_end_s, out_path)
        except Exception as exc:  # noqa: BLE001
            print(f"         ERROR: {exc}")
            errors.append((idx, str(exc)))

    # --- summary ---
    print("-" * 50)
    ok = len(clip_jobs) - len(errors)
    print(f"\nDone.  {ok}/{len(clip_jobs)} clip(s) saved to {out_dir}")
    if errors:
        print(f"  {len(errors)} error(s):")
        for idx, msg in errors:
            print(f"    clip {idx:03d}: {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
