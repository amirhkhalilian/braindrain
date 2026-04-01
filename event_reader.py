#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
from collections import defaultdict


def load_events(csv_path, patient_id):
    events = defaultdict(list)

    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)

        header1 = next(reader)  # patient IDs
        header2 = next(reader)  # column names

        # find columns belonging to the requested patient
        idx = [i for i, h in enumerate(header1) if h == patient_id]
        if not idx:
            raise ValueError(f"Patient ID {patient_id} not found in CSV header.")

        # group indices into date/time/event_type triples
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
                    f"{date_str} {time_str[:2]}:{time_str[2:]}", "%m/%d/%y %H:%M"
                )

                events[row[e_i]].append(dt)

    return dict(events)

def print_report(patient_id, events):
    print("=" * 60)
    print(f"Patient: {patient_id}")
    print("=" * 60)

    # flatten event dictionary into list of (datetime, event_type)
    all_events = []
    for event_type, dts in events.items():
        for dt in dts:
            all_events.append((dt, event_type))

    # sort by datetime
    all_events.sort(key=lambda x: x[0])

    prev_dt = None

    print(f"{'Datetime':20} {'Δt (hrs)':>10}  Event")
    print("-" * 60)

    for dt, event_type in all_events:
        if prev_dt is None:
            delta_str = "----"
        else:
            delta = (dt - prev_dt).total_seconds() / 3600
            delta_str = f"{delta:6.2f}"

        print(f"{dt.strftime('%Y-%m-%d %H:%M'):20} {delta_str:>10}  {event_type}")

        prev_dt = dt

    print("-" * 60)
    print(f"Total events: {len(all_events)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("patient_id")
    args = parser.parse_args()

    print(args)

    events = load_events(args.csv_path, args.patient_id)
    print_report(args.patient_id, events)


if __name__ == "__main__":
    main()
