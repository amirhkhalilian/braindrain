#!/usr/bin/env python3
"""
Copy channel names from one EDF to another, overwriting the target in place.

Channels are matched by position: channel 1 of SOURCE names channel 1 of
TARGET, and so on.  Only the label fields in TARGET's header are rewritten,
so the signal data is left untouched.

    copy_channel_names.py SOURCE_EDF TARGET_EDF
"""

import argparse
import sys
from pathlib import Path

LABEL_LEN = 16
ANNOT_LABEL = "EDF Annotations"


def read_labels(path: Path) -> list[str]:
    with open(path, "rb") as f:
        header = f.read(256)
        ns = int(header[252:256].decode("ascii").strip())
        raw = f.read(ns * LABEL_LEN)
    return [raw[i * LABEL_LEN:(i + 1) * LABEL_LEN].decode("ascii").strip()
            for i in range(ns)]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("source", metavar="SOURCE_EDF", help="EDF to take names from")
    p.add_argument("target", metavar="TARGET_EDF", help="EDF to rename (modified in place)")
    args = p.parse_args()

    src, tgt = Path(args.source), Path(args.target)
    if src.resolve() == tgt.resolve():
        p.error("SOURCE_EDF and TARGET_EDF are the same file.")

    src_labels = read_labels(src)
    tgt_labels = read_labels(tgt)

    if len(src_labels) != len(tgt_labels):
        print(f"ERROR: channel count differs: {src.name} has {len(src_labels)}, "
              f"{tgt.name} has {len(tgt_labels)}", file=sys.stderr)
        sys.exit(1)
    for i, (s, t) in enumerate(zip(src_labels, tgt_labels), start=1):
        if (s == ANNOT_LABEL) != (t == ANNOT_LABEL):
            print(f"ERROR: channel {i} is '{t}' in {tgt.name} but '{s}' in "
                  f"{src.name}; the annotation channel must stay in place",
                  file=sys.stderr)
            sys.exit(1)

    with open(tgt, "r+b") as f:
        f.seek(256)
        f.write(b"".join(s.ljust(LABEL_LEN).encode("ascii") for s in src_labels))

    for i, (old, new) in enumerate(zip(tgt_labels, src_labels), start=1):
        flag = "" if old == new else "  *"
        print(f"  {i:3d}  {old:<16} -> {new}{flag}")
    print(f"\nUpdated {tgt}")


if __name__ == "__main__":
    main()
