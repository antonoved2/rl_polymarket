#!/usr/bin/env python3
"""
Add multi-horizon ML forecast features to expanded_snapshots_v3.jsonl.

For each record, computes ML predictions at h=1,3,5,10 using TA features.
Adds 12 new features: ml_h{N}_prob_up, ml_h{N}_confidence, ml_h{N}_edge.

Output: expanded_snapshots_v4.jsonl
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ml_multi_horizon import MultiHorizonML

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
INPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v3.jsonl"
OUTPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v4.jsonl"


def extract_ta_features(record: dict) -> dict:
    """Extract TA features from record (strip 'ta_' prefix)."""
    ta = {}
    for k, v in record.items():
        if k.startswith("ta_"):
            ta[k[3:]] = v
    return ta


def main():
    print("=" * 60)
    print("  Add Multi-Horizon ML Features")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60)

    ml = MultiHorizonML(horizons=[1, 3, 5, 10])

    count = 0
    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                ta = extract_ta_features(record)
                ml_pred = ml.predict_from_ta(ta)

                for k, v in ml_pred.items():
                    record[k] = v

                fout.write(json.dumps(record) + "\n")
                count += 1

                if count % 10000 == 0:
                    print(f"  Processed {count}...")

            except Exception as e:
                pass

    print(f"\nDone! Wrote {count} records")

    # Verify
    with open(OUTPUT_FILE) as f:
        sample = json.loads(f.readline())
    mh_keys = sorted([k for k in sample.keys() if k.startswith("ml_h")])
    print(f"Multi-horizon features: {mh_keys}")
    for k in mh_keys:
        print(f"  {k}: {sample[k]:.4f}")


if __name__ == "__main__":
    main()
