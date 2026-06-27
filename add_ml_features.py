#!/usr/bin/env python3
"""
Add ML forecast features to expanded_snapshots_v2.jsonl.

Strategy: For each unique period_start, fetch klines at that time from Binance,
compute ML features, run prediction, and apply to all steps in that period.

This reduces API calls from 95K to ~360 (one per period).

Usage:
    python3 add_ml_features.py [--input data/expanded_snapshots_v2.jsonl] [--output data/expanded_snapshots_v3.jsonl]
"""

import json
import os
import sys
import time
import requests
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from ml_forecast import MLForecastModel

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
INPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v2.jsonl"
OUTPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v3.jsonl"

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=100"


def fetch_klines_for_period(symbol: str, period_start: int, limit: int = 100):
    """
    Fetch klines ending around period_start.
    Binance klines are fetched by time range.
    """
    try:
        # Fetch klines ending at period_start + 900 (end of 15-min period)
        end_ts = (period_start + 900) * 1000  # ms
        start_ts = end_ts - limit * 300 * 1000  # 5m candles
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval=5m&startTime={start_ts}&endTime={end_ts}&limit={limit}")
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data:
                return data
    except:
        pass

    # Fallback: just get latest klines
    try:
        r = requests.get(
            BINANCE_KLINES_URL.format(symbol=symbol, limit=limit),
            timeout=15
        )
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return []


def add_ml_features(input_path: str, output_path: str, asset: str = "btc"):
    """Add ML forecast features to all records."""
    print("=" * 60)
    print("  Add ML Forecast Features")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print("=" * 60)

    # Load ML model
    print("\nLoading ML model...")
    ml_model = MLForecastModel(horizon=10)

    # Read all records and group by period
    print("\nReading records...")
    records_by_period = defaultdict(list)
    all_records = []

    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            all_records.append(record)
            period = record.get("period_start", 0)
            records_by_period[period].append(record)

    print(f"Total records: {len(all_records)}")
    print(f"Unique periods: {len(records_by_period)}")

    # For each period, compute ML prediction
    print("\nComputing ML predictions per period...")
    symbol = f"{asset.upper()}USDT"
    period_predictions = {}
    periods = sorted(records_by_period.keys())

    for i, period in enumerate(periods):
        klines = fetch_klines_for_period(symbol, period)

        if klines:
            ml_features = ml_model.compute_features(klines)
            if ml_features:
                pred = ml_model.predict(ml_features)
                period_predictions[period] = pred
            else:
                period_predictions[period] = ml_model._default_prediction()
        else:
            period_predictions[period] = ml_model._default_prediction()

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(periods)} periods")

        time.sleep(0.12)  # ~8 req/sec

    print(f"\nGot predictions for {len(period_predictions)}/{len(periods)} periods")

    # Apply predictions to all records
    print("\nWriting output with ML features...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    count = 0
    default_pred = ml_model._default_prediction()

    with open(output_path, "w") as fout:
        for record in all_records:
            period = record.get("period_start", 0)
            pred = period_predictions.get(period, default_pred)

            for k, v in pred.items():
                record[k] = v

            fout.write(json.dumps(record) + "\n")
            count += 1

            if count % 10000 == 0:
                print(f"  Wrote {count} records...")

    print(f"\nDone! Wrote {count} records to {output_path}")

    # Verify
    with open(output_path) as f:
        sample = json.loads(f.readline())
    ml_keys = [k for k in sample.keys() if k.startswith("ml_")]
    print(f"\nML features in output: {ml_keys}")
    for k in ml_keys:
        v = sample[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(INPUT_FILE)
    output_file = sys.argv[2] if len(sys.argv) > 2 else str(OUTPUT_FILE)

    add_ml_features(input_file, output_file)
