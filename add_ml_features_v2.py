#!/usr/bin/env python3
"""
Add ML forecast features to expanded_snapshots_v2.jsonl — v2 (fast, no API calls).

Uses existing TA features to approximate ML model inputs.
For each record, computes a synthetic ML prediction based on available TA.

This is a lightweight approximation — the full ML model requires 47 features
which need raw klines. But we can extract ~10 key features from TA and use
a simplified model.

Usage:
    python3 add_ml_features_v2.py
"""

import json
import os
import sys
import math
import numpy as np
from pathlib import Path
from collections import defaultdict

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
INPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v2.jsonl"
OUTPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v3.jsonl"


def compute_ml_from_ta(record: dict) -> dict:
    """
    Compute ML-like prediction from available TA features.

    This approximates what the XGB+LGB ensemble would predict using
    the subset of features available in our TA data.

    Key signals:
    - RSI: overbought (>70) → predict down, oversold (<30) → predict up
    - MACD hist: positive → up, negative → down
    - BB %B: >1.0 → overbought, <0 → oversold
    - Momentum: direction of recent price movement
    - MA cross: golden cross → up, death cross → down
    - Volume ratio: high volume confirms trend
    """
    # Extract TA features
    rsi = record.get("ta_rsi", 0.5)
    macd_hist = record.get("ta_macd_hist", 0.0)
    bb_pct_b = record.get("ta_bb_pct_b", 0.5)
    momentum_5 = record.get("ta_momentum_5", 0.0)
    momentum_10 = record.get("ta_momentum_10", 0.0)
    ma_cross_5_20 = record.get("ta_ma_cross_5_20", 0.0)
    ma_cross_10_20 = record.get("ta_ma_cross_10_20", 0.0)
    vol_ratio = record.get("ta_vol_ratio", 1.0)
    atr_pct = record.get("ta_atr_pct", 0.01)
    stoch_k = record.get("ta_stoch_k", 0.5)
    stoch_d = record.get("ta_stoch_d", 0.5)
    price_vs_sma20 = record.get("ta_price_vs_sma20", 0.0)
    price_vs_ema50 = record.get("ta_price_vs_ema50", 0.0)
    obv = record.get("ta_obv", 0.0)

    # === Signal combination ===
    # Each signal contributes to a composite score
    signals = []

    # RSI signal (inverted: high RSI → likely to go down)
    rsi_signal = -(rsi - 0.5) * 2.0  # [-1, 1], negative when overbought
    signals.append(("rsi", rsi_signal, 0.20))

    # MACD histogram
    macd_signal = np.clip(macd_hist * 10.0, -1.0, 1.0)
    signals.append(("macd", macd_signal, 0.20))

    # BB %B (inverted: high %B → overbought)
    bb_signal = -(bb_pct_b - 0.5) * 2.0
    signals.append(("bb", bb_signal, 0.10))

    # Momentum (direction)
    mom_signal = np.clip(momentum_5 * 5.0 + momentum_10 * 3.0, -1.0, 1.0)
    signals.append(("momentum", mom_signal, 0.20))

    # MA cross (trend)
    cross_signal = np.clip(ma_cross_5_20 * 20.0 + ma_cross_10_20 * 20.0, -1.0, 1.0)
    signals.append(("ma_cross", cross_signal, 0.10))

    # Stochastic (inverted K-D)
    stoch_signal = -(stoch_k - stoch_d) * 2.0
    signals.append(("stoch", stoch_signal, 0.05))

    # Price vs MAs
    price_signal = np.clip(price_vs_sma20 * 10.0 + price_vs_ema50 * 10.0, -1.0, 1.0)
    signals.append(("price", price_signal, 0.10))

    # Volume confirmation (high vol strengthens other signals)
    vol_factor = min(vol_ratio / 2.0, 1.5)

    # === Weighted combination ===
    total_weight = sum(w for _, _, w in signals)
    raw_score = sum(s * w for _, s, w in signals) / total_weight

    # Apply volume confirmation
    score = raw_score * (0.7 + 0.3 * vol_factor)

    # Add mean reversion for extreme RSI
    if rsi > 0.8:
        score -= 0.15
    elif rsi < 0.2:
        score += 0.15

    # Clip final score
    score = np.clip(score, -1.0, 1.0)

    # Convert to probability
    prob_up = np.clip(0.5 + score * 0.35, 0.05, 0.95)

    # Confidence based on signal agreement
    signal_values = [s for _, s, _ in signals]
    signal_std = np.std(signal_values)
    confidence = np.clip(1.0 - signal_std * 2.0, 0.0, 1.0)

    # Edge
    edge = prob_up - 0.5

    return {
        "ml_prob_up": float(prob_up),
        "ml_prob_down": float(1.0 - prob_up),
        "ml_confidence": float(confidence),
        "ml_prediction": 1.0 if prob_up > 0.5 else 0.0,
        "ml_edge": float(edge),
        "ml_signal_strength": float(score),
        "ml_raw_xgb": float(prob_up),  # same as ensemble in this approximation
        "ml_raw_lgb": float(prob_up),
        "ml_features_count": 10,
    }


def main():
    print("=" * 60)
    print("  Add ML Forecast Features (v2 — from TA)")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"[ERROR] Input not found: {INPUT_FILE}")
        return

    count = 0
    prob_up_sum = 0.0

    os.makedirs(OUTPUT_FILE.parent, exist_ok=True)

    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                ml = compute_ml_from_ta(record)

                for k, v in ml.items():
                    record[k] = v

                fout.write(json.dumps(record) + "\n")
                count += 1
                prob_up_sum += ml["ml_prob_up"]

                if count % 10000 == 0:
                    print(f"  Processed {count} records...")

            except Exception as e:
                pass

    print(f"\nDone! Wrote {count} records to {OUTPUT_FILE}")
    print(f"Average ml_prob_up: {prob_up_sum/count:.4f}")

    # Sample
    with open(OUTPUT_FILE) as f:
        sample = json.loads(f.readline())
    ml_keys = [k for k in sample.keys() if k.startswith("ml_")]
    print(f"\nML features: {ml_keys}")
    for k in ml_keys:
        v = sample[k]
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
