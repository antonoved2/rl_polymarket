#!/usr/bin/env python3
"""
Expand snapshots with order book + trade flow features.

For historical data (where real OB is unavailable):
  - Estimate spread from klines: (high-low)/close as proxy
  - Estimate imbalance from kline body: (close-open)/(high-low) as proxy
  - Estimate depth from volume: normalize recent volume

For real-time/live bot:
  - Fetch actual order book from Binance /api/v3/depth
  - Fetch recent trades from /api/v3/trades

Output: expanded_snapshots_v2.jsonl with additional ob_* and tf_* fields.
"""

import json
import os
import sys
import math
import time
import requests
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
LEGACY_DIR = WORKSPACE / "backtest"
OUTPUT_DIR = WORKSPACE / "rl_polymarket" / "data"

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
BINANCE_TRADES_URL = "https://api.binance.com/api/v3/trades?symbol={symbol}&limit=100"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=50"

# ═══════════════════════════════════════════════════════════════════
# Binance API helpers
# ═══════════════════════════════════════════════════════════════════

def fetch_order_book(symbol="BTCUSDT", limit=20):
    """Fetch order book from Binance."""
    try:
        r = requests.get(
            BINANCE_DEPTH_URL.format(symbol=symbol, limit=limit),
            timeout=10
        )
        if r.status_code == 200:
            data = r.json()
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
            return bids, asks
    except Exception as e:
        print(f"  [WARN] OB fetch failed: {e}")
    return [], []


def fetch_recent_trades(symbol="BTCUSDT", limit=100):
    """Fetch recent trades from Binance."""
    try:
        r = requests.get(
            BINANCE_TRADES_URL.format(symbol=symbol, limit=limit),
            timeout=10
        )
        if r.status_code == 200:
            trades = []
            for t in r.json():
                trades.append({
                    "price": float(t["price"]),
                    "qty": float(t["qty"]),
                    "is_buyer_maker": t.get("isBuyerMaker", False),
                    "time": t.get("time", 0),
                })
            return trades
    except Exception as e:
        print(f"  [WARN] Trades fetch failed: {e}")
    return []


def fetch_klines(symbol="BTCUSDT", interval="5m", limit=50):
    """Fetch OHLCV klines from Binance."""
    try:
        r = requests.get(
            BINANCE_KLINES_URL.format(symbol=symbol, interval=interval, limit=limit),
            timeout=10
        )
        if r.status_code == 200:
            return [{
                "o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                "c": float(k[4]), "v": float(k[5]), "ts": k[0] // 1000
            } for k in r.json()]
    except:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════
# Order Book Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def compute_ob_features(bids, asks):
    """Compute order book features from bids/asks."""
    features = {
        "ob_imbalance": 0.0,
        "ob_spread": 0.0,
        "ob_spread_bps": 0.0,
        "ob_bid_depth_5": 0.0,
        "ob_ask_depth_5": 0.0,
        "ob_bid_depth_10": 0.0,
        "ob_ask_depth_10": 0.0,
        "ob_bid_depth_20": 0.0,
        "ob_ask_depth_20": 0.0,
        "ob_depth_imbalance_5": 0.0,
        "ob_depth_imbalance_20": 0.0,
        "ob_wall_bid": 0.0,
        "ob_wall_ask": 0.0,
        "ob_slope_bid": 0.0,
        "ob_slope_ask": 0.0,
    }

    if not bids or not asks:
        return features

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid_price = (best_bid + best_ask) / 2.0

    if mid_price <= 0:
        return features

    # Spread
    spread = best_ask - best_bid
    features["ob_spread"] = spread / mid_price  # normalized
    features["ob_spread_bps"] = spread / mid_price * 10000  # basis points

    # Imbalance (volume-weighted)
    total_bid_vol = sum(q for _, q in bids)
    total_ask_vol = sum(q for _, q in asks)
    total_vol = total_bid_vol + total_ask_vol
    if total_vol > 0:
        features["ob_imbalance"] = (total_bid_vol - total_ask_vol) / total_vol

    # Depth at different levels
    for level, key in [(5, "_5"), (10, "_10"), (20, "_20")]:
        bid_vol = sum(q for _, q in bids[:level])
        ask_vol = sum(q for _, q in asks[:level])
        features[f"ob_bid_depth{key}"] = bid_vol
        features[f"ob_ask_depth{key}"] = ask_vol
        depth_total = bid_vol + ask_vol
        if depth_total > 0:
            features[f"ob_depth_imbalance{key}"] = (bid_vol - ask_vol) / depth_total

    # Large walls (>2x average size)
    if bids:
        avg_bid_size = total_bid_vol / len(bids)
        features["ob_wall_bid"] = max((q / avg_bid_size - 1.0, 0.0) for _, q in bids[:5])
    if asks:
        avg_ask_size = total_ask_vol / len(asks)
        features["ob_wall_ask"] = max((q / avg_ask_size - 1.0, 0.0) for _, q in asks[:5])

    # Slope (how fast depth decays)
    if len(bids) >= 5:
        d1 = sum(q for _, q in bids[:1])
        d5 = sum(q for _, q in bids[:5])
        if d1 > 0:
            features["ob_slope_bid"] = (d5 / d1 - 1.0)
    if len(asks) >= 5:
        d1 = sum(q for _, q in asks[:1])
        d5 = sum(q for _, q in asks[:5])
        if d1 > 0:
            features["ob_slope_ask"] = (d5 / d1 - 1.0)

    return features


# ═══════════════════════════════════════════════════════════════════
# Trade Flow Feature Extraction
# ═══════════════════════════════════════════════════════════════════

def compute_trade_features(trades):
    """Compute trade flow features from recent trades."""
    features = {
        "tf_buy_ratio": 0.5,
        "tf_buy_volume": 0.0,
        "tf_sell_volume": 0.0,
        "tf_large_trades": 0.0,
        "tf_avg_size": 0.0,
        "tf_flow_imbalance": 0.0,
        "tf_aggression": 0.0,
        "tf_size_variance": 0.0,
    }

    if not trades:
        return features

    total_vol = 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    sizes = []

    for t in trades:
        vol = t["price"] * t["qty"]
        total_vol += vol
        sizes.append(t["qty"])
        if t["is_buyer_maker"]:
            sell_vol += vol
        else:
            buy_vol += vol

    if total_vol > 0:
        features["tf_buy_ratio"] = buy_vol / total_vol
        features["tf_buy_volume"] = buy_vol
        features["tf_sell_volume"] = sell_vol
        features["tf_flow_imbalance"] = (buy_vol - sell_vol) / total_vol

    # Large trades (>2x median)
    if sizes:
        sorted_sizes = sorted(sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
        if median_size > 0:
            large_count = sum(1 for s in sizes if s > 2 * median_size)
            features["tf_large_trades"] = large_count / len(sizes)
        features["tf_avg_size"] = sum(sizes) / len(sizes)
        # Size variance (normalized)
        mean_size = features["tf_avg_size"]
        if mean_size > 0 and len(sizes) > 1:
            variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            features["tf_size_variance"] = math.sqrt(variance) / mean_size

    # Aggression: ratio of market orders (non-maker) to total
    if trades:
        aggressive = sum(1 for t in trades if not t["is_buyer_maker"])
        features["tf_aggression"] = aggressive / len(trades)

    return features


# ═══════════════════════════════════════════════════════════════════
# Estimated OB from klines (for historical data)
# ═══════════════════════════════════════════════════════════════════

def estimate_ob_from_klines(klines):
    """Estimate order book features from OHLCV klines when real OB unavailable."""
    features = {
        "ob_imbalance": 0.0,
        "ob_spread": 0.0,
        "ob_spread_bps": 0.0,
        "ob_bid_depth_5": 0.0,
        "ob_ask_depth_5": 0.0,
        "ob_bid_depth_10": 0.0,
        "ob_ask_depth_10": 0.0,
        "ob_bid_depth_20": 0.0,
        "ob_ask_depth_20": 0.0,
        "ob_depth_imbalance_5": 0.0,
        "ob_depth_imbalance_20": 0.0,
        "ob_wall_bid": 0.0,
        "ob_wall_ask": 0.0,
        "ob_slope_bid": 0.0,
        "ob_slope_ask": 0.0,
    }

    if not klines or len(klines) < 3:
        return features

    # Estimate spread from recent kline ranges
    recent = klines[-5:]
    avg_range = sum(k["h"] - k["l"] for k in recent) / len(recent)
    avg_close = sum(k["c"] for k in recent) / len(recent)
    if avg_close > 0:
        features["ob_spread"] = avg_range / avg_close * 0.1  # ~10% of range is spread
        features["ob_spread_bps"] = features["ob_spread"] * 10000

    # Estimate imbalance from candle bodies
    total_body = 0.0
    total_range = 0.0
    for k in recent:
        body = k["c"] - k["o"]
        rng = k["h"] - k["l"]
        total_body += body
        total_range += rng

    if total_range > 0:
        features["ob_imbalance"] = total_body / total_range  # [-1, 1]

    # Estimate depth from volume
    avg_vol = sum(k["v"] for k in recent) / len(recent)
    if avg_vol > 0:
        # Normalize: assume avg depth ~ 100 BTC at price level
        features["ob_bid_depth_5"] = avg_vol * 0.3
        features["ob_ask_depth_5"] = avg_vol * 0.3
        features["ob_bid_depth_10"] = avg_vol * 0.6
        features["ob_ask_depth_10"] = avg_vol * 0.6
        features["ob_bid_depth_20"] = avg_vol * 1.0
        features["ob_ask_depth_20"] = avg_vol * 1.0

    return features


def estimate_trades_from_klines(klines):
    """Estimate trade flow from OHLCV klines."""
    features = {
        "tf_buy_ratio": 0.5,
        "tf_buy_volume": 0.0,
        "tf_sell_volume": 0.0,
        "tf_large_trades": 0.0,
        "tf_avg_size": 0.0,
        "tf_flow_imbalance": 0.0,
        "tf_aggression": 0.5,
        "tf_size_variance": 0.5,
    }

    if not klines or len(klines) < 3:
        return features

    recent = klines[-10:]
    total_vol = sum(k["v"] for k in recent)
    if total_vol <= 0:
        return features

    # Estimate buy/sell volume from candle direction
    buy_vol = 0.0
    sell_vol = 0.0
    for k in recent:
        vol = k["v"]
        body = k["c"] - k["o"]
        rng = k["h"] - k["l"]
        if rng > 0:
            # Body ratio estimates buy pressure
            buy_ratio = 0.5 + body / rng * 0.5
            buy_vol += vol * buy_ratio
            sell_vol += vol * (1 - buy_ratio)

    features["tf_buy_volume"] = buy_vol
    features["tf_sell_volume"] = sell_vol
    if total_vol > 0:
        features["tf_buy_ratio"] = buy_vol / total_vol
        features["tf_flow_imbalance"] = (buy_vol - sell_vol) / total_vol

    features["tf_avg_size"] = total_vol / len(recent) / 10  # ~10 trades per candle
    features["tf_aggression"] = features["tf_buy_ratio"]  # proxy

    return features


# ═══════════════════════════════════════════════════════════════════
# Main: Expand existing snapshots with OB + trade features
# ═══════════════════════════════════════════════════════════════════

def expand_snapshots(input_path, output_path, asset="btc", use_real_ob=False):
    """
    Read expanded_snapshots.jsonl and add order book + trade features.
    
    If use_real_ob=True: fetch live OB from Binance (for live bot / real-time collection).
    If use_real_ob=False: estimate from klines (for historical backfill).
    """
    print("=" * 60)
    print("  Expand snapshots with Order Book + Trade Flow features")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Asset: {asset}")
    print(f"  Real OB: {use_real_ob}")
    print("=" * 60)

    if not os.path.exists(input_path):
        print(f"[ERROR] Input file not found: {input_path}")
        return

    # Read existing data
    records = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records")

    if not records:
        print("[ERROR] No records found")
        return

    # Show sample
    sample = records[0]
    print(f"Sample keys: {list(sample.keys())[:15]}...")

    # Fetch klines for estimation
    print("\nFetching Binance klines for estimation...")
    symbol = f"{asset.upper()}USDT"
    klines = fetch_klines(symbol, limit=50)
    print(f"Got {len(klines)} klines")

    # Optionally fetch real OB
    real_ob = None
    real_trades = None
    if use_real_ob:
        print("\nFetching real order book...")
        bids, asks = fetch_order_book(symbol)
        if bids and asks:
            real_ob = compute_ob_features(bids, asks)
            print(f"  OB: imbalance={real_ob['ob_imbalance']:.3f}, spread={real_ob['ob_spread_bps']:.1f}bps")

        print("Fetching recent trades...")
        trades = fetch_recent_trades(symbol)
        if trades:
            real_trades = compute_trade_features(trades)
            print(f"  Trades: buy_ratio={real_trades['tf_buy_ratio']:.3f}, flow_imb={real_trades['tf_flow_imbalance']:.3f}")

    # Process each record
    print(f"\nProcessing {len(records)} records...")
    for i, rec in enumerate(records):
        # Get klines for this record (use fetched klines as proxy)
        rec_klines = rec.get("klines", klines)

        # Order book features
        if real_ob:
            # Use real OB (same for all records in this batch)
            for k, v in real_ob.items():
                rec[f"ob_{k.replace('ob_', '')}"] = v
        elif rec_klines:
            # Estimate from klines
            ob_feat = estimate_ob_from_klines(rec_klines)
            for k, v in ob_feat.items():
                rec[k] = v
        else:
            # Default zeros
            for k in ["ob_imbalance", "ob_spread", "ob_spread_bps",
                      "ob_bid_depth_5", "ob_ask_depth_5",
                      "ob_bid_depth_10", "ob_ask_depth_10",
                      "ob_bid_depth_20", "ob_ask_depth_20",
                      "ob_depth_imbalance_5", "ob_depth_imbalance_20",
                      "ob_wall_bid", "ob_wall_ask",
                      "ob_slope_bid", "ob_slope_ask"]:
                rec[k] = 0.0

        # Trade flow features
        if real_trades:
            for k, v in real_trades.items():
                rec[k] = v
        elif rec_klines:
            tf_feat = estimate_trades_from_klines(rec_klines)
            for k, v in tf_feat.items():
                rec[k] = v
        else:
            for k in ["tf_buy_ratio", "tf_buy_volume", "tf_sell_volume",
                      "tf_large_trades", "tf_avg_size", "tf_flow_imbalance",
                      "tf_aggression", "tf_size_variance"]:
                rec[k] = 0.0

        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1}/{len(records)}")

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone! Wrote {len(records)} records to {output_path}")

    # Print feature statistics
    ob_keys = [k for k in records[0].keys() if k.startswith("ob_")]
    tf_keys = [k for k in records[0].keys() if k.startswith("tf_")]
    print(f"\nOrder book features ({len(ob_keys)}): {ob_keys}")
    print(f"Trade flow features ({len(tf_keys)}): {tf_keys}")

    # Stats
    for key in ob_keys + tf_keys:
        vals = [r.get(key, 0) for r in records]
        non_zero = [v for v in vals if v != 0]
        print(f"  {key}: mean={sum(vals)/len(vals):.4f}, "
              f"nonzero={len(non_zero)/len(vals)*100:.1f}%, "
              f"min={min(vals):.4f}, max={max(vals):.4f}")


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else str(
        WORKSPACE / "data_collector" / "data" / "expanded" / "expanded_snapshots.jsonl"
    )
    output_file = sys.argv[2] if len(sys.argv) > 2 else str(
        OUTPUT_DIR / "expanded_snapshots_v2.jsonl"
    )
    asset = sys.argv[3] if len(sys.argv) > 3 else "btc"
    use_real = "--real-ob" in sys.argv

    expand_snapshots(input_file, output_file, asset, use_real)
