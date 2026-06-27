#!/usr/bin/env python3
"""
Convert legacy expanded_snapshots.jsonl to flat format with ta_ fields,
then add order book + trade flow features.

Input:  legacy format (markets + binance nested) with TA inside binance
Output: flat format (up_price, down_price, ta_*, ob_*, tf_*)

Usage:
    python3 convert_and_expand.py [--real-ob]
"""

import json
import os
import sys
import math
import time
import numpy as np
import requests
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
INPUT_FILE = WORKSPACE / "data_collector" / "data" / "expanded" / "expanded_snapshots.jsonl"
OUTPUT_FILE = WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v2.jsonl"

BINANCE_DEPTH_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"
BINANCE_TRADES_URL = "https://api.binance.com/api/v3/trades?symbol={symbol}&limit=100"

ASSET_MAP = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol"}


def fetch_order_book(symbol="BTCUSDT"):
    """Fetch order book from Binance."""
    try:
        r = requests.get(BINANCE_DEPTH_URL.format(symbol=symbol), timeout=10)
        if r.status_code == 200:
            data = r.json()
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
            return bids, asks
    except:
        pass
    return [], []


def fetch_recent_trades(symbol="BTCUSDT"):
    """Fetch recent trades from Binance."""
    try:
        r = requests.get(BINANCE_TRADES_URL.format(symbol=symbol), timeout=10)
        if r.status_code == 200:
            return [{
                "price": float(t["price"]),
                "qty": float(t["qty"]),
                "is_buyer_maker": t.get("isBuyerMaker", False),
            } for t in r.json()]
    except:
        pass
    return []


def compute_ob_features(bids, asks):
    """Compute order book features."""
    features = {}
    if not bids or not asks:
        return {k: 0.0 for k in [
            "ob_imbalance", "ob_spread", "ob_spread_bps",
            "ob_bid_depth_5", "ob_ask_depth_5",
            "ob_bid_depth_10", "ob_ask_depth_10",
            "ob_bid_depth_20", "ob_ask_depth_20",
            "ob_depth_imbalance_5", "ob_depth_imbalance_20",
            "ob_wall_bid", "ob_wall_ask",
            "ob_slope_bid", "ob_slope_ask",
        ]}

    best_bid = bids[0][0]
    best_ask = asks[0][0]
    mid = (best_bid + best_ask) / 2.0
    if mid <= 0:
        return {k: 0.0 for k in [
            "ob_imbalance", "ob_spread", "ob_spread_bps",
            "ob_bid_depth_5", "ob_ask_depth_5",
            "ob_bid_depth_10", "ob_ask_depth_10",
            "ob_bid_depth_20", "ob_ask_depth_20",
            "ob_depth_imbalance_5", "ob_depth_imbalance_20",
            "ob_wall_bid", "ob_wall_ask",
            "ob_slope_bid", "ob_slope_ask",
        ]}

    spread = best_ask - best_bid
    features["ob_spread"] = spread / mid
    features["ob_spread_bps"] = spread / mid * 10000

    total_bid = sum(q for _, q in bids)
    total_ask = sum(q for _, q in asks)
    total = total_bid + total_ask
    features["ob_imbalance"] = (total_bid - total_ask) / total if total > 0 else 0.0

    for level in [5, 10, 20]:
        bv = sum(q for _, q in bids[:level])
        av = sum(q for _, q in asks[:level])
        features[f"ob_bid_depth_{level}"] = bv
        features[f"ob_ask_depth_{level}"] = av
        dt = bv + av
        features[f"ob_depth_imbalance_{level}"] = (bv - av) / dt if dt > 0 else 0.0

    # Walls
    if bids:
        avg_bid_size = total_bid / len(bids)
        features["ob_wall_bid"] = max((q / avg_bid_size - 1.0, 0.0) for _, q in bids[:5]) if avg_bid_size > 0 else 0.0
    if asks:
        avg_ask_size = total_ask / len(asks)
        features["ob_wall_ask"] = max((q / avg_ask_size - 1.0, 0.0) for _, q in asks[:5]) if avg_ask_size > 0 else 0.0

    # Slopes
    if len(bids) >= 5 and sum(q for _, q in bids[:1]) > 0:
        features["ob_slope_bid"] = sum(q for _, q in bids[:5]) / sum(q for _, q in bids[:1]) - 1.0
    if len(asks) >= 5 and sum(q for _, q in asks[:1]) > 0:
        features["ob_slope_ask"] = sum(q for _, q in asks[:5]) / sum(q for _, q in asks[:1]) - 1.0

    return features


def compute_trade_features(trades):
    """Compute trade flow features."""
    features = {}
    if not trades:
        return {k: 0.0 for k in [
            "tf_buy_ratio", "tf_buy_volume", "tf_sell_volume",
            "tf_large_trades", "tf_avg_size", "tf_flow_imbalance",
            "tf_aggression", "tf_size_variance",
        ]}

    total_vol = 0.0
    buy_vol = 0.0
    sizes = []
    aggressive = 0

    for t in trades:
        vol = t["price"] * t["qty"]
        total_vol += vol
        sizes.append(t["qty"])
        if t["is_buyer_maker"]:
            sell_vol_local = vol
        else:
            buy_vol += vol
            aggressive += 1

    sell_vol = total_vol - buy_vol
    features["tf_buy_ratio"] = buy_vol / total_vol if total_vol > 0 else 0.5
    features["tf_buy_volume"] = buy_vol
    features["tf_sell_volume"] = sell_vol
    features["tf_flow_imbalance"] = (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0

    if sizes:
        sorted_sizes = sorted(sizes)
        median_size = sorted_sizes[len(sorted_sizes) // 2]
        if median_size > 0:
            large = sum(1 for s in sizes if s > 2 * median_size)
            features["tf_large_trades"] = large / len(sizes)
        features["tf_avg_size"] = sum(sizes) / len(sizes)
        mean_size = features["tf_avg_size"]
        if mean_size > 0 and len(sizes) > 1:
            var = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            features["tf_size_variance"] = math.sqrt(var) / mean_size

    features["tf_aggression"] = aggressive / len(trades) if trades else 0.5

    return features


def estimate_ob_from_ta(bn_data):
    """Estimate order book features from TA indicators (no klines available)."""
    features = {}

    # Spread proxy from Bollinger width (wider BB = more spread)
    bb_width = bn_data.get("bb_width", 0.0)
    features["ob_spread"] = min(bb_width * 0.5, 0.05)  # proxy
    features["ob_spread_bps"] = features["ob_spread"] * 10000

    # Imbalance proxy from MACD histogram + momentum
    macd_hist = bn_data.get("macd_hist", 0.0)
    momentum_5 = bn_data.get("momentum_5", 0.0)
    # Positive MACD hist + positive momentum → more buying pressure
    features["ob_imbalance"] = np.clip(macd_hist * 10.0 + momentum_5 * 2.0, -1.0, 1.0)

    # Depth proxy from volume ratio (higher vol = deeper market)
    vol_ratio = bn_data.get("vol_ratio", 1.0)
    depth_factor = min(vol_ratio / 3.0, 1.0)  # normalize
    features["ob_bid_depth_5"] = depth_factor * 30.0  # estimated BTC
    features["ob_ask_depth_5"] = depth_factor * 30.0
    features["ob_bid_depth_10"] = depth_factor * 60.0
    features["ob_ask_depth_10"] = depth_factor * 60.0
    features["ob_bid_depth_20"] = depth_factor * 100.0
    features["ob_ask_depth_20"] = depth_factor * 100.0

    # Depth imbalance from RSI (overbought = more asks, oversold = more bids)
    rsi = bn_data.get("rsi", 0.5)
    depth_imb = (rsi - 0.5) * 0.2  # small bias based on RSI
    features["ob_depth_imbalance_5"] = np.clip(depth_imb, -0.5, 0.5)
    features["ob_depth_imbalance_20"] = np.clip(depth_imb * 0.5, -0.3, 0.3)

    # Walls from ATR (high ATR = less likely to have walls)
    atr_pct = bn_data.get("atr_pct", 0.01)
    wall_factor = max(0.0, 1.0 - atr_pct * 20.0)  # low ATR → walls more likely
    features["ob_wall_bid"] = wall_factor * 0.3 if momentum_5 < 0 else 0.0
    features["ob_wall_ask"] = wall_factor * 0.3 if momentum_5 > 0 else 0.0

    # Slope from momentum (stronger momentum = steeper slope)
    features["ob_slope_bid"] = np.clip(momentum_5 * 5.0, -1.0, 1.0) if momentum_5 < 0 else 0.0
    features["ob_slope_ask"] = np.clip(-momentum_5 * 5.0, -1.0, 1.0) if momentum_5 > 0 else 0.0

    return features


def estimate_tf_from_ta(bn_data):
    """Estimate trade flow features from TA indicators."""
    features = {}

    # Buy ratio from MACD + momentum
    macd_hist = bn_data.get("macd_hist", 0.0)
    momentum_5 = bn_data.get("momentum_5", 0.0)
    stoch_k = bn_data.get("stoch_k", 0.5)
    # Combine signals
    buy_signal = macd_hist * 5.0 + momentum_5 * 2.0 + (stoch_k - 0.5) * 0.5
    features["tf_buy_ratio"] = np.clip(0.5 + buy_signal * 0.3, 0.0, 1.0)

    features["tf_flow_imbalance"] = np.clip(buy_signal * 0.5, -1.0, 1.0)

    # Large trades from volume ratio
    vol_ratio = bn_data.get("vol_ratio", 1.0)
    features["tf_large_trades"] = min(max(0.0, vol_ratio - 1.5) * 0.3, 1.0)

    # Average size from ATR + volume
    atr_pct = bn_data.get("atr_pct", 0.01)
    features["tf_avg_size"] = min(vol_ratio * atr_pct * 10.0, 1.0)

    features["tf_size_variance"] = min(vol_ratio * 0.3, 1.0)

    # Aggression from momentum + stoch
    features["tf_aggression"] = np.clip(0.5 + abs(momentum_5) * 5.0, 0.0, 1.0)

    # Buy/sell volumes
    features["tf_buy_volume"] = features["tf_buy_ratio"] * vol_ratio * 100.0
    features["tf_sell_volume"] = (1.0 - features["tf_buy_ratio"]) * vol_ratio * 100.0

    return features


def convert_snapshot(snap, ob_features, tf_features, use_estimated=False):
    """Convert legacy snapshot to flat format with all features."""
    import numpy as np

    # Extract market data
    markets = snap.get("markets", {})
    if not markets:
        return None

    market_key = list(markets.keys())[0]
    market_data = markets[market_key]

    # Extract binance data
    binance = snap.get("binance", {})
    bn_data = {}
    for sym, data in binance.items():
        if sym in ASSET_MAP:
            bn_data = data
            break

    # Build flat record
    record = {
        "timestamp": snap["timestamp"],
        "period_start": snap.get("period_start", snap["timestamp"]),
        "market_key": market_key,
        "up_price": market_data.get("up", 0.5),
        "down_price": market_data.get("down", 0.5),
        "binance_price": bn_data.get("price", 0.0),
    }

    # TA fields (flatten from binance data)
    ta_fields = [
        "sma_5", "sma_10", "sma_20", "ema_5", "ema_10", "ema_12", "ema_26", "ema_50",
        "ma_cross_5_20", "ma_cross_10_20", "ma_cross_ema_12_26",
        "price_vs_sma20", "price_vs_ema50",
        "rsi", "macd_line", "macd_signal", "macd_hist",
        "bb_lower", "bb_middle", "bb_upper", "bb_width", "bb_pct_b",
        "atr", "atr_pct", "stoch_k", "stoch_d", "stoch_cross",
        "vol_ratio", "obv", "realized_vol",
        "momentum_5", "momentum_10",
    ]
    for field in ta_fields:
        record[f"ta_{field}"] = bn_data.get(field, 0.0)

    # Order book features (real or estimated from TA)
    if use_estimated:
        est_ob = estimate_ob_from_ta(bn_data)
        for k, v in est_ob.items():
            record[k] = v
    else:
        for k, v in ob_features.items():
            record[k] = v

    # Trade flow features (real or estimated from TA)
    if use_estimated:
        est_tf = estimate_tf_from_ta(bn_data)
        for k, v in est_tf.items():
            record[k] = v
    else:
        for k, v in tf_features.items():
            record[k] = v

    return record


def main():
    use_real_ob = "--real-ob" in sys.argv
    use_estimated = "--estimate-ob" in sys.argv or (not use_real_ob)

    print("=" * 60)
    print("  Convert legacy → flat format + Order Book + Trade Flow")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Real OB: {use_real_ob}")
    print(f"  Estimated OB from TA: {use_estimated}")
    print("=" * 60)

    if not INPUT_FILE.exists():
        print(f"[ERROR] Input not found: {INPUT_FILE}")
        return

    # Fetch real OB if requested
    ob_features = {}
    tf_features = {}
    if use_real_ob:
        print("\nFetching real order book for BTC...")
        bids, asks = fetch_order_book("BTCUSDT")
        if bids and asks:
            ob_features = compute_ob_features(bids, asks)
            print(f"  OB: imbalance={ob_features['ob_imbalance']:.3f}, "
                  f"spread={ob_features['ob_spread_bps']:.1f}bps")

        print("Fetching recent trades for BTC...")
        trades = fetch_recent_trades("BTCUSDT")
        if trades:
            tf_features = compute_trade_features(trades)
            print(f"  Trades: buy_ratio={tf_features['tf_buy_ratio']:.3f}")

    # Process records
    print(f"\nProcessing records...")
    count = 0
    skipped = 0

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE) as fin, open(OUTPUT_FILE, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                snap = json.loads(line)
                record = convert_snapshot(snap, ob_features, tf_features, use_estimated=use_estimated)
                if record:
                    fout.write(json.dumps(record) + "\n")
                    count += 1
                else:
                    skipped += 1

                if count % 10000 == 0 and count > 0:
                    print(f"  Processed {count} records...")
            except Exception as e:
                skipped += 1

    print(f"\nDone! Wrote {count} records, skipped {skipped}")
    print(f"Output: {OUTPUT_FILE}")

    # Stats
    if count > 0 and record:
        print(f"\nSample record keys: {sorted(list(record.keys()))}")
        ob_keys = [k for k in record.keys() if k.startswith("ob_")]
        tf_keys = [k for k in record.keys() if k.startswith("tf_")]
        ta_keys = [k for k in record.keys() if k.startswith("ta_")]
        print(f"  TA features: {len(ta_keys)}")
        print(f"  OB features: {len(ob_keys)}")
        print(f"  TF features: {len(tf_keys)}")
        print(f"  Total features: {len(ta_keys) + len(ob_keys) + len(tf_keys) + 6}")

        # OB stats
        print(f"\nOB feature stats (first 1000 records):")
        for k in ob_keys[:5]:
            vals = []
            # Re-read first 1000 lines for stats
            with open(OUTPUT_FILE) as f:
                for i, line in enumerate(f):
                    if i >= 1000:
                        break
                    r = json.loads(line)
                    vals.append(r.get(k, 0))
            non_zero = [v for v in vals if v != 0]
            print(f"  {k}: mean={sum(vals)/len(vals):.4f}, nonzero={len(non_zero)/len(vals)*100:.1f}%")


if __name__ == "__main__":
    main()
