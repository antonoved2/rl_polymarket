#!/usr/bin/env python3
"""
Market Data Fetcher v5 — with Order Book + Trade Flow from Binance.

Provides real-time microstructure features for the RL bot.
"""

import json
import time
import math
import requests
from typing import Dict, Optional

BINANCE_URL = "https://api.binance.com/api/v3"


class MarketDataFetcherV5:
    """Fetch market data including order book and trade flow from Binance."""

    def __init__(self, asset: str = "btc"):
        self.asset = asset
        self.symbol = f"{asset.upper()}USDT"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RLBot/2.0"})
        # Cache for rate limiting
        self._ob_cache = None
        self._ob_cache_time = 0
        self._tf_cache = None
        self._tf_cache_time = 0

    def get_order_book(self, use_cache=True) -> Dict:
        """Fetch order book features from Binance."""
        now = time.time()
        if use_cache and self._ob_cache and (now - self._ob_cache_time) < 5:
            return self._ob_cache

        try:
            r = self.session.get(
                f"{BINANCE_URL}/depth?symbol={self.symbol}&limit=20",
                timeout=10
            )
            if r.status_code == 200:
                data = r.json()
                bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
                features = self._compute_ob(bids, asks)
                self._ob_cache = features
                self._ob_cache_time = now
                return features
        except Exception as e:
            pass
        return self._default_ob()

    def get_trade_flow(self, use_cache=True) -> Dict:
        """Fetch trade flow features from Binance."""
        now = time.time()
        if use_cache and self._tf_cache and (now - self._tf_cache_time) < 5:
            return self._tf_cache

        try:
            r = self.session.get(
                f"{BINANCE_URL}/trades?symbol={self.symbol}&limit=100",
                timeout=10
            )
            if r.status_code == 200:
                trades = [{
                    "price": float(t["price"]),
                    "qty": float(t["qty"]),
                    "is_buyer_maker": t.get("isBuyerMaker", False),
                } for t in r.json()]
                features = self._compute_tf(trades)
                self._tf_cache = features
                self._tf_cache_time = now
                return features
        except:
            pass
        return self._default_tf()

    def get_ta_features(self, interval="5m", limit=50) -> Dict:
        """Fetch klines and compute TA indicators."""
        try:
            r = self.session.get(
                f"{BINANCE_URL}/klines?symbol={self.symbol}&interval={interval}&limit={limit}",
                timeout=10
            )
            if r.status_code == 200:
                klines = [{
                    "o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                    "c": float(k[4]), "v": float(k[5])
                } for k in r.json()]
                closes = [k["c"] for k in klines]
                highs = [k["h"] for k in klines]
                lows = [k["l"] for k in klines]
                volumes = [k["v"] for k in klines]
                return self._compute_ta(closes, highs, lows, volumes)
        except:
            pass
        return {}

    def get_all_features(self) -> Dict:
        """Get all features (OB + TF + TA) in one call."""
        return {
            "ob": self.get_order_book(),
            "tf": self.get_trade_flow(),
            "ta": self.get_ta_features(),
        }

    # ═══════════════════════════════════════════════════════════════════
    # Internal computation
    # ═══════════════════════════════════════════════════════════════════

    def _compute_ob(self, bids, asks) -> Dict:
        """Compute order book features."""
        if not bids or not asks:
            return self._default_ob()

        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2.0
        if mid <= 0:
            return self._default_ob()

        spread = best_ask - best_bid
        total_bid = sum(q for _, q in bids)
        total_ask = sum(q for _, q in asks)
        total = total_bid + total_ask

        result = {
            "ob_imbalance": (total_bid - total_ask) / total if total > 0 else 0.0,
            "ob_spread": spread / mid,
            "ob_spread_bps": spread / mid * 10000,
        }

        # Depth at levels
        for level in [5, 10, 20]:
            bv = sum(q for _, q in bids[:level])
            av = sum(q for _, q in asks[:level])
            result[f"ob_bid_depth_{level}"] = bv
            result[f"ob_ask_depth_{level}"] = av
            dt = bv + av
            result[f"ob_depth_imbalance_{level}"] = (bv - av) / dt if dt > 0 else 0.0

        # Walls
        if bids:
            avg_bid_size = total_bid / len(bids)
            result["ob_wall_bid"] = max((q / avg_bid_size - 1.0, 0.0) for _, q in bids[:5]) if avg_bid_size > 0 else 0.0
        if asks:
            avg_ask_size = total_ask / len(asks)
            result["ob_wall_ask"] = max((q / avg_ask_size - 1.0, 0.0) for _, q in asks[:5]) if avg_ask_size > 0 else 0.0

        # Slopes
        if len(bids) >= 5 and sum(q for _, q in bids[:1]) > 0:
            result["ob_slope_bid"] = sum(q for _, q in bids[:5]) / sum(q for _, q in bids[:1]) - 1.0
        if len(asks) >= 5 and sum(q for _, q in asks[:1]) > 0:
            result["ob_slope_ask"] = sum(q for _, q in asks[:5]) / sum(q for _, q in asks[:1]) - 1.0

        return result

    def _compute_tf(self, trades) -> Dict:
        """Compute trade flow features."""
        if not trades:
            return self._default_tf()

        total_vol = 0.0
        buy_vol = 0.0
        sizes = []
        aggressive = 0

        for t in trades:
            vol = t["price"] * t["qty"]
            total_vol += vol
            sizes.append(t["qty"])
            if t["is_buyer_maker"]:
                pass  # sell
            else:
                buy_vol += vol
                aggressive += 1

        sell_vol = total_vol - buy_vol
        result = {
            "tf_buy_ratio": buy_vol / total_vol if total_vol > 0 else 0.5,
            "tf_buy_volume": buy_vol,
            "tf_sell_volume": sell_vol,
            "tf_flow_imbalance": (buy_vol - sell_vol) / total_vol if total_vol > 0 else 0.0,
        }

        if sizes:
            sorted_sizes = sorted(sizes)
            median_size = sorted_sizes[len(sorted_sizes) // 2]
            if median_size > 0:
                large = sum(1 for s in sizes if s > 2 * median_size)
                result["tf_large_trades"] = large / len(sizes)
            result["tf_avg_size"] = sum(sizes) / len(sizes)
            mean_size = result["tf_avg_size"]
            if mean_size > 0 and len(sizes) > 1:
                var = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
                result["tf_size_variance"] = math.sqrt(var) / mean_size

        result["tf_aggression"] = aggressive / len(trades) if trades else 0.5
        return result

    def _compute_ta(self, closes, highs=None, lows=None, volumes=None) -> Dict:
        """Compute TA indicators from klines."""
        n = len(closes)
        if n < 30:
            return {}

        last = closes[-1] if closes else 1.0
        if last <= 0:
            last = 1.0

        def _ema(data, period):
            if len(data) < period:
                return data[-1] if data else 0.0
            k = 2.0 / (period + 1)
            e = sum(data[:period]) / period
            for p in data[period:]:
                e = p * k + e * (1 - k)
            return e

        def _rsi(data, period=14):
            if len(data) < period + 1:
                return 50.0
            gains, losses = [], []
            for i in range(len(data) - period, len(data)):
                diff = data[i] - data[i-1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            ag = sum(gains) / period
            al = sum(losses) / period
            if al == 0:
                return 100.0
            return 100.0 - (100.0 / (1.0 + ag / al))

        def _macd(data):
            if len(data) < 35:
                return 0.0, 0.0, 0.0
            ema12 = _ema(data, 12)
            ema26 = _ema(data, 26)
            ml = ema12 - ema26
            ms = []
            for i in range(26, len(data) + 1):
                ms.append(_ema(data[:i], 12) - _ema(data[:i], 26))
            sl = _ema(ms, 9) if len(ms) >= 9 else ml
            return ml, sl, ml - sl

        def _bb(data, period=20):
            if len(data) < period:
                m = data[-1] if data else 0.0
                return m, m, m, 0.0
            s = data[-period:]
            m = sum(s) / period
            std = math.sqrt(sum((x - m)**2 for x in s) / period)
            return m - 2*std, m, m + 2*std, std

        sma_5 = sum(closes[-5:]) / 5 if n >= 5 else last
        sma_10 = sum(closes[-10:]) / 10 if n >= 10 else last
        sma_20 = sum(closes[-20:]) / 20 if n >= 20 else last
        ema_5 = _ema(closes, 5)
        ema_10 = _ema(closes, 10)
        ema_12 = _ema(closes, 12)
        ema_26 = _ema(closes, 26)
        ema_50 = _ema(closes, min(50, n))
        rsi = _rsi(closes)
        ml, sl, hist = _macd(closes)
        bbl, bbm, bbu, bbs = _bb(closes)

        vol_ratio = 1.0
        if volumes and len(volumes) >= 20:
            vs = sum(volumes[-20:]) / 20
            vol_ratio = volumes[-1] / vs if vs > 0 else 1.0

        mom_5 = (closes[-1] - closes[-6]) / closes[-6] if n >= 6 and closes[-6] > 0 else 0.0
        mom_10 = (closes[-1] - closes[-11]) / closes[-11] if n >= 11 and closes[-11] > 0 else 0.0

        return {
            "sma_5": sma_5, "sma_10": sma_10, "sma_20": sma_20,
            "ema_5": ema_5, "ema_10": ema_10, "ema_12": ema_12, "ema_26": ema_26, "ema_50": ema_50,
            "ma_cross_5_20": (sma_5 - sma_20) / last,
            "ma_cross_10_20": (sma_10 - sma_20) / last,
            "ma_cross_ema_12_26": (ema_12 - ema_26) / last,
            "price_vs_sma20": (last - sma_20) / last,
            "price_vs_ema50": (last - ema_50) / last,
            "rsi": rsi / 100.0,
            "macd_line": ml / last, "macd_signal": sl / last, "macd_hist": hist / last,
            "bb_lower": bbl, "bb_middle": bbm, "bb_upper": bbu,
            "bb_width": (bbu - bbl) / bbm if bbm > 0 else 0.0,
            "bb_pct_b": (closes[-1] - bbl) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5,
            "atr_pct": 0.0,
            "stoch_k": 0.5, "stoch_d": 0.5, "stoch_cross": 0.0,
            "vol_ratio": min(vol_ratio, 5.0),
            "obv": 0.0, "realized_vol": 0.0,
            "momentum_5": mom_5, "momentum_10": mom_10,
        }

    def _default_ob(self) -> Dict:
        return {k: 0.0 for k in [
            "ob_imbalance", "ob_spread", "ob_spread_bps",
            "ob_bid_depth_5", "ob_ask_depth_5",
            "ob_bid_depth_10", "ob_ask_depth_10",
            "ob_bid_depth_20", "ob_ask_depth_20",
            "ob_depth_imbalance_5", "ob_depth_imbalance_20",
            "ob_wall_bid", "ob_wall_ask",
            "ob_slope_bid", "ob_slope_ask",
        ]}

    def _default_tf(self) -> Dict:
        return {
            "tf_buy_ratio": 0.5, "tf_buy_volume": 0.0, "tf_sell_volume": 0.0,
            "tf_large_trades": 0.0, "tf_avg_size": 0.0,
            "tf_flow_imbalance": 0.0, "tf_aggression": 0.5,
            "tf_size_variance": 0.5,
        }
