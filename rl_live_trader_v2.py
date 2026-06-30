#!/usr/bin/env python3
"""
RL Live Trader v2 — production-grade live trader for Polymarket.
Uses PPO v8 model (95 features) with real-time Order Book + Trade Flow from Binance.

Features (95):
  0-4: price (up, down, spread, momentum_5, momentum_full)
  5-9: volatility (short_vol, large_move, vol_accel, micro_trend, trend_reversal)
  10-13: cross-market (macd_hist, momentum_10, vol_ratio, rsi)
  14: time remaining
  15-17: position (has, side, unrealized_pnl)
  18-19: regime (trend_strength, volatility)
  20-34: ORDER BOOK (15 features from Binance)
  35-42: TRADE FLOW (8 features from Binance)
  43-74: TA (32 indicators from Binance klines)
  75-82: ML FORECAST (8 features — from pre-computed ML models)
  83-94: MULTI-HORIZON ML (12 features — h=1,3,5,10)

Risk Management:
  - Max position size: configurable (default 2% of capital)
  - Max drawdown: configurable (default 20%)
  - Cooldown after loss: configurable
  - Auto-close at end of period
  - Daily loss limit

Launch:
    python3 rl_live_trader_v2.py --model models/ppo_v8_btc_steps500000 --asset btc --hours 24
"""

import argparse
import json
import os
import sys
import time

# Force IPv4 globally — IPv6 connections hang on VPS
import socket
_orig_gai = socket.getaddrinfo
def _ipv4_gai(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_gai(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = _ipv4_gai

# Graceful shutdown on SIGTERM (systemd stop)
import signal
_running = True
def _sigterm_handler(signum, frame):
    global _running
    _running = False
signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)

# Force line buffering for log files
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)

# Global exception handler to log errors
import traceback as _tb

def _global_excepthook(exc_type, exc_value, exc_tb):
    msg = ''.join(_tb.format_exception(exc_type, exc_value, exc_tb))
    sys.stdout.write(f'EXCEPTION: {msg}\n')
    sys.stdout.flush()

sys.excepthook = _global_excepthook
import math
import signal
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from stable_baselines3 import PPO

# Trend-aware TP/SL
try:
    from trend_tpsl import check_tpsl, get_trend_tpsl
    HAS_TREND_TPSL = True
except ImportError:
    HAS_TREND_TPSL = False

try:
    from py_clob_client import ClobClient
    from py_clob_client.clob_types import OrderType
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    ClobClient = None
    OrderType = None

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

WALLET_ADDRESS = "0x2307F20EB8CAaaD5E83b9d2e326DA06cCC28B208"
PRIVATE_KEY = "68fe024167ad9e0ad41229d5f40c406114ffae87539e8b3accb3cb77ec8f9f91"
BUILDER_ADDRESS = "0xA136Fbd3B76a1304742370BddeCadad997837888"

CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
BINANCE_URL = "https://api.binance.com/api/v3"

N_FEATURES = 95
POLL_INTERVAL = 15       # seconds between ticks
MIN_HOLD_STEPS = 3       # minimum ticks before TP/SL can trigger
POSITION_SIZE_PCT = 0.02  # 2% of capital per trade (conservative)
TAKER_FEE = 0.025        # 2.5% taker fee
INITIAL_CAPITAL = 1000.0
PRICE_MIN = 0.15
PRICE_MAX = 0.85
COOLDOWN_TICKS = 3

# Risk Management
MAX_DRAWSOWN_PCT = 0.20     # 20% max drawdown → stop trading
DAILY_LOSS_LIMIT_PCT = 0.10  # 10% daily loss → stop for the day
MAX_POSITIONS_PER_PERIOD = 1  # max 1 position per period
TAKE_PROFIT_PCT = 0.15       # 15% take profit (default, overridden by trend-aware)
STOP_LOSS_PCT = 0.08         # 8% stop loss (default, overridden by trend-aware)
MAX_HOLD_STEPS = 40          # max steps before forced close
TRCT = 0.05     # 5%USE_TRAILING_STOP = True

# Telegram
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

running = True


def send_telegram(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"
        }, timeout=5)
    except:
        pass


def handle_signal(signum, frame):
    global running
    running = False
    print("\n[STOP] Shutdown signal received")


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor (95 features)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor95:
    """Extract 95 normalized features for PPO v8 model."""

    N_FEATURES = N_FEATURES

    def __init__(self, lookback=5):
        self.lookback = lookback
        self.price_history = []
        self.return_history = []

    def update(self, up_price, down_price, elapsed_sec=0,
               ob_data=None, tf_data=None, ta_data=None,
               ml_data=None, ml_multi_data=None):
        mid_price = float(up_price)
        self.price_history.append(mid_price)
        if len(self.price_history) > self.lookback + 1:
            self.price_history = self.price_history[-(self.lookback + 1):]

        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.lookback:
                self.return_history = self.return_history[-self.lookback:]

        f = np.zeros(self.N_FEATURES, dtype=np.float32)

        # === Price (0-4) ===
        f[0] = np.clip(up_price, 0.0, 1.0)
        f[1] = np.clip(down_price, 0.0, 1.0)
        f[2] = np.clip(up_price + down_price - 1.0, -0.1, 0.1) * 10.0
        if len(self.price_history) >= 6:
            f[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        if len(self.price_history) >= 2:
            f[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # === Volatility (5-9) ===
        if len(self.price_history) >= 6:
            f[5] = np.clip((self.price_history[-1] - self.price_history[-6]) * 20.0, -1.0, 1.0)
        if len(self.return_history) >= 2:
            f[6] = np.clip(abs(self.return_history[-1]) * 50.0, 0.0, 1.0)
        if len(self.return_history) >= 3:
            f[7] = 1.0 if abs(self.return_history[-1]) > 0.05 else 0.0
        if len(self.return_history) >= 3:
            f[8] = np.clip((self.return_history[-1] - self.return_history[-3]) * 50.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            f[9] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # === Cross-market (10-13) ===
        if ta_data:
            f[10] = np.clip(ta_data.get("macd_hist", 0.0), -1.0, 1.0)
            f[11] = np.clip(ta_data.get("momentum_10", 0.0) * 5.0, -1.0, 1.0)
            f[12] = np.clip(ta_data.get("vol_ratio", 1.0) - 1.0, -1.0, 1.0)
            f[13] = np.clip(ta_data.get("rsi", 0.5) - 0.5, -0.5, 0.5) * 2.0
        else:
            f[10] = f[11] = f[12] = f[13] = 0.0

        # === Time (14) ===
        remaining = max(0, 900 - elapsed_sec)
        f[14] = remaining / 900.0

        # === Position (15-17) — filled by trader ===
        # === Regime (18-19) ===
        if len(self.price_history) >= 5:
            total_move = abs(self.price_history[-1] - self.price_history[-5])
            total_range = sum(abs(self.return_history[-i]) for i in range(min(5, len(self.return_history))))
            if total_range > 0:
                f[18] = np.clip(total_move / total_range * 2.0 - 1.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            f[19] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # === Order Book (20-34) ===
        if ob_data:
            f[20] = np.clip(ob_data.get("ob_imbalance", 0.0), -1.0, 1.0)
            f[21] = np.clip(ob_data.get("ob_spread_bps", 2.0) / 10.0, 0.0, 1.0)
            f[22] = np.clip(ob_data.get("ob_bid_depth_5", 0.0) / 100.0, 0.0, 1.0)
            f[23] = np.clip(ob_data.get("ob_ask_depth_5", 0.0) / 100.0, 0.0, 1.0)
            f[24] = np.clip(ob_data.get("ob_bid_depth_20", 0.0) / 500.0, 0.0, 1.0)
            f[25] = np.clip(ob_data.get("ob_ask_depth_20", 0.0) / 500.0, 0.0, 1.0)
            f[26] = np.clip(ob_data.get("ob_depth_imbalance_5", 0.0), -1.0, 1.0)
            f[27] = np.clip(ob_data.get("ob_depth_imbalance_20", 0.0), -1.0, 1.0)
            f[28] = np.clip(ob_data.get("ob_wall_bid", 0.0), 0.0, 1.0)
            f[29] = np.clip(ob_data.get("ob_wall_ask", 0.0), 0.0, 1.0)
            f[30] = np.clip(ob_data.get("ob_slope_bid", 0.0), -1.0, 1.0)
            f[31] = np.clip(ob_data.get("ob_slope_ask", 0.0), -1.0, 1.0)
            f[32] = np.clip(ob_data.get("ob_spread", 0.0) * 20.0, 0.0, 1.0)
            f[33] = np.clip(ob_data.get("ob_bid_depth_10", 0.0) / 200.0, 0.0, 1.0)
            f[34] = np.clip(ob_data.get("ob_ask_depth_10", 0.0) / 200.0, 0.0, 1.0)

        # === Trade Flow (35-42) ===
        if tf_data:
            f[35] = np.clip(tf_data.get("tf_buy_ratio", 0.5), 0.0, 1.0)
            f[36] = np.clip(tf_data.get("tf_flow_imbalance", 0.0), -1.0, 1.0)
            f[37] = np.clip(tf_data.get("tf_large_trades", 0.0), 0.0, 1.0)
            f[38] = np.clip(tf_data.get("tf_avg_size", 0.0) * 10.0, 0.0, 1.0)
            f[39] = np.clip(tf_data.get("tf_size_variance", 0.5), 0.0, 1.0)
            f[40] = np.clip(tf_data.get("tf_aggression", 0.5), 0.0, 1.0)
            f[41] = np.clip(tf_data.get("tf_buy_volume", 0.0) / 1e6, 0.0, 1.0)
            f[42] = np.clip(tf_data.get("tf_sell_volume", 0.0) / 1e6, 0.0, 1.0)

        # === TA (43-74) ===
        if ta_data:
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
            for i, field in enumerate(ta_fields):
                val = ta_data.get(field, 0.0)
                f[43 + i] = np.clip(float(val), -1.0, 1.0)

        # === ML Forecast (75-82) ===
        if ml_data:
            f[75] = np.clip(ml_data.get("ml_prob_up", 0.5), 0.0, 1.0)
            f[76] = np.clip(ml_data.get("ml_prob_down", 0.5), 0.0, 1.0)
            f[77] = np.clip(ml_data.get("ml_confidence", 0.0), 0.0, 1.0)
            f[78] = np.clip(ml_data.get("ml_prediction", 0.5), 0.0, 1.0)
            f[79] = np.clip(ml_data.get("ml_edge", 0.0), -1.0, 1.0)
            f[80] = np.clip(ml_data.get("ml_signal_strength", 0.0), -1.0, 1.0)
            f[81] = np.clip(ml_data.get("ml_raw_xgb", 0.5), 0.0, 1.0)
            f[82] = np.clip(ml_data.get("ml_raw_lgb", 0.5), 0.0, 1.0)

        # === Multi-Horizon ML (83-94) ===
        if ml_multi_data:
            for h, base in [(1, 83), (3, 86), (5, 89), (10, 92)]:
                f[base] = np.clip(ml_multi_data.get(f"ml_h{h}_prob_up", 0.5), 0.0, 1.0)
                f[base + 1] = np.clip(ml_multi_data.get(f"ml_h{h}_confidence", 0.0), 0.0, 1.0)
                f[base + 2] = np.clip(ml_multi_data.get(f"ml_h{h}_edge", 0.0), -1.0, 1.0)

        return f

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data Fetcher (with OB + TF)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDataFetcherV2:
    """Fetches market data from Gamma API, Binance (klines + OB + TF)."""

    def __init__(self, asset="btc"):
        self.asset = asset
        self.symbol = f"{asset.upper()}USDT"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "RLBot/2.0"})
        self._ob_cache = None
        self._ob_cache_time = 0
        self._tf_cache = None
        self._tf_cache_time = 0

    def get_period_data(self, period):
        slug = f"{self.asset}-updown-15m-{period}"
        try:
            url = f"{GAMMA_API_URL}/markets/slug/{slug}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                prices = data.get("outcomePrices", "[]")
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if isinstance(prices, list) and len(prices) >= 2:
                    clob_ids = data.get("clobTokenIds", "[]")
                    if isinstance(clob_ids, str):
                        clob_ids = json.loads(clob_ids)
                    result = {
                        "slug": slug,
                        "period_start": period,
                        "up_price": float(prices[0]),
                        "down_price": float(prices[1]),
                        "condition_id": data.get("conditionId", ""),
                    }
                    if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                        result["up_token_id"] = clob_ids[0]
                        result["down_token_id"] = clob_ids[1]
                    return result
        except:
            pass
        return None

    def get_current_period(self):
        now = int(time.time())
        current = (now // 900) * 900
        next_period = current + 900
        results = {}
        for period in [current, next_period]:
            data = self.get_period_data(period)
            if data:
                results[period] = data
        return results

    def get_binance_klines(self, interval="5m", limit=50):
        try:
            url = f"{BINANCE_URL}/klines?symbol={self.symbol}&interval={interval}&limit={limit}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return [{"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                         "c": float(k[4]), "v": float(k[5])} for k in resp.json()]
        except:
            pass
        return []

    def get_binance_price(self):
        try:
            url = f"{BINANCE_URL}/ticker/price?symbol={self.symbol}"
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                return float(resp.json().get("price", 0))
        except:
            pass
        return 0.0

    def get_order_book(self):
        """Fetch order book from Binance."""
        now = time.time()
        if self._ob_cache and (now - self._ob_cache_time) < 5:
            return self._ob_cache
        try:
            r = self.session.get(f"{BINANCE_URL}/depth?symbol={self.symbol}&limit=20", timeout=10)
            if r.status_code == 200:
                data = r.json()
                bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
                asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
                features = self._compute_ob(bids, asks)
                self._ob_cache = features
                self._ob_cache_time = now
                return features
        except:
            pass
        return self._default_ob()

    def get_trade_flow(self):
        """Fetch trade flow from Binance."""
        now = time.time()
        if self._tf_cache and (now - self._tf_cache_time) < 5:
            return self._tf_cache
        try:
            r = self.session.get(f"{BINANCE_URL}/trades?symbol={self.symbol}&limit=100", timeout=10)
            if r.status_code == 200:
                trades = [{"price": float(t["price"]), "qty": float(t["qty"]),
                           "is_buyer_maker": t.get("isBuyerMaker", False)} for t in r.json()]
                features = self._compute_tf(trades)
                self._tf_cache = features
                self._tf_cache_time = now
                return features
        except:
            pass
        return self._default_tf()

    def compute_ta(self, closes, volumes=None):
        n = len(closes)
        if n < 30:
            return {}
        last = closes[-1]
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
            ema12 = _ema(data, 12)
            ema26 = _ema(data, 26)
            ml = ema12 - ema26
            ms = [_ema(data[:i], 12) - _ema(data[:i], 26) for i in range(26, len(data) + 1)]
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
            "sma_5": sma_5 / last, "sma_10": sma_10 / last, "sma_20": sma_20 / last,
            "ema_5": ema_5 / last, "ema_10": ema_10 / last,
            "ema_12": ema_12 / last, "ema_26": ema_26 / last, "ema_50": ema_50 / last,
            "ma_cross_5_20": (sma_5 - sma_20) / last,
            "ma_cross_10_20": (sma_10 - sma_20) / last,
            "ma_cross_ema_12_26": (ema_12 - ema_26) / last,
            "price_vs_sma20": (last - sma_20) / last,
            "price_vs_ema50": (last - ema_50) / last,
            "rsi": rsi / 100.0,
            "macd_line": ml / last, "macd_signal": sl / last, "macd_hist": hist / last,
            "bb_lower": bbl / last, "bb_middle": bbm / last, "bb_upper": bbu / last,
            "bb_width": (bbu - bbl) / bbm if bbm > 0 else 0.0,
            "bb_pct_b": (closes[-1] - bbl) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5,
            "atr_pct": bbs / last if last > 0 else 0.0,
            "stoch_k": 0.5, "stoch_d": 0.5, "stoch_cross": 0.0,
            "vol_ratio": min(vol_ratio, 5.0),
            "obv": 0.0, "realized_vol": 0.0,
            "momentum_5": mom_5, "momentum_10": mom_10,
        }

    def _compute_ob(self, bids, asks):
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
        for level in [5, 10, 20]:
            bv = sum(q for _, q in bids[:level])
            av = sum(q for _, q in asks[:level])
            result[f"ob_bid_depth_{level}"] = bv
            result[f"ob_ask_depth_{level}"] = av
            dt = bv + av
            result[f"ob_depth_imbalance_{level}"] = (bv - av) / dt if dt > 0 else 0.0
        if bids:
            avg_bid_size = total_bid / len(bids)
            if avg_bid_size > 0:
                wall_bid = max([q / avg_bid_size - 1.0 for _, q in bids[:5]] + [0.0])
                result["ob_wall_bid"] = float(wall_bid)
        if asks:
            avg_ask_size = total_ask / len(asks)
            if avg_ask_size > 0:
                wall_ask = max([q / avg_ask_size - 1.0 for _, q in asks[:5]] + [0.0])
                result["ob_wall_ask"] = float(wall_ask)
        if len(bids) >= 5 and sum(q for _, q in bids[:1]) > 0:
            result["ob_slope_bid"] = sum(q for _, q in bids[:5]) / sum(q for _, q in bids[:1]) - 1.0
        if len(asks) >= 5 and sum(q for _, q in asks[:1]) > 0:
            result["ob_slope_ask"] = sum(q for _, q in asks[:5]) / sum(q for _, q in asks[:1]) - 1.0
        return result

    def _compute_tf(self, trades):
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
                pass
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

    def _default_ob(self):
        return {k: 0.0 for k in [
            "ob_imbalance", "ob_spread", "ob_spread_bps",
            "ob_bid_depth_5", "ob_ask_depth_5", "ob_bid_depth_10", "ob_ask_depth_10",
            "ob_bid_depth_20", "ob_ask_depth_20",
            "ob_depth_imbalance_5", "ob_depth_imbalance_20",
            "ob_wall_bid", "ob_wall_ask", "ob_slope_bid", "ob_slope_ask",
        ]}

    def _default_tf(self):
        return {
            "tf_buy_ratio": 0.5, "tf_buy_volume": 0.0, "tf_sell_volume": 0.0,
            "tf_large_trades": 0.0, "tf_avg_size": 0.0,
            "tf_flow_imbalance": 0.0, "tf_aggression": 0.5, "tf_size_variance": 0.5,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TradeLogger:
    def __init__(self, log_dir="trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trades = []

    def log_trade(self, trade):
        self.trades.append(trade)
        date_str = datetime.fromtimestamp(trade["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
        log_file = self.log_dir / f"trades_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def get_total_stats(self):
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        losses = total_trades - wins
        total_pnl = sum(t["pnl"] for t in self.trades)
        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "avg_pnl": total_pnl / total_trades if total_trades > 0 else 0,
            "avg_win": np.mean([t["pnl"] for t in self.trades if t["pnl"] > 0]) if wins > 0 else 0,
            "avg_loss": np.mean([t["pnl"] for t in self.trades if t["pnl"] <= 0]) if losses > 0 else 0,
            "max_win": max((t["pnl"] for t in self.trades), default=0),
            "max_loss": min((t["pnl"] for t in self.trades), default=0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Manager
# ═══════════════════════════════════════════════════════════════════════════════

class RiskManager:
    """Risk management: drawdown limits, daily loss limits, position sizing."""

    def __init__(self, initial_capital, max_drawdown_pct=MAX_DRAWSOWN_PCT,
                 daily_loss_limit_pct=DAILY_LOSS_LIMIT_PCT):
        self.initial_capital = initial_capital
        self.peak_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.daily_pnl = 0.0
        self.current_day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.trading_halted = False
        self.halt_reason = ""

    def update(self, capital):
        self.peak_capital = max(self.peak_capital, capital)
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day != self.current_day:
            self.current_day = day
            self.daily_pnl = 0.0
            self.trading_halted = False
            self.halt_reason = ""

        drawdown = (self.peak_capital - capital) / self.peak_capital
        if drawdown >= self.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown reached: {drawdown*100:.1f}%"

    def can_trade(self):
        return not self.trading_halted

    def get_position_size(self, capital, base_pct=POSITION_SIZE_PCT):
        """Dynamic position size — reduce after losses."""
        drawdown = (self.peak_capital - capital) / self.peak_capital
        if drawdown > 0.10:
            return capital * base_pct * 0.5  # half size after 10% DD
        return capital * base_pct

    def record_pnl(self, pnl):
        self.daily_pnl += pnl
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if day == self.current_day:
            daily_limit = self.initial_capital * self.daily_loss_limit_pct
            if self.daily_pnl < -daily_limit:
                self.trading_halted = True
                self.halt_reason = f"Daily loss limit: ${self.daily_pnl:.2f}"


# ═══════════════════════════════════════════════════════════════════════════════
# CLOB Client Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ClobManager:
    def __init__(self, private_key, wallet_address, builder_address, dry_run=True):
        self.dry_run = dry_run
        self.client = None
        if not HAS_CLOB:
            print("[WARN] py_clob_client not installed!")
            return
        try:
            self.client = ClobClient(
                CLOB_API_URL, key=private_key, chain_id=137,
                funder=builder_address, signature_type="poly1271",
            )
            print(f"[CLOB] Connected (dry_run={dry_run})")
        except Exception as e:
            print(f"[CLOB] Connection failed: {e}")

    def place_order(self, token_id, side, price, size_tokens):
        if self.dry_run:
            print(f"  [DRY RUN] {side} {size_tokens:.2f} tokens @ {price:.4f}")
            return {"status": "dry_run"}
        if not self.client:
            return None
        try:
            clob_side = "BUY" if side == "BUY" else "SELL"
            order = self.client.create_order(
                token_id=token_id, side=clob_side,
                price=int(round(price * 10000)) / 10000,
                size=int(round(size_tokens * 100 * 100)),
                order_type=OrderType.GTC,
            )
            signed = self.client.sign(order)
            result = self.client.post_order(signed)
            return {"status": "ok", "order_id": str(result)}
        except Exception as e:
            print(f"  [CLOB] Order failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# RL Live Trader v2
# ═══════════════════════════════════════════════════════════════════════════════

class RLTraderV2:
    def __init__(self, model_path, asset="btc", initial_capital=INITIAL_CAPITAL,
                 position_size_pct=POSITION_SIZE_PCT, dry_run=True):
        self.asset = asset
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.dry_run = dry_run

        print(f"[Trader] Loading model: {model_path}")
        self.model = PPO.load(model_path)
        print(f"[Trader] Model loaded (expects {self.model.observation_space.shape[0]} features)")

        self.position = None
        self.current_step = 0
        self.peak_capital = initial_capital
        self.total_pnl = 0.0
        self.cooldown_counter = COOLDOWN_TICKS

        self.feature_extractor = FeatureExtractor95(lookback=5)
        self.market_fetcher = MarketDataFetcherV2(asset)
        self.trade_logger = TradeLogger()
        self.risk_manager = RiskManager(initial_capital)

        self.clob_manager = ClobManager(
            private_key=PRIVATE_KEY, wallet_address=WALLET_ADDRESS,
            builder_address=BUILDER_ADDRESS, dry_run=dry_run,
        )

        print(f"[Trader] Ready. Capital: ${self.capital:.2f}, Dry run: {self.dry_run}")

    def get_observation(self, period_data, elapsed_sec, ob_data, tf_data, ta_data):
        now = int(time.time())
        current_period = (now // 900) * 900
        period_info = period_data.get(current_period)
        if not period_info:
            period_info = list(period_data.values())[0] if period_data else None
            if not period_info:
                return None

        up_price = period_info.get("up_price", 0.5)
        down_price = period_info.get("down_price", 0.5)

        features = self.feature_extractor.update(
            up_price, down_price, elapsed_sec,
            ob_data=ob_data, tf_data=tf_data, ta_data=ta_data,
        )

        # Position features (15-17)
        if self.position is not None:
            features[15] = 1.0
            features[16] = float(self.position["side"])
            cp = up_price if self.position["side"] == 1 else down_price
            unrealized = (cp - self.position["entry_price"]) * self.position["shares"]
            features[17] = np.clip(unrealized / self.position["size_usd"], -1.0, 1.0)
        else:
            features[15] = 0.0
            features[16] = 0.0
            features[17] = 0.0

        return features

    def execute_action(self, action, period_data):
        now = int(time.time())
        current_period = (now // 900) * 900
        period_info = period_data.get(current_period)
        if not period_info:
            return None

        elapsed = now - current_period

        # Risk check
        self.risk_manager.update(self.capital)
        if not self.risk_manager.can_trade():
            if action in (1, 2):
                return None

        # Cooldown
        if self.cooldown_counter < COOLDOWN_TICKS and action in (1, 2):
            return None

        def get_price(side_str):
            return period_info.get("up_price", 0.5) if side_str == "UP" else period_info.get("down_price", 0.5)

        def get_token_id(side_str):
            return period_info.get("up_token_id", "") if side_str == "UP" else period_info.get("down_token_id", "")

        if action == 0:  # HOLD
            # Check TP/SL for open positions
            if self.position is not None:
                up_p = period_info.get("up_price", 0.5)
                down_p = period_info.get("down_price", 0.5)
                self._check_tp_sl(up_p, down_p)
            return None

        elif action == 3:  # SELL
            if self.position is not None:
                steps_held = self.current_step - self.position["entry_step"]
                if steps_held >= MIN_HOLD_STEPS:
                    side_str = "UP" if self.position["side"] == 1 else "DOWN"
                    exit_price = get_price(side_str)
                    return self._close_position(exit_price, current_period, elapsed)
            return None

        elif action == 1:  # BUY UP
            if self.position is not None:
                return None
            entry_price = get_price("UP")
            if entry_price < PRICE_MIN or entry_price > PRICE_MAX:
                return None
            size_usd = self.risk_manager.get_position_size(self.capital, self.position_size_pct)
            if size_usd < 1.0:
                return None
            shares = size_usd / entry_price
            fee = size_usd * TAKER_FEE
            token_id = get_token_id("UP")

            self.position = {
                "side": 1, "entry_price": entry_price, "size_usd": size_usd,
                "shares": shares, "entry_step": self.current_step,
                "period": current_period, "token_id": token_id,
            }
            self.capital -= fee
            self.clob_manager.place_order(token_id, "BUY", entry_price, shares)
            send_telegram(f"🟢 BUY UP @ {entry_price:.3f} | ${size_usd:.2f} | Cap: ${self.capital:.2f}")
            return {"timestamp": now, "action": "BUY_UP", "price": entry_price,
                    "size_usd": size_usd, "shares": shares, "capital": self.capital}

        elif action == 2:  # BUY DOWN
            if self.position is not None:
                return None
            entry_price = get_price("DOWN")
            if entry_price < PRICE_MIN or entry_price > PRICE_MAX:
                return None
            size_usd = self.risk_manager.get_position_size(self.capital, self.position_size_pct)
            if size_usd < 1.0:
                return None
            shares = size_usd / entry_price
            fee = size_usd * TAKER_FEE
            token_id = get_token_id("DOWN")

            self.position = {
                "side": -1, "entry_price": entry_price, "size_usd": size_usd,
                "shares": shares, "entry_step": self.current_step,
                "period": current_period, "token_id": token_id,
            }
            self.capital -= fee
            self.clob_manager.place_order(token_id, "BUY", entry_price, shares)
            send_telegram(f"🔴 BUY DOWN @ {entry_price:.3f} | ${size_usd:.2f} | Cap: ${self.capital:.2f}")
            return {"timestamp": now, "action": "BUY_DOWN", "price": entry_price,
                    "size_usd": size_usd, "shares": shares, "capital": self.capital}

        return None

    def _check_tp_sl(self, up_price, down_price):
        """Check take-profit / stop-loss with trend-aware logic."""
        if self.position is None:
            return
        
        pos = self.position
        side_str = "UP" if pos["side"] == 1 else "DOWN"
        current_price = up_price if pos["side"] == 1 else down_price
        entry_price = pos["entry_price"]
        steps_held = self.current_step - pos["entry_step"]
        
        # Calculate unrealized PnL %
        if entry_price > 0:
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            return
        
        # Get trend info from TA data if available
        adx = 0.0
        trend_regime = 0.0
        if hasattr(self, '_latest_ta') and self._latest_ta:
            # ADX and regime from precomputed data
            adx = self._latest_ta.get("adx", 0.0)
            trend_regime = self._latest_ta.get("trend_regime", 0.0)
        
        # Determine TP/SL thresholds
        if HAS_TREND_TPSL and adx > 0:
            # Trend-aware TP/SL
            peak_price = pos.get("peak_price", current_price)
            # Update peak
            if current_price > peak_price:
                self.position["peak_price"] = current_price
                peak_price = current_price
            
            trend_aligned = (
                (pos["side"] == 1 and trend_regime > 0) or
                (pos["side"] == -1 and trend_regime < 0)
            )
            params = get_trend_tpsl(adx, trend_regime, pos["side"], trend_aligned)
            tp_pct = params.take_profit_pct
            sl_pct = params.stop_loss_pct
            max_hold = params.max_hold_steps
        else:
            # Default static TP/SL (v8 behavior)
            tp_pct = TAKE_PROFIT_PCT
            sl_pct = STOP_LOSS_PCT
            max_hold = MAX_HOLD_STEPS
        
        # Check TP
        if pnl_pct >= tp_pct:
            elapsed = steps_held * POLL_INTERVAL
            reason = f"TP +{pnl_pct*100:.1f}%"
            send_telegram(f"🎯 {reason} | {side_str} | {entry_price:.3f}→{current_price:.3f}")
            self._close_position(current_price, pos.get("period", ""), elapsed)
            return
        
        # Check SL
        if pnl_pct <= -sl_pct:
            elapsed = steps_held * POLL_INTERVAL
            reason = f"SL -{abs(pnl_pct)*100:.1f}%"
            send_telegram(f"🛑 {reason} | {side_str} | {entry_price:.3f}→{current_price:.3f}")
            self._close_position(current_price, pos.get("period", ""), elapsed)
            return
        
        # Check max hold
        if steps_held >= max_hold:
            elapsed = steps_held * POLL_INTERVAL
            reason = f"MaxHold {steps_held} steps"
            send_telegram(f"⏰ {reason} | {side_str} | PnL {pnl_pct*100:+.1f}%")
            self._close_position(current_price, pos.get("period", ""), elapsed)
            return
        
        # Check trailing stop (if trend-aware with trailing)
        if HAS_TREND_TPSL and adx > 0:
            params = get_trend_tpsl(adx, trend_regime, pos["side"], 
                (pos["side"] == 1 and trend_regime > 0) or (pos["side"] == -1 and trend_regime < 0))
            if params.trailing:
                peak_price = pos.get("peak_price", entry_price)
                if peak_price > entry_price:  # trailing only after profit
                    trail_pct = params.trailing_pct
                    if pos["side"] == 1:
                        trail_level = peak_price * (1 - trail_pct)
                        if current_price < trail_level:
                            drop = (peak_price - current_price) / peak_price * 100
                            elapsed = steps_held * POLL_INTERVAL
                            send_telegram(f"📉 Trail -{drop:.1f}% | {side_str} | Peak {peak_price:.3f}→{current_price:.3f}")
                            self._close_position(current_price, pos.get("period", ""), elapsed)
                            return
                    else:
                        trail_level = peak_price * (1 + trail_pct)
                        if current_price > trail_level:
                            drop = (current_price - peak_price) / peak_price * 100
                            elapsed = steps_held * POLL_INTERVAL
                            send_telegram(f"📉 Trail -{drop:.1f}% | {side_str} | Peak {peak_price:.3f}→{current_price:.3f}")
                            self._close_position(current_price, pos.get("period", ""), elapsed)
                            return

    def _close_position(self, exit_price, period, elapsed):
        if self.position is None:
            return None
        pos = self.position
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        exit_fee = pos["size_usd"] * TAKER_FEE
        pnl -= exit_fee
        self.capital += pos["size_usd"] + pnl
        self.total_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)
        self.risk_manager.record_pnl(pnl)

        trade = {
            "timestamp": int(time.time()),
            "datetime": datetime.now(timezone.utc).isoformat(),
            "period": pos["period"],
            "side": "UP" if pos["side"] == 1 else "DOWN",
            "entry_price": float(pos["entry_price"]),
            "exit_price": float(exit_price),
            "shares": float(pos["shares"]),
            "size_usd": float(pos["size_usd"]),
            "pnl": float(pnl),
            "capital_after": float(self.capital),
            "elapsed_sec": elapsed,
            "steps_held": self.current_step - pos["entry_step"],
        }
        self.trade_logger.log_trade(trade)
        emoji = "✅" if pnl > 0 else "❌"
        send_telegram(f"{emoji} CLOSE {trade['side']} | P&L: ${pnl:.2f} | Cap: ${self.capital:.2f}")
        self.position = None
        self.cooldown_counter = 0
        return trade

    def run(self, duration_hours=24, poll_interval=POLL_INTERVAL):
        print(f"\n{'='*60}")
        print(f"  RL Live Trader v2 — {self.asset.upper()}")
        print(f"  Duration: {duration_hours}h | Poll: {poll_interval}s | Dry: {self.dry_run}")
        print(f"  Capital: ${self.initial_capital:.2f} | Max DD: {MAX_DRAWSOWN_PCT*100:.0f}%")
        print(f"{'='*60}\n")

        start_time = time.time()
        end_time = start_time + duration_hours * 3600
        iteration = 0

        global _running
        while _running:
            iteration += 1
            try:
                period_data = self.market_fetcher.get_current_period()
                if not period_data:
                    if iteration % 20 == 0:
                        print(f"[{iteration}] No period data | Cap: ${self.capital:.2f}")
                    time.sleep(poll_interval)
                    continue

                # Fetch real-time data
                klines = self.market_fetcher.get_binance_klines()
                ta_data = None
                if klines:
                    closes = [k["c"] for k in klines]
                    volumes = [k["v"] for k in klines]
                    ta_data = self.market_fetcher.compute_ta(closes, volumes)

                ob_data = self.market_fetcher.get_order_book()
                tf_data = self.market_fetcher.get_trade_flow()
                
                # Compute trend features for trend-aware TP/SL
                self._latest_ta = ta_data or {}
                if ta_data and HAS_TREND_TPSL and closes and len(closes) >= 10:
                    # Compute ADX approximation from closes
                    from trend_tpsl import compute_adx_simple
                    self._latest_ta['adx'] = compute_adx_simple(closes)
                    # Compute trend regime
                    import numpy as _np
                    recent = _np.array(closes[-10:])
                    x = _np.arange(len(recent))
                    slope = (_np.polyfit(x, recent, 1)[0] / _np.mean(recent)) * 100 if len(recent) > 0 else 0
                    if slope > 0.2:
                        self._latest_ta['trend_regime'] = 1.0
                    elif slope < -0.2:
                        self._latest_ta['trend_regime'] = -1.0
                    else:
                        self._latest_ta['trend_regime'] = 0.0

                now = int(time.time())
                current_period = (now // 900) * 900
                elapsed = now - current_period

                obs = self.get_observation(period_data, elapsed, ob_data, tf_data, ta_data)
                if obs is None:
                    time.sleep(poll_interval)
                    continue

                # Epsilon-greedy exploration
                epsilon = max(0.01, 0.2 * (0.995 ** self.current_step))  # Decay from 0.2 to 0.01
                if random.random() < epsilon:
                    action = random.choice([0, 1, 2, 3])  # Random action
                    self.last_epsilon_action = True
                else:
                    action, _ = self.model.predict(obs, deterministic=True)
                    self.last_epsilon_action = False
                
                result = self.execute_action(int(action), period_data)

                action_names = ["HOLD", "BUY_UP", "BUY_DOWN", "SELL"]
                pos_str = f"POS={'UP' if self.position and self.position['side']==1 else 'DOWN' if self.position else 'NONE'}"

                # Heartbeat every tick
                print(f"[{iteration}] {action_names[int(action)]} | {pos_str} | {elapsed}s | Cap: ${self.capital:.2f}", flush=True)

                if isinstance(result, dict):
                    if "action" in result:
                        print(f"  → {result['action']} @ {result['price']:.3f} | ${result['size_usd']:.2f}", flush=True)
                    elif "pnl" in result:
                        pnl_pct = result['pnl'] / result['size_usd'] * 100
                        emoji = "✅" if result['pnl'] > 0 else "❌"
                        print(f"  → {emoji} CLOSE {result['side']} | ${result['pnl']:.2f} ({pnl_pct:+.1f}%)", flush=True)

                # Period end → close
                if elapsed >= 895 and self.position is not None:
                    side_str = "UP" if self.position["side"] == 1 else "DOWN"
                    exit_price = None
                    for period, info in period_data.items():
                        if period == current_period:
                            exit_price = info.get("up_price", 0.5) if side_str == "UP" else info.get("down_price", 0.5)
                            break
                    if exit_price is None:
                        exit_price = 0.5
                    self._close_position(exit_price, current_period, elapsed)

                self.current_step += 1
                if self.cooldown_counter < COOLDOWN_TICKS:
                    self.cooldown_counter += 1

            except KeyboardInterrupt:
                break
            except SystemExit:
                break
            except Exception as e:
                print(f"[Trader] Error: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(poll_interval)

        # Final close
        if self.position is not None:
            print(f"[Trader] Closing final position...")
            side_str = "UP" if self.position["side"] == 1 else "DOWN"
            exit_price = 0.5
            try:
                period_data = self.market_fetcher.get_current_period()
                if period_data:
                    for period, info in period_data.items():
                        if period == current_period:
                            exit_price = info.get("up_price", 0.5) if side_str == "UP" else info.get("down_price", 0.5)
            except:
                pass
            self._close_position(exit_price, "final", 0)

        # Final close
        if self.position is not None:
            period_data = self.market_fetcher.get_current_period()
            for period, info in period_data.items():
                side_str = "UP" if self.position["side"] == 1 else "DOWN"
                exit_price = info.get("up_price", 0.5) if side_str == "UP" else info.get("down_price", 0.5)
                self._close_position(exit_price, self.position["period"], 900)
                break

        self._print_summary()

    def _print_summary(self):
        stats = self.trade_logger.get_total_stats()
        if stats["total_trades"] == 0:
            print("\n[Summary] No trades made.")
            return

        print(f"\n{'='*60}")
        print(f"  TRADE SUMMARY — {self.asset.upper()}")
        print(f"{'='*60}")
        print(f"  Total trades:     {stats['total_trades']}")
        print(f"  Wins/Losses:      {stats['wins']}/{stats['losses']}")
        print(f"  Win Rate:         {stats['win_rate']*100:.1f}%")
        print(f"  Total P&L:        ${stats['total_pnl']:.2f}")
        print(f"  Avg P&L:          ${stats['avg_pnl']:.2f}")
        print(f"  Avg Win:          ${stats['avg_win']:.2f}")
        print(f"  Avg Loss:         ${stats['avg_loss']:.2f}")
        print(f"  Final Capital:    ${self.capital:.2f}")
        print(f"  Total Return:     {(self.capital - self.initial_capital) / self.initial_capital * 100:.2f}%")
        print(f"  Peak Capital:     ${self.peak_capital:.2f}")
        print(f"{'='*60}")

        summary = {
            "asset": self.asset, **stats,
            "final_capital": self.capital,
            "peak_capital": self.peak_capital,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "timestamp": int(time.time()),
        }
        with open(f"trade_logs/summary_{self.asset}.json", "w") as f:
            json.dump(summary, f, indent=2)

        emoji = "📈" if stats["total_pnl"] > 0 else "📉"
        send_telegram(
            f"{emoji} <b>Summary — {self.asset.upper()}</b>\n"
            f"Trades: {stats['total_trades']} | WR: {stats['win_rate']*100:.1f}%\n"
            f"P&L: ${stats['total_pnl']:.2f} | Cap: ${self.capital:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Live Trader v2 — 95 features")
    parser.add_argument("--model", required=True, help="Path to PPO model (.zip)")
    parser.add_argument("--asset", default="btc", help="Asset to trade")
    parser.add_argument("--hours", type=float, default=24, help="Trading duration")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL, help="Poll interval")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode")
    parser.add_argument("--position-size", type=float, default=POSITION_SIZE_PCT, help="Position size fraction")
    parser.add_argument("--max-drawdown", type=float, default=MAX_DRAWSOWN_PCT, help="Max drawdown before halt")
    args = parser.parse_args()

    trader = RLTraderV2(
        model_path=args.model, asset=args.asset,
        initial_capital=args.capital, position_size_pct=args.position_size,
        dry_run=args.dry_run,
    )
    trader.risk_manager.max_drawdown_pct = args.max_drawdown
    trader.run(duration_hours=args.hours, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
