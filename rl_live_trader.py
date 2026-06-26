#!/usr/bin/env python3
"""
RL Live Trader — production-grade live trader for Polymarket.

PPO model (45 features) → CLOB orders → statistics → retraining.

Cycle:
  1. Connect to Polymarket CLOB (Poly1271 auth)
  2. Load PPO model (45 features)
  3. For each 15-minute period:
     a. Collect data (Gamma API + Binance klines + TA indicators)
     b. Compute 45 features
     c. PPO decides: HOLD / BUY_UP / BUY_DOWN
     d. Place order via CLOB
     e. Manage position (MIN_HOLD_STEPS, TP/SL)
  4. Collect snapshots → save to expanded_snapshots.jsonl
  5. Periodic PPO retraining
  6. Full statistics: trades, wins, losses, P&L, win rate

Launch:
    python3 rl_live_trader.py --model models/ppo_v4_btc_steps150000 --asset btc --hours 24
"""

import argparse
import json
import os
import sys
import time
import math
import signal
import requests
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from py_clob_client import ClobClient
    from py_clob_client.clob_types import OrderType
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    print("[WARN] py_clob_client not installed!")
    ClobClient = None
    OrderType = None

try:
    from environment_v3 import PolymarketEnvV3, N_FEATURES
    HAS_ENV = True
except ImportError:
    HAS_ENV = False

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

WALLET_ADDRESS = "0x2307F20EB8CAaaD5E83b9d2e326DA06cCC28B208"
PRIVATE_KEY = "68fe024167ad9e0ad41229d5f40c406114ffae87539e8b3accb3cb77ec8f9f91"
BUILDER_ADDRESS = "0xA136Fbd3B76a1304742370BddeCadad997837888"

CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
BINANCE_URL = "https://api.binance.com/api/v3"

MODEL_DIR = "/home/antonov5/.openclaw/workspace/rl_polymarket/models"
DATA_PATH = "/opt/rl_trader/data/expanded_snapshots.jsonl"

POLL_INTERVAL = 15      # seconds between ticks (Gamma API updates slowly)
MIN_HOLD_STEPS = 3       # minimum ticks before TP/SL can trigger (3 * 15s = 45s)
POSITION_SIZE_PCT = 0.10  # 10% of capital per trade
TAKER_FEE = 0.025       # 2.5% taker fee (entry + exit)
INITIAL_CAPITAL = 1000.0
TAKE_PROFIT_PCT = 0.15  # 15% take profit
STOP_LOSS_PCT = 0.10    # 10% stop loss
PRICE_MIN = 0.15         # don't trade if price below this (avoid dead tokens)
PRICE_MAX = 0.85         # don't trade if price above this
COOLDOWN_TICKS = 5       # wait after closing before entering

# Telegram (optional)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

running = True


def send_telegram(message):
    """Send Telegram notification (silent fail if not configured)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}, timeout=5)
    except:
        pass


def handle_signal(signum, frame):
    global running
    running = False
    print("\n[STOP] Shutdown signal received")


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor (45 features, matching PPO v4)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Extracts 45 normalized features — compatible with PPO v4+ model."""

    N_FEATURES = 45

    def __init__(self, lookback=5):
        self.lookback = lookback
        self.price_history = []
        self.return_history = []

    def update(self, up_price, down_price, binance_price=0.0, binance_return_1m=0.0,
               binance_return_5m=0.0, volatility=0.0, elapsed_sec=0, period_start_ts=0,
               ta_data=None):
        mid_price = float(up_price)
        self.price_history.append(mid_price)
        max_len = self.lookback + 1
        if len(self.price_history) > max_len:
            self.price_history = self.price_history[-max_len:]

        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.lookback:
                self.return_history = self.return_history[-self.lookback:]

        features = np.zeros(self.N_FEATURES, dtype=np.float32)

        # === Price features (0-4) ===
        features[0] = np.clip(up_price, 0.0, 1.0)
        features[1] = np.clip(down_price, 0.0, 1.0)
        features[2] = np.clip(up_price + down_price - 1.0, -0.1, 0.1) * 10.0

        if len(self.price_history) >= 6:
            features[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        if len(self.price_history) >= 2:
            features[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # === Order Book features (5-9) ===
        base_spread = 0.005 + 0.02 * (1.0 - abs(up_price - 0.5) * 2)
        spread = min(base_spread, 0.05)
        features[5] = np.clip(spread * 20.0, 0.0, 1.0)

        if len(self.price_history) >= 6:
            features[6] = np.clip((self.price_history[-1] - self.price_history[-6]) * 20.0, -1.0, 1.0)
        if len(self.return_history) >= 2:
            features[7] = np.clip(abs(self.return_history[-1]) * 50.0, 0.0, 1.0)
        if len(self.return_history) >= 3:
            features[8] = 1.0 if abs(self.return_history[-1]) > 0.05 else 0.0
        if len(self.return_history) >= 3:
            features[9] = np.clip((self.return_history[-1] - self.return_history[-3]) * 50.0, -1.0, 1.0)

        # === Cross-market (10-13) ===
        features[10] = np.clip(binance_return_1m * 100.0, -1.0, 1.0)
        features[11] = np.clip(binance_return_5m * 100.0, -1.0, 1.0)
        if len(self.return_history) >= 3:
            features[12] = np.clip(np.std(self.return_history) * 100.0, 0.0, 1.0)
        else:
            features[12] = 0.0
        features[13] = np.clip(volatility * 100.0, 0.0, 1.0)

        # === Time (14) ===
        remaining = max(0, 900 - elapsed_sec)
        features[14] = remaining / 900.0

        # === Position (15-17) ===
        features[15] = 0.0  # has_position
        features[16] = 0.0  # position_side
        features[17] = 0.0  # unrealized_pnl_pct

        # === Regime (18-19) ===
        if len(self.price_history) >= 5:
            total_move = abs(self.price_history[-1] - self.price_history[-5])
            total_range = sum(abs(self.return_history[-i]) for i in range(min(5, len(self.return_history))))
            if total_range > 0:
                features[18] = np.clip(total_move / total_range * 2.0 - 1.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            features[19] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # === TA indicators (20-44) ===
        if ta_data:
            ta_fields = [
                "ma_cross_5_20", "ma_cross_10_20", "ma_cross_ema_12_26",
                "price_vs_sma20", "price_vs_ema50",
                "rsi", "macd_line", "macd_signal", "macd_hist",
                "bb_width", "bb_pct_b", "bb_upper", "bb_lower",
                "atr_pct", "stoch_k", "stoch_d",
                "vol_ratio", "obv",
                "momentum_5", "momentum_10",
                "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
            ]
            for i, field in enumerate(ta_fields):
                val = ta_data.get(field, 0.0)
                features[20 + i] = np.clip(float(val), -1.0, 1.0)

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# Market Data Fetcher
# ═══════════════════════════════════════════════════════════════════════════════

class MarketDataFetcher:
    """Fetches market data from Gamma API and Binance."""

    def __init__(self, asset="btc"):
        self.asset = asset
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (RLBot/2.0)"})

    def get_period_data(self, period):
        """Get data for a specific period."""
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
        except Exception as e:
            pass
        return None

    def get_current_period(self):
        """Get current and next period data."""
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
        """Get OHLCV candles from Binance for TA."""
        try:
            symbol = f"{self.asset.upper()}USDT"
            url = f"{BINANCE_URL}/klines?symbol={symbol}&interval={interval}&limit={limit}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                return [{"o": float(k[1]), "h": float(k[2]), "l": float(k[3]),
                         "c": float(k[4]), "v": float(k[5])} for k in resp.json()]
        except:
            pass
        return []

    def get_binance_price(self):
        """Get current price from Binance."""
        try:
            symbol = f"{self.asset.upper()}USDT"
            url = f"{BINANCE_URL}/ticker/price?symbol={symbol}"
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                return float(resp.json().get("price", 0))
        except:
            pass
        return 0.0

    def compute_ta(self, closes, volumes=None):
        """Compute TA indicators from close prices."""
        n = len(closes)
        if n < 30:
            return {}

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

        last = closes[-1] if closes else 1.0
        if last <= 0:
            last = 1.0

        sma_5 = sum(closes[-5:]) / 5 if n >= 5 else last
        sma_10 = sum(closes[-10:]) / 10 if n >= 10 else last
        sma_20 = sum(closes[-20:]) / 20 if n >= 20 else last
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
            "ma_cross_5_20": (sma_5 - sma_20) / last,
            "ma_cross_10_20": (sma_10 - sma_20) / last,
            "ma_cross_ema_12_26": (ema_12 - ema_26) / last,
            "price_vs_sma20": (last - sma_20) / last,
            "price_vs_ema50": (last - ema_50) / last,
            "rsi": rsi / 100.0,
            "macd_line": ml / last,
            "macd_signal": sl / last,
            "macd_hist": hist / last,
            "bb_width": (bbu - bbl) / bbm if bbm > 0 else 0.0,
            "bb_pct_b": (closes[-1] - bbl) / (bbu - bbl) if (bbu - bbl) > 0 else 0.5,
            "bb_upper": bbu,
            "bb_lower": bbl,
            "atr_pct": 0.0,
            "stoch_k": 0.5,
            "stoch_d": 0.5,
            "vol_ratio": min(vol_ratio, 5.0),
            "obv": 0.0,
            "momentum_5": mom_5,
            "momentum_10": mom_10,
            "sma_5": sma_5,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "ema_26": ema_26,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TradeLogger:
    """Full trade journal and statistics."""

    def __init__(self, log_dir="trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trades = []
        self.daily_stats = defaultdict(lambda: {
            "wins": 0, "losses": 0, "pnl": 0.0, "trades": 0,
            "total_buy": 0.0, "total_sell": 0.0,
        })

    def log_trade(self, trade):
        self.trades.append(trade)
        date_str = datetime.fromtimestamp(trade["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
        stats = self.daily_stats[date_str]
        stats["trades"] += 1
        stats["pnl"] += trade["pnl"]
        stats["total_buy"] += trade.get("size_usd", 0)
        if trade["pnl"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        log_file = self.log_dir / f"trades_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def log_snapshot(self, snapshot):
        with open(DATA_PATH, "a") as f:
            f.write(json.dumps(snapshot) + "\n")

    def get_total_stats(self):
        total_trades = len(self.trades)
        wins = sum(1 for t in self.trades if t["pnl"] > 0)
        losses = total_trades - wins
        total_pnl = sum(t["pnl"] for t in self.trades)
        total_buy = sum(t.get("size_usd", 0) for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        avg_win = np.mean([t["pnl"] for t in self.trades if t["pnl"] > 0]) if wins > 0 else 0
        avg_loss = np.mean([t["pnl"] for t in self.trades if t["pnl"] <= 0]) if losses > 0 else 0
        max_win = max((t["pnl"] for t in self.trades), default=0)
        max_loss = min((t["pnl"] for t in self.trades), default=0)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / total_trades if total_trades > 0 else 0,
            "total_pnl": total_pnl,
            "total_buy": total_buy,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# CLOB Client Manager
# ═══════════════════════════════════════════════════════════════════════════════

class ClobManager:
    """Manages Polymarket CLOB connection."""

    def __init__(self, private_key, wallet_address, builder_address, dry_run=True):
        self.private_key = private_key
        self.wallet_address = wallet_address
        self.builder_address = builder_address
        self.dry_run = dry_run
        self.client = None

        if not HAS_CLOB:
            raise ImportError("py_clob_client not installed!")

        self._connect()

    def _connect(self):
        try:
            self.client = ClobClient(
                CLOB_API_URL,
                key=self.private_key,
                chain_id=137,
                funder=self.builder_address,
                signature_type="poly1271",
            )
            print(f"[CLOB] Connected to {CLOB_API_URL}")
            print(f"[CLOB] Wallet: {self.wallet_address}")
            print(f"[CLOB] Builder: {self.builder_address}")
            print(f"[CLOB] Dry run: {self.dry_run}")
        except Exception as e:
            print(f"[CLOB] Connection failed: {e}")
            raise

    def place_order(self, token_id, side, price, size_tokens):
        if self.dry_run:
            print(f"  [DRY RUN] Would place order: {side} {size_tokens:.2f} tokens @ {price:.4f}")
            return {"status": "dry_run", "token_id": token_id, "side": side,
                    "price": price, "size": size_tokens}

        try:
            clob_side = "BUY" if side == "BUY" else "SELL"
            price_cents = int(round(price * 10000)) / 10000
            size_cents = int(round(size_tokens * 100 * 100))

            order = self.client.create_order(
                token_id=token_id,
                side=clob_side,
                price=price_cents,
                size=size_cents,
                order_type=OrderType.GTC,
            )
            signed = self.client.sign(order)
            result = self.client.post_order(signed)
            print(f"  [CLOB] Order placed: {side} {size_tokens:.2f} tokens @ {price:.4f}")
            return {"status": "ok", "order_id": str(result)}
        except Exception as e:
            print(f"  [CLOB] Order failed: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# RL Live Trader
# ═══════════════════════════════════════════════════════════════════════════════

class RLTrader:
    """Main trader class."""

    def __init__(self, model_path, asset="btc", initial_capital=INITIAL_CAPITAL,
                 position_size_pct=POSITION_SIZE_PCT, dry_run=True):
        self.asset = asset
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.dry_run = dry_run

        # Load model
        print(f"[Trader] Loading model: {model_path}")
        self.model = PPO.load(model_path)
        print(f"[Trader] Model loaded successfully")

        # State
        self.position = None  # {side, entry_price, shares, entry_step, token_id, size_usd}
        self.current_step = 0
        self.peak_capital = initial_capital
        self.total_pnl = 0.0
        self.cooldown_counter = COOLDOWN_TICKS  # no cooldown at start
        self.last_close_step = -999

        # Data
        self.feature_extractor = FeatureExtractor(lookback=5)
        self.market_fetcher = MarketDataFetcher(asset)
        self.trade_logger = TradeLogger()

        # CLOB
        print(f"[DEBUG] HAS_CLOB={HAS_CLOB}, ClobClient={ClobClient}")
        self.clob_manager = ClobManager(
            private_key=PRIVATE_KEY,
            wallet_address=WALLET_ADDRESS,
            builder_address=BUILDER_ADDRESS,
            dry_run=dry_run,
        )

        print(f"[Trader] Ready. Capital: ${self.capital:.2f}, Dry run: {self.dry_run}")

    def get_observation(self, period_data, binance_price, elapsed_sec, ta_data=None):
        now = int(time.time())
        current_period = (now // 900) * 900

        period_info = period_data.get(current_period)
        if not period_info:
            period_info = list(period_data.values())[0] if period_data else None
            if not period_info:
                return None

        up_price = period_info.get("up_price", 0.5)
        down_price = period_info.get("down_price", 0.5)

        binance_return_1m = 0.0
        binance_return_5m = 0.0
        volatility = 0.0

        features = self.feature_extractor.update(
            up_price, down_price, binance_price,
            binance_return_1m, binance_return_5m, volatility,
            elapsed_sec, current_period, ta_data=ta_data
        )

        # Add position features (indices 15-17)
        if self.position is not None:
            features[15] = 1.0
            features[16] = float(self.position["side"])
            current_price = up_price if self.position["side"] == 1 else down_price
            unrealized = (current_price - self.position["entry_price"]) * self.position["shares"]
            features[17] = np.clip(unrealized / self.position["size_usd"], -1.0, 1.0)

        return features

    def execute_action(self, action, period_data, market_prices):
        """Execute model action. Returns trade decision dict or None."""
        now = int(time.time())
        current_period = (now // 900) * 900
        period_info = period_data.get(current_period)
        if not period_info:
            return None

        elapsed = now - current_period

        def get_price(side_str):
            for (p, s), price in market_prices.items():
                if s == side_str:
                    return price
            if side_str == "UP":
                return period_info.get("up_price", 0.5)
            return period_info.get("down_price", 0.5)

        def get_token_id(side_str):
            if side_str == "UP":
                return period_info.get("up_token_id", "")
            return period_info.get("down_token_id", "")

        # Cooldown check — don't enter if recently closed
        if self.cooldown_counter < COOLDOWN_TICKS:
            if action in (1, 2):
                return None
            if action == 0:
                return None

        if action == 0:  # HOLD
            # TP/SL check is handled in the main loop (before model.predict)
            return None

        elif action == 1:  # BUY UP
            if self.position is None:
                size_usd = self.capital * self.position_size_pct
                if size_usd < 1.0:
                    return None
                entry_price = get_price("UP")
                if entry_price < PRICE_MIN or entry_price > PRICE_MAX:
                    return None
                shares = size_usd / entry_price
                fee = size_usd * TAKER_FEE

                token_id = get_token_id("UP")

                self.position = {
                    "side": 1,
                    "entry_price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "entry_step": self.current_step,
                    "period": current_period,
                    "token_id": token_id,
                }
                self.capital -= fee

                if token_id:
                    self.clob_manager.place_order(token_id, "BUY", entry_price, shares)

                send_telegram(f"🟢 BUY UP @ {entry_price:.3f} | Size: ${size_usd:.2f} | Capital: ${self.capital:.2f}")

                return {
                    "timestamp": now,
                    "action": "BUY_UP",
                    "price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "elapsed_sec": elapsed,
                    "capital": self.capital,
                }

        elif action == 2:  # BUY DOWN
            if self.position is None:
                size_usd = self.capital * self.position_size_pct
                if size_usd < 1.0:
                    return None
                entry_price = get_price("DOWN")
                if entry_price < PRICE_MIN or entry_price > PRICE_MAX:
                    return None
                shares = size_usd / entry_price
                fee = size_usd * TAKER_FEE

                token_id = get_token_id("DOWN")

                self.position = {
                    "side": -1,
                    "entry_price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "entry_step": self.current_step,
                    "period": current_period,
                    "token_id": token_id,
                }
                self.capital -= fee

                if token_id:
                    self.clob_manager.place_order(token_id, "BUY", entry_price, shares)

                send_telegram(f"🔴 BUY DOWN @ {entry_price:.3f} | Size: ${size_usd:.2f} | Capital: ${self.capital:.2f}")

                return {
                    "timestamp": now,
                    "action": "BUY_DOWN",
                    "price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "elapsed_sec": elapsed,
                    "capital": self.capital,
                }

        return None

    def _close_position(self, exit_price, period, elapsed):
        """Close current position. Returns trade dict."""
        if self.position is None:
            return None

        pos = self.position
        # PnL: for UP, profit when price goes up; for DOWN, profit when price goes down
        # exit_price is the CURRENT price of the token we hold
        # For UP token: if exit=1.0, profit = (1.0 - entry) * shares
        # For DOWN token: if exit=0.0 (DOWN won, so UP=0), we need DOWN price
        # Actually: we hold UP or DOWN token. At resolution:
        #   UP wins → UP token = $1, DOWN token = $0
        #   DOWN wins → UP token = $0, DOWN token = $1
        # But we exit EARLY at market price, so exit_price is current market price of our token
        pnl = (exit_price - pos["entry_price"]) * pos["shares"]
        exit_fee = pos["size_usd"] * TAKER_FEE
        pnl -= exit_fee

        self.capital += pos["size_usd"] + pnl
        self.total_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

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

        emoji = "✅" if trade["pnl"] > 0 else "❌"
        send_telegram(f"{emoji} CLOSE {trade['side']} | P&L: ${trade['pnl']:.2f} | Capital: ${self.capital:.2f}")

        self.position = None
        self.cooldown_counter = 0
        self.last_close_step = self.current_step
        return trade

    def run(self, duration_hours=24, poll_interval=POLL_INTERVAL):
        """Run the trader."""
        print(f"\n{'='*60}")
        print(f"  RL Live Trader — {self.asset.upper()}")
        print(f"  Duration: {duration_hours}h, Poll: {poll_interval}s")
        print(f"  Dry run: {self.dry_run}")
        print(f"  Initial capital: ${self.initial_capital:.2f}")
        print(f"{'='*60}\n")

        start_time = time.time()
        end_time = start_time + duration_hours * 3600
        iteration = 0
        period_trades = 0

        while time.time() < end_time and running:
            iteration += 1
            loop_start = time.time()

            try:
                # Fetch market data
                period_data = self.market_fetcher.get_current_period()
                if not period_data:
                    print(f"[{iteration}] No market data, sleeping...")
                    time.sleep(poll_interval)
                    continue

                binance_price = self.market_fetcher.get_binance_price()
                market_prices = {}
                for period, info in period_data.items():
                    market_prices[(period, "UP")] = info.get("up_price", 0.5)
                    market_prices[(period, "DOWN")] = info.get("down_price", 0.5)

                # Fetch Binance klines for TA
                klines = self.market_fetcher.get_binance_klines()
                ta_data = None
                if klines:
                    closes = [k["c"] for k in klines]
                    volumes = [k["v"] for k in klines]
                    ta_data = self.market_fetcher.compute_ta(closes, volumes)

                now = int(time.time())
                current_period = (now // 900) * 900
                elapsed = now - current_period

                # Get observation
                obs = self.get_observation(period_data, binance_price, elapsed, ta_data=ta_data)
                if obs is None:
                    time.sleep(poll_interval)
                    continue

                # TP/SL check — close position if hit (after min hold)
                if self.position is not None:
                    steps_held = self.current_step - self.position["entry_step"]
                    if steps_held >= MIN_HOLD_STEPS:
                        side_str = "UP" if self.position["side"] == 1 else "DOWN"
                        current_price = None
                        for (p, s), price in market_prices.items():
                            if s == side_str:
                                current_price = price
                                break
                        if current_price is not None and self.position["entry_price"] > 0:
                            pnl_pct = (current_price - self.position["entry_price"]) / self.position["entry_price"]
                            if pnl_pct >= TAKE_PROFIT_PCT or pnl_pct <= -STOP_LOSS_PCT:
                                print(f'[{iteration}] TP/SL triggered: {pnl_pct:.3f}', flush=True)
                                self._close_position(current_price, current_period, elapsed)
                                period_trades += 1

                # Model decision
                action, _ = self.model.predict(obs, deterministic=True)

                # Execute
                result = self.execute_action(int(action), period_data, market_prices)

                # Log
                action_names = ["HOLD", "BUY_UP", "BUY_DOWN"]
                pos_str = f"POS={'UP' if self.position and self.position['side']==1 else 'DOWN' if self.position else 'NONE'}"

                if isinstance(result, dict):
                    if "action" in result:
                        print(f"[{iteration}] {result['action']} @ {result['price']:.3f} | "
                              f"Size: ${result['size_usd']:.2f} | Capital: ${self.capital:.2f}")
                    elif "pnl" in result:
                        pnl_pct = result['pnl'] / result['size_usd'] * 100
                        emoji = "✅" if result['pnl'] > 0 else "❌"
                        print(f"[{iteration}] {emoji} CLOSE {result['side']} | "
                              f"P&L: ${result['pnl']:.2f} ({pnl_pct:+.1f}%) | "
                              f"Capital: ${self.capital:.2f}")
                else:
                    if iteration % 50 == 0:
                        print(f"[{iteration}] {action_names[int(action)]} | {pos_str} | "
                              f"Period: {elapsed}s | Capital: ${self.capital:.2f}")

                # Save snapshot
                snapshot = {
                    "timestamp": now,
                    "period_start": current_period,
                    "markets": {},
                    "binance": {},
                }
                for period, info in period_data.items():
                    snapshot["markets"][info["slug"]] = {
                        "up": info.get("up_price", 0.5),
                        "down": info.get("down_price", 0.5),
                    }
                if ta_data and klines:
                    snapshot["binance"][f"{self.asset.upper()}USDT"] = {
                        "price": binance_price,
                        **ta_data,
                    }
                self.trade_logger.log_snapshot(snapshot)

                self.current_step += 1
                if self.cooldown_counter < COOLDOWN_TICKS:
                    self.cooldown_counter += 1

                # Period change check - close positions at end of period
                if elapsed >= 895 and self.position is not None:
                    side_str = "UP" if self.position["side"] == 1 else "DOWN"
                    exit_price = None
                    for (p, s), price in market_prices.items():
                        if s == side_str:
                            exit_price = price
                            break
                    if exit_price is None:
                        exit_price = 0.5
                    trade = self._close_position(exit_price, current_period, elapsed)
                    if trade:
                        pnl_pct = trade['pnl'] / trade['size_usd'] * 100
                        emoji = "✅" if trade['pnl'] > 0 else "❌"
                        print(f"[{iteration}] {emoji} PERIOD END CLOSE | "
                              f"P&L: ${trade['pnl']:.2f} ({pnl_pct:+.1f}%)")
                        period_trades += 1

            except KeyboardInterrupt:
                print("\n[Trader] Stopping...")
                break
            except Exception as e:
                print(f"[Trader] Error: {e}")
                import traceback
                traceback.print_exc()

            # Sleep
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0, poll_interval - elapsed_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Final close
        if self.position is not None:
            side_str = "UP" if self.position["side"] == 1 else "DOWN"
            period_data = self.market_fetcher.get_current_period()
            market_prices = {}
            for period, info in period_data.items():
                market_prices[(period, "UP")] = info.get("up_price", 0.5)
                market_prices[(period, "DOWN")] = info.get("down_price", 0.5)
            exit_price = None
            for (p, s), price in market_prices.items():
                if s == side_str:
                    exit_price = price
                    break
            if exit_price is None:
                exit_price = 0.5
            self._close_position(exit_price, self.position["period"], 900)

        # Print summary
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
        print(f"  Wins:             {stats['wins']}")
        print(f"  Losses:           {stats['losses']}")
        print(f"  Win Rate:         {stats['win_rate']*100:.1f}%")
        print(f"  Total P&L:        ${stats['total_pnl']:.2f}")
        print(f"  Total Buy:        ${stats['total_buy']:.2f}")
        print(f"  Avg P&L:          ${stats['avg_pnl']:.2f}")
        print(f"  Avg Win:          ${stats['avg_win']:.2f}")
        print(f"  Avg Loss:         ${stats['avg_loss']:.2f}")
        print(f"  Max Win:          ${stats['max_win']:.2f}")
        print(f"  Max Loss:         ${stats['max_loss']:.2f}")
        print(f"  Final Capital:    ${self.capital:.2f}")
        print(f"  Total Return:     {(self.capital - self.initial_capital) / self.initial_capital * 100:.2f}%")
        print(f"  Peak Capital:     ${self.peak_capital:.2f}")
        print(f"  Max Drawdown:     ${(self.peak_capital - self.capital) / self.peak_capital * 100:.2f}%")
        print(f"{'='*60}")

        summary = {
            "asset": self.asset,
            **stats,
            "final_capital": self.capital,
            "peak_capital": self.peak_capital,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "timestamp": int(time.time()),
        }
        with open(f"trade_logs/summary_{self.asset}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Summary] Saved to trade_logs/summary_{self.asset}.json")

        # Telegram summary
        emoji = "📈" if summary.get("total_pnl", 0) > 0 else "📉"
        send_telegram(
            f"{emoji} <b>Trade Summary — {self.asset.upper()}</b>\n"
            f"Trades: {summary.get('total_trades', 0)} | "
            f"WR: {summary.get('win_rate', 0)*100:.1f}%\n"
            f"P&L: ${summary.get('total_pnl', 0):.2f}\n"
            f"Capital: ${self.capital:.2f} "
            f"({(self.capital - self.initial_capital) / self.initial_capital * 100:+.1f}%)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Live Trader for Polymarket")
    parser.add_argument("--model", required=True, help="Path to PPO model (.zip)")
    parser.add_argument("--asset", default="btc", help="Asset to trade")
    parser.add_argument("--hours", type=float, default=24, help="Trading duration in hours")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL, help="Poll interval in seconds")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode")
    parser.add_argument("--position-size", type=float, default=POSITION_SIZE_PCT, help="Position size fraction")
    args = parser.parse_args()

    trader = RLTrader(
        model_path=args.model,
        asset=args.asset,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        dry_run=args.dry_run,
    )

    trader.run(duration_hours=args.hours, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
