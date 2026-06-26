#!/usr/bin/env python3
"""
RL Bot — бот для торговли на Polymarket с trained RL моделью.

Использует PPO модель для принятия решений о входе/выходе.
Режим тестирования (dry_run) — без реальных ордеров, только симуляция.

Запуск:
    python3 rl_bot.py --model models/ppo_v3_btc_steps150000 --asset btc --hours 24
"""

import argparse
import json
import os
import sys
import time
import hashlib
import hmac
import base64
import math
import requests
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np
from stable_baselines3 import PPO

try:
    from py_clob_client import ClobClient
    from py_clob_client.clob_types import OrderType, Side, OrderArgs
    HAS_CLOB = True
except ImportError:
    HAS_CLOB = False
    print("[WARN] py_clob_client not installed, running in simulation mode")


# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

WALLET_ADDRESS = "0x2307F20EB8CAaaD5E83b9d2e326DA06cCC28B208"
PRIVATE_KEY = "68fe024167ad9e0ad41229d5f40c406114ffae87539e8b3accb3cb77ec8f9f91"
BUILDER_ADDRESS = "0xA136Fbd3B76a1304742370BddeCadad997837888"

CLOB_API_URL = "https://clob.polymarket.com"
GAMMA_API_URL = "https://gamma-api.polymarket.com"
BINANCE_URL = "https://api.binance.com/api/v3"

INITIAL_CAPITAL = 1000.0
POSITION_SIZE_PCT = 0.10
TAKER_FEE = 0.025
MIN_HOLD_STEPS = 5

# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor (same as environment.py)
# ═══════════════════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Извлекает 45 нормализованных фичей — совместимо с PPO v4+ моделью."""

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
        features[15] = 0.0
        features[16] = 0.0
        features[17] = 0.0

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
    """Получает данные рынка из Gamma API и Binance."""

    def __init__(self, asset="btc"):
        self.asset = asset
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (RLBot/1.0)"})

    def get_current_period(self):
        """Получает текущий и следующий периоды."""
        now = int(time.time())
        current = (now // 900) * 900
        next_period = current + 900

        results = {}
        for period in [current, next_period]:
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
                        results[period] = {
                            "slug": slug,
                            "period_start": period,
                            "up_price": float(prices[0]),
                            "down_price": float(prices[1]),
                            "condition_id": data.get("conditionId", ""),
                            "slug_raw": data.get("slug", slug),
                        }
                        # Get CLOB token IDs
                        clob_ids = data.get("clobTokenIds", "[]")
                        if isinstance(clob_ids, str):
                            clob_ids = json.loads(clob_ids)
                        if isinstance(clob_ids, list) and len(clob_ids) >= 2:
                            results[period]["up_token_id"] = clob_ids[0]
                            results[period]["down_token_id"] = clob_ids[1]
            except Exception as e:
                print(f"[WARN] Failed to fetch {slug}: {e}")
        return results

    def get_binance_price(self):
        """Получает текущую цену на Binance."""
        try:
            symbol = f"{self.asset.upper()}USDT"
            url = f"{BINANCE_URL}/ticker/price?symbol={symbol}"
            resp = self.session.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                return float(data.get("price", 0))
        except:
            pass
        return 0.0

    def get_binance_klines(self, interval="5m", limit=50):
        """Получить OHLCV свечи из Binance для TA."""
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

    def compute_ta(self, closes, volumes=None):
        """Вычислить TA индикаторы из закрытий."""
        import math
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
            "realized_vol": 0.0,
            "momentum_5": mom_5,
            "momentum_10": mom_10,
            "sma_5": sma_5,
            "sma_10": sma_10,
            "sma_20": sma_20,
            "ema_12": ema_12,
            "ema_26": ema_26,
        }

    def get_market_prices(self, period_data):
        """Получает текущие цены из Gamma API."""
        result = {}
        for period, data in period_data.items():
            result[(period, "UP")] = data.get("up_price", 0.5)
            result[(period, "DOWN")] = data.get("down_price", 0.5)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# Trade Logger
# ═══════════════════════════════════════════════════════════════════════════════

class TradeLogger:
    """Ведёт дневник сделок."""

    def __init__(self, log_dir="trade_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trades = []
        self.daily_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": 0})

    def log_trade(self, trade):
        """Записывает сделку в дневник."""
        self.trades.append(trade)
        date_str = datetime.fromtimestamp(trade["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d")
        stats = self.daily_stats[date_str]
        stats["trades"] += 1
        stats["pnl"] += trade["pnl"]
        if trade["pnl"] > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        # Save to file
        log_file = self.log_dir / f"trades_{date_str}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(trade) + "\n")

    def log_decision(self, decision):
        """Записывает решение модели."""
        log_file = self.log_dir / "decisions.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(decision) + "\n")

    def get_summary(self, date_str=None):
        """Возвращает сводку за день."""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        stats = self.daily_stats[date_str]
        total = stats["wins"] + stats["losses"]
        return {
            "date": date_str,
            "total_trades": stats["trades"],
            "wins": stats["wins"],
            "losses": stats["losses"],
            "win_rate": stats["wins"] / total if total > 0 else 0,
            "total_pnl": stats["pnl"],
        }

    def get_all_trades(self):
        return self.trades


# ═══════════════════════════════════════════════════════════════════════════════
# RL Bot
# ═══════════════════════════════════════════════════════════════════════════════

class RLBot:
    """Бот для торговли на Polymarket с RL моделью."""

    def __init__(self, model_path, asset="btc", initial_capital=INITIAL_CAPITAL,
                 position_size_pct=POSITION_SIZE_PCT, dry_run=True):
        self.asset = asset
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position_size_pct = position_size_pct
        self.dry_run = dry_run

        # Load model
        print(f"[Bot] Loading model: {model_path}")
        self.model = PPO.load(model_path)
        print(f"[Bot] Model loaded successfully")

        # State
        self.position = None  # {side, entry_price, shares, entry_step}
        self.current_step = 0
        self.peak_capital = initial_capital
        self.total_pnl = 0.0

        # Data
        self.feature_extractor = FeatureExtractor(lookback=5)
        self.market_fetcher = MarketDataFetcher(asset)
        self.trade_logger = TradeLogger()

        print(f"[Bot] Ready. Capital: ${self.capital:.2f}, Dry run: {self.dry_run}")

    def _warmup(self):
        """Разогрев feature extractor на  historical данных."""
        data_path = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl"
        with open(data_path) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                snap = json.loads(line)
                for key, m in snap.get("markets", {}).items():
                    if key.startswith(f"{self.asset}-updown-15m-"):
                        ta = {k: v for k, v in snap.get("binance", {}).get(self.asset.upper() + "USDT", {}).items() if k not in ("price", "klines")}
                        self.feature_extractor.update(
                            m.get("up", 0.5), m.get("down", 0.5),
                            snap.get("binance", {}).get(f"{self.asset.upper()}USDT", {}).get("price", 0.0), ta_data=ta
                        )

    def get_observation(self, period_data, binance_price, elapsed_sec, ta_data=None):
        """Создает observation из текущих данных рынка."""
        now = int(time.time())
        current_period = (now // 900) * 900

        period_info = period_data.get(current_period)
        if not period_info:
            period_info = list(period_data.values())[0] if period_data else None
            if not period_info:
                return np.zeros(FeatureExtractor.N_FEATURES, dtype=np.float32)

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
        """Выполняет действие модели."""
        now = int(time.time())
        current_period = (now // 900) * 900
        period_info = period_data.get(current_period)
        if not period_info:
            return None

        elapsed = now - current_period

        trade_result = None

        # Helper: get price for side from market_prices
        def get_price(side_str):
            for (p, s), price in market_prices.items():
                if s == side_str:
                    return price
            # Fallback
            if side_str == "UP":
                return period_info.get("up_price", 0.5) if period_info else 0.5
            return period_info.get("down_price", 0.5) if period_info else 0.5

        if action == 0:  # HOLD
            # Check if we should close position
            if self.position is not None:
                steps_held = self.current_step - self.position["entry_step"]
                if steps_held >= MIN_HOLD_STEPS:
                    side_str = "UP" if self.position["side"] == 1 else "DOWN"
                    exit_price = get_price(side_str)
                    if exit_price > 0:
                        trade_result = self._close_position(exit_price, current_period, elapsed)
            return trade_result

        elif action == 1:  # BUY UP
            if self.position is None:
                size_usd = self.capital * self.position_size_pct
                entry_price = get_price("UP")
                if entry_price <= 0 or entry_price >= 0.99:
                    return None
                shares = size_usd / entry_price
                fee = size_usd * TAKER_FEE

                self.position = {
                    "side": 1,
                    "entry_price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "entry_step": self.current_step,
                    "period": current_period,
                }
                self.capital -= fee

                decision = {
                    "timestamp": now,
                    "action": "BUY_UP",
                    "price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "elapsed_sec": elapsed,
                    "capital": self.capital,
                }
                self.trade_logger.log_decision(decision)
                return decision

        elif action == 2:  # BUY DOWN
            if self.position is None:
                size_usd = self.capital * self.position_size_pct
                entry_price = get_price("DOWN")
                if entry_price <= 0 or entry_price >= 0.99:
                    return None
                shares = size_usd / entry_price
                fee = size_usd * TAKER_FEE

                self.position = {
                    "side": -1,
                    "entry_price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "entry_step": self.current_step,
                    "period": current_period,
                }
                self.capital -= fee

                decision = {
                    "timestamp": now,
                    "action": "BUY_DOWN",
                    "price": entry_price,
                    "size_usd": size_usd,
                    "shares": shares,
                    "elapsed_sec": elapsed,
                    "capital": self.capital,
                }
                self.trade_logger.log_decision(decision)
                return decision

        return trade_result

    def _get_exit_price(self, period_info, market_prices, period, side_str):
        """Получает цену выхода."""
        price = market_prices.get((period, side_str), 0)
        if price <= 0:
            # Fallback to gamma price
            key = "up_price" if side_str == "UP" else "down_price"
            price = period_info.get(key, 0.5)
        return price

    def _close_position(self, exit_price, period, elapsed):
        """Закрывает позицию."""
        if self.position is None:
            return None

        pnl = (exit_price - self.position["entry_price"]) * self.position["shares"]
        exit_fee = self.position["size_usd"] * TAKER_FEE
        pnl -= exit_fee

        self.capital += self.position["size_usd"] + pnl
        self.total_pnl += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

        trade = {
            "timestamp": int(time.time()),
            "datetime": datetime.now(timezone.utc).isoformat(),
            "period": self.position["period"],
            "side": "UP" if self.position["side"] == 1 else "DOWN",
            "entry_price": float(self.position["entry_price"]),
            "exit_price": float(exit_price),
            "shares": float(self.position["shares"]),
            "size_usd": float(self.position["size_usd"]),
            "pnl": float(pnl),
            "capital_after": float(self.capital),
            "elapsed_sec": elapsed,
            "steps_held": self.current_step - self.position["entry_step"],
        }
        self.trade_logger.log_trade(trade)
        self.position = None
        return trade

    def run(self, duration_hours=24, poll_interval=10):
        """Запускает бота на указанное время."""
        print(f"\n{'='*60}")
        print(f"  RL Bot — {self.asset.upper()}")
        print(f"  Duration: {duration_hours}h, Poll: {poll_interval}s")
        print(f"  Dry run: {self.dry_run}")
        print(f"  Initial capital: ${self.initial_capital:.2f}")
        print(f"{'='*60}\n")

        start_time = time.time()
        end_time = start_time + duration_hours * 3600
        iteration = 0

        while time.time() < end_time:
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
                market_prices = self.market_fetcher.get_market_prices(period_data)

                # Fetch Binance klines for TA indicators
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

                # Auto-close position after MIN_HOLD_STEPS
                if self.position is not None:
                    steps_held = self.current_step - self.position["entry_step"]
                    if steps_held >= MIN_HOLD_STEPS:
                        side_str = "UP" if self.position["side"] == 1 else "DOWN"
                        # Find exit price
                        exit_price = None
                        for (p, s), price in market_prices.items():
                            if s == side_str:
                                exit_price = price
                                break
                        if exit_price is None:
                            exit_price = 0.5
                        print(f'[{iteration}] AUTO-CLOSE {side_str} after {steps_held} steps', flush=True)
                        self._close_position(exit_price, current_period, elapsed)

                # Model decision
                action, _ = self.model.predict(obs, deterministic=True)

                # Execute
                print(f'[{iteration}] Executing action {action}...', flush=True)
                result = self.execute_action(int(action), period_data, market_prices)
                print(f'[{iteration}] Action executed, result={type(result).__name__}', flush=True)

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

                self.current_step += 1

                # Period change check - close positions at end of period
                if elapsed >= 895 and self.position is not None:
                    side_str = "UP" if self.position["side"] == 1 else "DOWN"
                    exit_price = self._get_exit_price(
                        period_data.get(current_period, {}),
                        market_prices, current_period, side_str
                    )
                    trade = self._close_position(exit_price, current_period, elapsed)
                    if trade:
                        pnl_pct = trade['pnl'] / trade['size_usd'] * 100
                        emoji = "✅" if trade['pnl'] > 0 else "❌"
                        print(f"[{iteration}] {emoji} PERIOD END CLOSE | "
                              f"P&L: ${trade['pnl']:.2f} ({pnl_pct:+.1f}%)")

            except KeyboardInterrupt:
                print("\n[Bot] Stopping...")
                break
            except Exception as e:
                print(f"[Bot] Error: {e}")
                import traceback
                traceback.print_exc()

            # Sleep
            print(f'[{iteration}] Sleeping {poll_interval}s...', flush=True)
            elapsed_loop = time.time() - loop_start
            sleep_time = max(0, poll_interval - elapsed_loop)
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Final close
        if self.position is not None:
            side_str = "UP" if self.position["side"] == 1 else "DOWN"
            period_info = self.market_fetcher.get_current_period().get(self.position["period"], {})
            market_prices = self.market_fetcher.get_market_prices(
                self.market_fetcher.get_current_period()
            )
            exit_price = self._get_exit_price(period_info, market_prices, self.position["period"], side_str)
            self._close_position(exit_price, self.position["period"], 900)

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Печатает итоговый отчёт."""
        trades = self.trade_logger.get_all_trades()
        if not trades:
            print("\n[Summary] No trades made.")
            return

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in trades)
        avg_pnl = total_pnl / len(trades)
        avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0
        max_win = max((t["pnl"] for t in wins), default=0)
        max_loss = min((t["pnl"] for t in trades), default=0)

        print(f"\n{'='*60}")
        print(f"  TRADE SUMMARY — {self.asset.upper()}")
        print(f"{'='*60}")
        print(f"  Total trades:     {len(trades)}")
        print(f"  Wins:             {len(wins)}")
        print(f"  Losses:           {len(losses)}")
        print(f"  Win Rate:         {len(wins)/len(trades)*100:.1f}%")
        print(f"  Total P&L:        ${total_pnl:.2f}")
        print(f"  Avg P&L:          ${avg_pnl:.2f}")
        print(f"  Avg Win:          ${avg_win:.2f}")
        print(f"  Avg Loss:         ${avg_loss:.2f}")
        print(f"  Max Win:          ${max_win:.2f}")
        print(f"  Max Loss:         ${max_loss:.2f}")
        print(f"  Final Capital:    ${self.capital:.2f}")
        print(f"  Total Return:     {(self.capital - self.initial_capital) / self.initial_capital * 100:.2f}%")
        print(f"{'='*60}")

        # Save summary
        summary = {
            "asset": self.asset,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades),
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "final_capital": self.capital,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "timestamp": int(time.time()),
        }
        with open(f"trade_logs/summary_{self.asset}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Summary] Saved to trade_logs/summary_{self.asset}.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RL Bot for Polymarket")
    parser.add_argument("--model", required=True, help="Path to PPO model")
    parser.add_argument("--asset", default="btc", help="Asset to trade")
    parser.add_argument("--hours", type=float, default=24, help="Trading duration in hours")
    parser.add_argument("--poll-interval", type=int, default=10, help="Poll interval in seconds")
    parser.add_argument("--capital", type=float, default=INITIAL_CAPITAL, help="Initial capital")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode (no real orders)")
    args = parser.parse_args()

    bot = RLBot(
        model_path=args.model,
        asset=args.asset,
        initial_capital=args.capital,
        dry_run=args.dry_run,
    )

    bot.run(duration_hours=args.hours, poll_interval=args.poll_interval)


if __name__ == "__main__":
    main()
