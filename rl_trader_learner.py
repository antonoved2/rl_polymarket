#!/usr/bin/env python3
"""
RL Trader Learner — цикл "торгуй-обучись" в реальном времени.

Цикл:
  1. Ждём начала нового 15-минутного периода
  2. Торгуем период (модель решает BUY_UP/BUY_DOWN/HOLD)
  3. Собираем снапшоты (Polymarket + Binance OHLCV + TA)
  4. Добавляем в expanded_snapshots.jsonl
  5. Переобучаем PPO
  6. Повторяем

Запуск:
    python3 rl_trader_learner.py --model models/ppo_v4_btc_steps150000 --asset btc --cycles 5
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

sys.path.insert(0, str(Path(__file__).parent))

from environment_v3 import PolymarketEnvV3, N_FEATURES
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ── Config ──────────────────────────────────────────────────
DATA_PATH = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl"
MODEL_DIR = "/home/antonov5/.openclaw/workspace/rl_polymarket/models"

GAMMA_URL = "https://gamma-api.polymarket.com/markets/slug/{slug}"
BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=5m&limit=50"

POLL_INTERVAL = 5       # секунд между тиками
MIN_HOLD_STEPS = 5
POSITION_SIZE_PCT = 0.10
TAKER_FEE = 0.025
INITIAL_CAPITAL = 1000.0

running = True
session = requests.Session()
session.headers.update({"User-Agent": "RLBot/2.0"})


def handle_signal(signum, frame):
    global running
    running = False
    print("\n[STOP] Shutdown signal")


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


# ── Helpers ──────────────────────────────────────────────────

def now_period():
    t = int(time.time())
    return (t // 900) * 900


def wait_for_next_period():
    """Ждём начала следующего 15-минутного периода."""
    now = int(time.time())
    next_p = ((now // 900) + 1) * 900
    wait = next_p - now
    if wait > 0:
        print(f"  [Wait] Next period in {wait}s ({datetime.fromtimestamp(next_p, tz=timezone.utc).strftime('%H:%M:%S')} UTC)")
        # Спим до начала периода (проверяем каждую секунду)
        while running and int(time.time()) < next_p:
            time.sleep(1)
    return next_p


def fetch_poly(asset, period):
    """Получить UP/DOWN цены."""
    try:
        r = session.get(GAMMA_URL.format(slug=f"{asset}-updown-15m-{period}"), timeout=10)
        if r.status_code == 200:
            d = r.json()
            p = d.get("outcomePrices", "[]")
            if isinstance(p, str):
                p = json.loads(p)
            if isinstance(p, list) and len(p) >= 2:
                return float(p[0]), float(p[1])
    except Exception as e:
        pass
    return 0.5, 0.5


def fetch_klines(symbol="BTCUSDT"):
    """Получить OHLCV."""
    try:
        r = session.get(BINANCE_KLINES.format(symbol=symbol), timeout=10)
        if r.status_code == 200:
            return [{"c": float(k[4]), "v": float(k[5])} for k in r.json()]
    except:
        pass
    return []


def compute_ta(closes, volumes=None):
    """Вычислить TA индикаторы."""
    n = len(closes)
    if n < 30:
        return {}

    def _ema(d, p):
        if len(d) < p:
            return d[-1] if d else 0.0
        k = 2.0 / (p + 1)
        e = sum(d[:p]) / p
        for x in d[p:]:
            e = x * k + e * (1 - k)
        return e

    def _rsi(d, p=14):
        if len(d) < p + 1:
            return 50.0
        gains = [max(d[i]-d[i-1], 0) for i in range(len(d)-p, len(d))]
        losses = [max(d[i-1]-d[i], 0) for i in range(len(d)-p, len(d))]
        ag, al = sum(gains)/p, sum(losses)/p
        if al == 0:
            return 100.0
        return 100.0 - (100.0 / (1.0 + ag/al))

    def _macd(d):
        if len(d) < 35:
            return 0, 0, 0
        ms = [_ema(d[:i], 12) - _ema(d[:i], 26) for i in range(26, len(d)+1)]
        ml = ms[-1]
        sl = _ema(ms, 9) if len(ms) >= 9 else ml
        return ml, sl, ml - sl

    def _bb(d, p=20):
        if len(d) < p:
            m = d[-1] if d else 0
            return m, m, m, 0
        s = d[-p:]
        m = sum(s)/p
        std = math.sqrt(sum((x-m)**2 for x in s)/p)
        return m-2*std, m, m+2*std, std

    last = closes[-1] if closes else 1.0
    if last <= 0:
        last = 1.0

    sma5 = sum(closes[-5:])/5 if n >= 5 else last
    sma10 = sum(closes[-10:])/10 if n >= 10 else last
    sma20 = sum(closes[-20:])/20 if n >= 20 else last
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    ema50 = _ema(closes, min(50, n))
    rsi = _rsi(closes)
    ml, sl, hist = _macd(closes)
    bbl, bbm, bbu, bbs = _bb(closes)

    vr = 1.0
    if volumes and len(volumes) >= 20:
        vs = sum(volumes[-20:])/20
        vr = volumes[-1]/vs if vs > 0 else 1.0

    m5 = (closes[-1]-closes[-6])/closes[-6] if n >= 6 and closes[-6] > 0 else 0
    m10 = (closes[-1]-closes[-11])/closes[-11] if n >= 11 and closes[-11] > 0 else 0

    return {
        "ma_cross_5_20": (sma5-sma20)/last,
        "ma_cross_10_20": (sma10-sma20)/last,
        "ma_cross_ema_12_26": (ema12-ema26)/last,
        "price_vs_sma20": (last-sma20)/last,
        "price_vs_ema50": (last-ema50)/last,
        "rsi": rsi/100.0,
        "macd_line": ml/last,
        "macd_signal": sl/last,
        "macd_hist": hist/last,
        "bb_width": (bbu-bbl)/bbm if bbm > 0 else 0,
        "bb_pct_b": (closes[-1]-bbl)/(bbu-bbl) if (bbu-bbl) > 0 else 0.5,
        "bb_upper": bbu, "bb_lower": bbl,
        "atr_pct": 0.0,
        "stoch_k": 0.5, "stoch_d": 0.5,
        "vol_ratio": min(vr, 5.0), "obv": 0.0,
        "momentum_5": m5, "momentum_10": m10,
        "sma_5": sma5, "sma_10": sma10, "sma_20": sma20,
        "ema_12": ema12, "ema_26": ema26,
    }


# ── Feature Extractor (inline, matching rl_bot.py) ──────────

class FE:
    """Feature extractor — 45 features."""
    N = 45

    def __init__(self):
        self.ph = []
        self.rh = []

    def update(self, up, down, bp, ta):
        self.ph.append(float(up))
        if len(self.ph) > 6:
            self.ph = self.ph[-6:]
        if len(self.ph) >= 2:
            self.rh.append(self.ph[-1] - self.ph[-2])
            if len(self.rh) > 5:
                self.rh = self.rh[-5:]

        f = np.zeros(self.N, dtype=np.float32)

        # Price (0-4)
        f[0] = np.clip(up, 0, 1)
        f[1] = np.clip(down, 0, 1)
        f[2] = np.clip((up+down-1)*10, -1, 1)
        if len(self.ph) >= 6:
            f[3] = np.clip((self.ph[-1]-self.ph[-6])*10, -1, 1)
        if len(self.ph) >= 2:
            f[4] = np.clip((self.ph[-1]-self.ph[0])*5, -1, 1)

        # Order book (5-9)
        spread = min(0.005 + 0.02*(1-abs(up-0.5)*2), 0.05)
        f[5] = np.clip(spread*20, 0, 1)
        if len(self.ph) >= 6:
            f[6] = np.clip((self.ph[-1]-self.ph[-6])*20, -1, 1)
        if len(self.rh) >= 2:
            f[7] = np.clip(abs(self.rh[-1])*50, 0, 1)
        if len(self.rh) >= 3:
            f[8] = 1.0 if abs(self.rh[-1]) > 0.05 else 0.0
            f[9] = np.clip((self.rh[-1]-self.rh[-3])*50, -1, 1)

        # Cross-market (10-13) — simplified
        f[10] = 0.0
        f[11] = 0.0
        if len(self.rh) >= 3:
            f[12] = np.clip(np.std(self.rh)*100, 0, 1)
        f[13] = 0.0

        # Time (14)
        # Will be set by caller

        # Position (15-17) — set by caller

        # Regime (18-19)
        if len(self.ph) >= 5:
            tm = abs(self.ph[-1]-self.ph[-5])
            tr = sum(abs(self.rh[-i]) for i in range(min(5,len(self.rh))))
            f[18] = np.clip(tm/tr*2-1, -1, 1) if tr > 0 else 0
        if len(self.rh) >= 5:
            f[19] = np.clip(np.std(self.rh[-5:])*200, 0, 1)

        # TA (20-44)
        if ta:
            fields = [
                "ma_cross_5_20","ma_cross_10_20","ma_cross_ema_12_26",
                "price_vs_sma20","price_vs_ema50",
                "rsi","macd_line","macd_signal","macd_hist",
                "bb_width","bb_pct_b","bb_upper","bb_lower",
                "atr_pct","stoch_k","stoch_d",
                "vol_ratio","obv",
                "momentum_5","momentum_10",
                "sma_5","sma_10","sma_20","ema_12","ema_26",
            ]
            for i, fld in enumerate(fields):
                v = ta.get(fld, 0.0)
                f[20+i] = np.clip(float(v), -1, 1)

        return f


# ── Trade Period ─────────────────────────────────────────────

def trade_period(model, asset, period, capital=INITIAL_CAPITAL):
    """Торгуем один период. Возвращает (trades, snapshots, capital)."""
    fe = FE()
    trades = []
    snapshots = []
    position = None
    step = 0

    period_end = period + 900

    # Warmup
    for _ in range(5):
        up, down = fetch_poly(asset, period)
        fe.update(up, down, 0, None)

    while running and int(time.time()) < period_end:
        now = int(time.time())
        elapsed = now - period

        # Fetch
        up, down = fetch_poly(asset, period)
        klines = fetch_klines(f"{asset.upper()}USDT")
        closes = [k["c"] for k in klines] if klines else []
        volumes = [k["v"] for k in klines] if klines else []
        ta = compute_ta(closes, volumes) if len(closes) >= 30 else {}

        # Snapshot
        snapshots.append({
            "timestamp": now,
            "period_start": period,
            "markets": {f"{asset}-updown-15m-{period}": {"up": round(up, 6), "down": round(down, 6)}},
            "binance": {f"{asset.upper()}USDT": {"price": closes[-1] if closes else 0, **ta}},
        })

        # Observation
        bp = closes[-1] if closes else 0
        obs = fe.update(up, down, bp, ta)
        remaining = max(0, 900 - elapsed)
        obs[14] = remaining / 900.0

        # Position management
        if position is not None:
            held = step - position["step"]
            if held >= MIN_HOLD_STEPS:
                ep = up if position["side"] == 1 else down
                pnl = (ep - position["ep"]) * position["sh"]
                pnl -= position["sz"] * TAKER_FEE
                capital += position["sz"] + pnl
                trades.append({"side": "UP" if position["side"] == 1 else "DOWN",
                               "ep": position["ep"], "xp": ep, "sh": position["sh"],
                               "sz": position["sz"], "pnl": pnl, "cap": capital})
                position = None

        # Model decision
        if position is not None:
            obs[15] = 1.0
            obs[16] = float(position["side"])
            cp = up if position["side"] == 1 else down
            obs[17] = np.clip((cp - position["ep"]) * position["sh"] / position["sz"], -1, 1)

        action, _ = model.predict(obs, deterministic=True)

        if action == 1 and position is None and 0.02 < up < 0.98:
            sz = capital * POSITION_SIZE_PCT
            sh = sz / up
            capital -= sz * TAKER_FEE
            position = {"side": 1, "ep": up, "sz": sz, "sh": sh, "step": step}
        elif action == 2 and position is None and 0.02 < down < 0.98:
            sz = capital * POSITION_SIZE_PCT
            sh = sz / down
            capital -= sz * TAKER_FEE
            position = {"side": -1, "ep": down, "sz": sz, "sh": sh, "step": step}

        step += 1
        time.sleep(POLL_INTERVAL)

    # Force close
    if position is not None:
        now = int(time.time())
        up, down = fetch_poly(asset, period)
        ep = up if position["side"] == 1 else down
        pnl = (ep - position["ep"]) * position["sh"]
        pnl -= position["sz"] * TAKER_FEE
        capital += position["sz"] + pnl
        trades.append({"side": "UP" if position["side"] == 1 else "DOWN",
                       "ep": position["ep"], "xp": ep, "sh": position["sh"],
                       "sz": position["sz"], "pnl": pnl, "cap": capital})

    return trades, snapshots, capital


# ── Retrain ──────────────────────────────────────────────────

def retrain(asset, steps=50000, seed=42):
    """Переобучить PPO."""
    print(f"\n  [Retrain] Loading data ({count_lines(DATA_PATH)} snapshots)...")

    def _init():
        return PolymarketEnvV3(data_path=DATA_PATH, asset=asset,
                               initial_capital=1000.0, position_size_pct=0.10,
                               taker_fee=0.025, max_steps_per_episode=90, seed=seed)

    env = DummyVecEnv([_init])
    model = PPO("MlpPolicy", env, learning_rate=5e-5, n_steps=1024,
                batch_size=256, n_epochs=20, gamma=0.995, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
                verbose=0, device="auto", seed=seed)

    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    elapsed = time.time() - t0

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODEL_DIR, f"ppo_learn_{asset}_{ts}_steps{steps}")
    model.save(path)
    latest = os.path.join(MODEL_DIR, f"ppo_learn_{asset}_latest")
    model.save(latest)
    env.close()

    print(f"  [Retrain] Done in {elapsed:.0f}s → {path}")
    return path, elapsed


def count_lines(path):
    n = 0
    with open(path) as f:
        for _ in f:
            n += 1
    return n


# ── Main Loop ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--asset", default="btc")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--retrain-steps", type=int, default=50000)
    args = parser.parse_args()

    model_path = args.model or os.path.join(MODEL_DIR, f"ppo_learn_{args.asset}_latest")
    if not Path(model_path).exists():
        model_path = os.path.join(MODEL_DIR, f"ppo_v4_{args.asset}_steps150000")

    print("=" * 60)
    print("  RL Trader Learner — торгуй-обучись (LIVE)")
    print("=" * 60)
    print(f"  Model: {model_path}")
    print(f"  Asset: {args.asset}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Retrain: {args.retrain_steps:,} steps")
    print(f"  Poll: {POLL_INTERVAL}s")
    print("=" * 60)

    model = PPO.load(model_path)
    all_stats = []

    for cycle in range(1, args.cycles + 1):
        if not running:
            break

        print(f"\n{'─'*60}")
        print(f"  CYCLE {cycle}/{args.cycles}")
        print(f"{'─'*60}")

        # Wait for next period
        period = wait_for_next_period()
        if not running:
            break

        # Phase 1: Trade
        print(f"\n  [1/3] Trading period {period}...")
        trades, snapshots, capital = trade_period(model, args.asset, period)

        wins = sum(1 for t in trades if t["pnl"] > 0)
        total_pnl = sum(t["pnl"] for t in trades)
        wr = wins / len(trades) * 100 if trades else 0
        print(f"  [1/3] Done: {len(trades)} trades, W:{wins} L:{len(trades)-wins}, "
              f"WR={wr:.0f}%, PnL=${total_pnl:.2f}, Capital=${capital:.2f}")

        # Phase 2: Save data
        print(f"  [2/3] Saving {len(snapshots)} snapshots...")
        with open(DATA_PATH, "a") as f:
            for s in snapshots:
                f.write(json.dumps(s) + "\n")
        total = count_lines(DATA_PATH)
        print(f"  [2/3] Total: {total} snapshots")

        # Phase 3: Retrain
        print(f"  [3/3] Retraining ({args.retrain_steps:,} steps)...")
        new_path, train_time = retrain(args.asset, args.retrain_steps)
        model = PPO.load(new_path)

        stats = {"cycle": cycle, "period": period, "trades": len(trades),
                 "wins": wins, "win_rate": wr, "pnl": total_pnl,
                 "capital": capital, "snapshots": len(snapshots),
                 "total_snapshots": total, "train_time": train_time,
                 "model": new_path}
        all_stats.append(stats)

        with open(os.path.join(MODEL_DIR, f"learn_stats_{args.asset}.json"), "w") as f:
            json.dump(all_stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY — {len(all_stats)} cycles")
    print(f"{'='*60}")
    if all_stats:
        total_t = sum(s["trades"] for s in all_stats)
        total_w = sum(s["wins"] for s in all_stats)
        total_p = sum(s["pnl"] for s in all_stats)
        print(f"  Trades: {total_t} (W:{total_w} L:{total_t-total_w})")
        print(f"  Win Rate: {total_w/total_t*100:.0f}%" if total_t else "  Win Rate: N/A")
        print(f"  Total P&L: ${total_p:.2f}")
        print(f"  Final Capital: ${all_stats[-1]['capital']:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
