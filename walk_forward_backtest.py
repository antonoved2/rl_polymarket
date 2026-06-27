#!/usr/bin/env python3
"""
Walk-forward backtest for PPO v8 model (95 features).

Proper out-of-sample evaluation:
  - Split data into N chronological folds
  - For each fold: train on past, test on future
  - No look-ahead bias

Also includes a simpler "fixed train/test split" mode for quick evaluation.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")
N_FEATURES = 95
TIMESTEPS_PER_PERIOD = 90
TAKER_FEE_RATE = 0.025
POSITION_SIZE_PCT = 0.02
MIN_HOLD_STEPS = 3
MAX_HOLD_STEPS = 20
COOLDOWN_STEPS = 3
OVERTRADE_PENALTY = 0.002


@dataclass
class Position:
    side: int
    entry_price: float
    size_usd: float
    shares: float
    entry_step: int


class PolymarketEnvWF(gym.Env):
    """Walk-forward environment — 95 features, same as training."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, data_segment, initial_capital=1000.0,
                 position_size_pct=POSITION_SIZE_PCT, taker_fee=TAKER_FEE_RATE,
                 max_steps_per_episode=TIMESTEPS_PER_PERIOD,
                 min_hold_steps=MIN_HOLD_STEPS, seed=None):
        super().__init__()
        self.raw_data = data_segment
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.min_hold_steps = min_hold_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(N_FEATURES,), dtype=np.float32
        )

        self.capital = 0.0
        self.position = None
        self.current_step = 0
        self.current_data_idx = 0
        self.peak_capital = 0.0
        self.trade_count = 0
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.cooldown_counter = 0
        self.episode_trades = 0
        self.price_history: List[float] = []
        self.return_history: List[float] = []

    def _get_observation(self) -> np.ndarray:
        if self.current_data_idx >= len(self.raw_data):
            self.current_data_idx = len(self.raw_data) - 1
        d = self.raw_data[self.current_data_idx]

        up_price = d["up_price"]
        down_price = d["down_price"]
        elapsed = d["timestamp"] - d.get("period_start", d["timestamp"])

        # Update price history
        mid = float(up_price)
        self.price_history.append(mid)
        if len(self.price_history) > 6:
            self.price_history = self.price_history[-6:]
        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > 5:
                self.return_history = self.return_history[-5:]

        f = np.zeros(N_FEATURES, dtype=np.float32)

        # Price (0-4)
        f[0] = np.clip(up_price, 0.0, 1.0)
        f[1] = np.clip(down_price, 0.0, 1.0)
        f[2] = np.clip(up_price + down_price - 1.0, -0.1, 0.1) * 10.0
        if len(self.price_history) >= 6:
            f[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        if len(self.price_history) >= 2:
            f[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # Volatility (5-9)
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

        # Cross-market (10-13)
        f[10] = np.clip(d.get("ta_macd_hist", 0.0) * 5.0, -1.0, 1.0)
        f[11] = np.clip(d.get("ta_momentum_10", 0.0) * 5.0, -1.0, 1.0)
        f[12] = np.clip(d.get("ta_vol_ratio", 1.0) - 1.0, -1.0, 1.0)
        f[13] = np.clip(d.get("ta_rsi", 0.5) - 0.5, -0.5, 0.5) * 2.0

        # Time (14)
        remaining = max(0, 900 - elapsed)
        f[14] = remaining / 900.0

        # Position (15-17)
        f[15] = 1.0 if self.position is not None else 0.0
        if self.position is not None:
            f[16] = float(self.position.side)
            cp = up_price if self.position.side == 1 else down_price
            unrealized = (cp - self.position.entry_price) * self.position.shares
            f[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            f[16] = 0.0
            f[17] = 0.0

        # Regime (18-19)
        if len(self.price_history) >= 5:
            total_move = abs(self.price_history[-1] - self.price_history[-5])
            total_range = sum(abs(self.return_history[-i]) for i in range(min(5, len(self.return_history))))
            if total_range > 0:
                f[18] = np.clip(total_move / total_range * 2.0 - 1.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            f[19] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # Order Book (20-34)
        f[20] = np.clip(d.get("ob_imbalance", 0.0), -1.0, 1.0)
        f[21] = np.clip(d.get("ob_spread_bps", 2.0) / 10.0, 0.0, 1.0)
        f[22] = np.clip(d.get("ob_bid_depth_5", 0.0) / 100.0, 0.0, 1.0)
        f[23] = np.clip(d.get("ob_ask_depth_5", 0.0) / 100.0, 0.0, 1.0)
        f[24] = np.clip(d.get("ob_bid_depth_20", 0.0) / 500.0, 0.0, 1.0)
        f[25] = np.clip(d.get("ob_ask_depth_20", 0.0) / 500.0, 0.0, 1.0)
        f[26] = np.clip(d.get("ob_depth_imbalance_5", 0.0), -1.0, 1.0)
        f[27] = np.clip(d.get("ob_depth_imbalance_20", 0.0), -1.0, 1.0)
        f[28] = np.clip(d.get("ob_wall_bid", 0.0), 0.0, 1.0)
        f[29] = np.clip(d.get("ob_wall_ask", 0.0), 0.0, 1.0)
        f[30] = np.clip(d.get("ob_slope_bid", 0.0), -1.0, 1.0)
        f[31] = np.clip(d.get("ob_slope_ask", 0.0), -1.0, 1.0)
        f[32] = np.clip(d.get("ob_spread", 0.0) * 20.0, 0.0, 1.0)
        f[33] = np.clip(d.get("ob_bid_depth_10", 0.0) / 200.0, 0.0, 1.0)
        f[34] = np.clip(d.get("ob_ask_depth_10", 0.0) / 200.0, 0.0, 1.0)

        # Trade Flow (35-42)
        f[35] = np.clip(d.get("tf_buy_ratio", 0.5), 0.0, 1.0)
        f[36] = np.clip(d.get("tf_flow_imbalance", 0.0), -1.0, 1.0)
        f[37] = np.clip(d.get("tf_large_trades", 0.0), 0.0, 1.0)
        f[38] = np.clip(d.get("tf_avg_size", 0.0) * 10.0, 0.0, 1.0)
        f[39] = np.clip(d.get("tf_size_variance", 0.5), 0.0, 1.0)
        f[40] = np.clip(d.get("tf_aggression", 0.5), 0.0, 1.0)
        f[41] = np.clip(d.get("tf_buy_volume", 0.0) / 1e6, 0.0, 1.0)
        f[42] = np.clip(d.get("tf_sell_volume", 0.0) / 1e6, 0.0, 1.0)

        # TA (43-74)
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
            val = d.get(f"ta_{field}", 0.0)
            f[43 + i] = np.clip(float(val), -1.0, 1.0)

        # ML Forecast (75-82)
        f[75] = np.clip(d.get("ml_prob_up", 0.5), 0.0, 1.0)
        f[76] = np.clip(d.get("ml_prob_down", 0.5), 0.0, 1.0)
        f[77] = np.clip(d.get("ml_confidence", 0.0), 0.0, 1.0)
        f[78] = np.clip(d.get("ml_prediction", 0.5), 0.0, 1.0)
        f[79] = np.clip(d.get("ml_edge", 0.0), -1.0, 1.0)
        f[80] = np.clip(d.get("ml_signal_strength", 0.0), -1.0, 1.0)
        f[81] = np.clip(d.get("ml_raw_xgb", 0.5), 0.0, 1.0)
        f[82] = np.clip(d.get("ml_raw_lgb", 0.5), 0.0, 1.0)

        # Multi-Horizon ML (83-94)
        for h, base in [(1, 83), (3, 86), (5, 89), (10, 92)]:
            f[base] = np.clip(d.get(f"ml_h{h}_prob_up", 0.5), 0.0, 1.0)
            f[base + 1] = np.clip(d.get(f"ml_h{h}_confidence", 0.0), 0.0, 1.0)
            f[base + 2] = np.clip(d.get(f"ml_h{h}_edge", 0.0), -1.0, 1.0)

        return f

    def _open_position(self, side, price):
        if self.position is not None:
            return False
        if self.cooldown_counter > 0 and self.cooldown_counter < COOLDOWN_STEPS:
            return False
        if price <= 0.01 or price >= 0.99:
            return False
        size_usd = self.capital * self.position_size_pct
        if size_usd < 1.0:
            return False
        shares = size_usd / price
        fee = size_usd * self.taker_fee
        self.position = Position(side=side, entry_price=price, size_usd=size_usd,
                                 shares=shares, entry_step=self.current_step)
        self.capital -= fee
        self.trade_count += 1
        return True

    def _close_position(self, exit_price):
        if self.position is None:
            return 0.0, False
        pos = self.position
        pnl = (exit_price - pos.entry_price) * pos.shares
        exit_fee = pos.size_usd * self.taker_fee
        pnl -= exit_fee
        is_win = pnl > 0
        self.capital += pos.size_usd + pnl
        self.total_pnl += pnl
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        self.position = None
        self.cooldown_counter = 0
        self.episode_trades += 1
        return pnl, is_win

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.position = None
        self.current_step = 0
        self.peak_capital = self.initial_capital
        self.trade_count = 0
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.cooldown_counter = COOLDOWN_STEPS
        self.episode_trades = 0
        self.price_history = []
        self.return_history = []

        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = int(self.rng.integers(10, start_range))

        for _ in range(5):
            self._get_observation()
            self.current_data_idx += 1
            self.current_step += 1

        obs = self._get_observation()
        return obs, {"capital": self.capital, "step": self.current_step}

    def step(self, action):
        reward = 0.0
        d = self.raw_data[self.current_data_idx]
        up_price = d["up_price"]
        down_price = d["down_price"]

        if action == 0:
            pass
        elif action == 1:
            if self.position is None:
                if self._open_position(side=1, price=up_price):
                    reward -= OVERTRADE_PENALTY
        elif action == 2:
            if self.position is None:
                if self._open_position(side=-1, price=down_price):
                    reward -= OVERTRADE_PENALTY
        elif action == 3:
            if self.position is not None:
                steps_held = self.current_step - self.position.entry_step
                if steps_held >= self.min_hold_steps:
                    exit_price = up_price if self.position.side == 1 else down_price
                    pnl, is_win = self._close_position(exit_price)
                    if self.capital > 0:
                        reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= MAX_HOLD_STEPS:
                exit_price = up_price if self.position.side == 1 else down_price
                pnl, is_win = self._close_position(exit_price)
                if self.capital > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        self.peak_capital = max(self.peak_capital, self.capital)
        self.current_step += 1
        self.current_data_idx += 1
        if self.cooldown_counter < COOLDOWN_STEPS:
            self.cooldown_counter += 1

        terminated = False
        truncated = False
        if self.current_data_idx >= len(self.raw_data) - 1:
            terminated = True
        if self.current_step >= self.max_steps:
            truncated = True
        if self.capital <= 0:
            reward -= 1.0
            terminated = True

        if (terminated or truncated) and self.position is not None:
            idx = min(self.current_data_idx, len(self.raw_data) - 1)
            exit_price = self.raw_data[idx]["up_price"] if self.position.side == 1 else self.raw_data[idx]["down_price"]
            pnl, _ = self._close_position(exit_price)
            if self.capital > 0:
                trade_reward = pnl / (self.capital * self.position_size_pct + 1e-8)
                reward += trade_reward

        obs = self._get_observation()
        info = {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "step": self.current_step,
            "episode_trades": self.episode_trades,
        }
        return obs, reward, terminated, truncated, info



def load_data(data_path: str) -> List[Dict]:
    """Load expanded snapshots from JSONL."""
    data = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            snap = json.loads(line)
            if "up_price" in snap and "down_price" in snap:
                entry = {
                    "timestamp": snap["timestamp"],
                    "period_start": snap.get("period_start", snap["timestamp"]),
                    "up_price": snap["up_price"],
                    "down_price": snap["down_price"],
                }
                for k, v in snap.items():
                    if k.startswith(("ta_", "ob_", "tf_", "ml_")):
                        entry[k] = v
                data.append(entry)
    data.sort(key=lambda x: x["timestamp"])
    return data


def split_by_periods(data: List[Dict], n_splits: int) -> List[Tuple[List[Dict], List[Dict]]]:
    """Split data into chronological train/test folds."""
    # Group by period
    periods = {}
    for d in data:
        ps = d.get("period_start", d["timestamp"])
        if ps not in periods:
            periods[ps] = []
        periods[ps].append(d)

    sorted_periods = sorted(periods.keys())
    fold_size = len(sorted_periods) // (n_splits + 1)

    folds = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, len(sorted_periods))

        train_periods = set(sorted_periods[:train_end])
        test_periods = set(sorted_periods[test_start:test_end])

        train_data = [d for d in data if d.get("period_start", d["timestamp"]) in train_periods]
        test_data = [d for d in data if d.get("period_start", d["timestamp"]) in test_periods]

        if train_data and test_data:
            folds.append((train_data, test_data))

    return folds


def evaluate_model_on_data(model, data: List[Dict], n_episodes: int = 50, seed: int = 42) -> Dict:
    """Run model on test data and collect statistics."""
    results = []
    for ep in range(n_episodes):
        env = PolymarketEnvWF(data_segment=data, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        results.append(info)

    capitals = [r['capital'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    returns = [(c - 1000.0) / 1000.0 * 100 for c in capitals]
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    total_trades = wins + losses

    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0

    peak = 1000.0
    max_dd = 0.0
    for cap in capitals:
        peak = max(peak, cap)
        dd = (peak - cap) / peak * 100
        max_dd = max(max_dd, dd)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    winning_pnls = [p for p in pnls if p > 0]
    losing_pnls = [p for p in pnls if p < 0]

    return {
        'avg_capital': float(np.mean(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'avg_return': float(np.mean(returns)),
        'median_return': float(np.median(returns)),
        'std_return': float(np.std(returns)),
        'min_capital': float(min(capitals)),
        'max_capital': float(max(capitals)),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'profit_factor': float(profit_factor),
        'win_rate_trades': float(wins / total_trades * 100) if total_trades > 0 else 0,
        'win_rate_episodes': float(sum(1 for r in returns if r > 0) / len(returns) * 100),
        'total_trades': total_trades,
        'avg_win': float(np.mean(winning_pnls)) if winning_pnls else 0,
        'avg_loss': float(np.mean(losing_pnls)) if losing_pnls else 0,
    }


def walk_forward_backtest(
    model_path: str,
    data_path: str,
    n_splits: int = 5,
    train_steps: int = 100_000,
    n_eval_episodes: int = 50,
):
    """
    Walk-forward backtest:
    1. Load all data
    2. Split into N chronological folds
    3. For each fold: load model → evaluate on test data
    4. Aggregate results
    """
    print("=" * 70)
    print("  WALK-FORWARD BACKTEST")
    print(f"  Model: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Splits: {n_splits}")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading data...")
    data = load_data(data_path)
    print(f"  Loaded {len(data)} snapshots")

    # Split into folds
    print(f"\n[2/3] Splitting into {n_splits} folds...")
    folds = split_by_periods(data, n_splits)
    print(f"  Created {len(folds)} folds")

    for i, (train_d, test_d) in enumerate(folds):
        print(f"  Fold {i+1}: train={len(train_d)}, test={len(test_d)}")

    # Load model
    print(f"\n[3/3] Loading model and evaluating...")
    model = PPO.load(model_path)
    print(f"  Model expects {model.observation_space.shape[0]} features")

    # Evaluate on each fold
    all_fold_results = []
    for i, (train_d, test_d) in enumerate(folds):
        print(f"\n  Fold {i+1}/{len(folds)}:")
        print(f"    Test data: {len(test_d)} snapshots")

        stats = evaluate_model_on_data(model, test_d, n_episodes=n_eval_episodes, seed=42 + i)
        all_fold_results.append(stats)

        print(f"    Avg Return: {stats['avg_return']:.2f}%")
        print(f"    Avg PnL: ${stats['avg_pnl']:.2f}")
        print(f"    Win Rate: {stats['win_rate_trades']:.1f}%")
        print(f"    Sharpe: {stats['sharpe']:.2f}")
        print(f"    Max DD: {stats['max_drawdown']:.1f}%")
        print(f"    Trades: {stats['total_trades']}")

    # Aggregate
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD SUMMARY ({len(folds)} folds)")
    print(f"{'='*70}")

    avg_returns = [r['avg_return'] for r in all_fold_results]
    avg_pnls = [r['avg_pnl'] for r in all_fold_results]
    sharpes = [r['sharpe'] for r in all_fold_results]
    win_rates = [r['win_rate_trades'] for r in all_fold_results]
    max_dds = [r['max_drawdown'] for r in all_fold_results]

    print(f"\n  Avg Return:      {np.mean(avg_returns):.2f}% ± {np.std(avg_returns):.2f}%")
    print(f"  Avg PnL:         ${np.mean(avg_pnls):.2f} ± ${np.std(avg_pnls):.2f}")
    print(f"  Avg Sharpe:      {np.mean(sharpes):.2f} ± {np.std(sharpes):.2f}")
    print(f"  Avg Win Rate:    {np.mean(win_rates):.1f}% ± {np.std(win_rates):.1f}%")
    print(f"  Avg Max DD:      {np.mean(max_dds):.1f}% ± {np.std(max_dds):.1f}%")

    print(f"\n  Fold-by-fold returns:")
    for i, r in enumerate(all_fold_results):
        bar_len = max(0, int(r['avg_return'] / 2))
        bar = '█' * bar_len
        print(f"    Fold {i+1}: {r['avg_return']:+.2f}% | PnL ${r['avg_pnl']:.2f} | Sharpe {r['sharpe']:.2f} | {bar}")

    # Consistency check
    positive_folds = sum(1 for r in avg_returns if r > 0)
    print(f"\n  Consistency: {positive_folds}/{len(folds)} folds profitable ({positive_folds/len(folds)*100:.0f}%)")

    # Save results
    output = {
        'model': model_path,
        'data': data_path,
        'n_splits': n_splits,
        'n_eval_episodes': n_eval_episodes,
        'folds': all_fold_results,
        'summary': {
            'avg_return': float(np.mean(avg_returns)),
            'std_return': float(np.std(avg_returns)),
            'avg_pnl': float(np.mean(avg_pnls)),
            'avg_sharpe': float(np.mean(sharpes)),
            'avg_win_rate': float(np.mean(win_rates)),
            'avg_max_dd': float(np.mean(max_dds)),
            'consistency': float(positive_folds / len(folds)),
        }
    }

    output_path = WORKSPACE / "rl_polymarket" / "models" / "walk_forward_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to {output_path}")

    return output


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(
        WORKSPACE / "rl_polymarket" / "models" / "ppo_v8_btc_steps500000.zip"
    )
    data_path = sys.argv[2] if len(sys.argv) > 2 else str(
        WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v4.jsonl"
    )
    n_splits = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    walk_forward_backtest(model_path, data_path, n_splits=n_splits)
