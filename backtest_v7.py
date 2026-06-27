#!/usr/bin/env python3
"""
Proper backtest for PPO v7 model (83 features).
Uses a standalone environment with exactly 83 features.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")

N_FEATURES = 83
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


class PolymarketEnv83(gym.Env):
    """Standalone 83-feature environment for backtesting v7 model."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, data_path, asset="btc", initial_capital=1000.0,
                 position_size_pct=POSITION_SIZE_PCT, taker_fee=TAKER_FEE_RATE,
                 max_steps_per_episode=TIMESTEPS_PER_PERIOD, min_hold_steps=MIN_HOLD_STEPS,
                 seed=None):
        super().__init__()
        self.asset = asset
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.min_hold_steps = min_hold_steps
        self.rng = np.random.default_rng(seed)

        self.raw_data = self._load_data(data_path, asset)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-3.0, high=3.0, shape=(N_FEATURES,), dtype=np.float32)

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

    def _load_data(self, path, asset):
        data = []
        try:
            with open(path) as f:
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
        except FileNotFoundError:
            return []
        data.sort(key=lambda x: x["timestamp"])
        return data

    def _get_observation(self):
        if self.current_data_idx >= len(self.raw_data):
            self.current_data_idx = len(self.raw_data) - 1
        d = self.raw_data[self.current_data_idx]
        up_price = d["up_price"]
        down_price = d["down_price"]
        elapsed = d["timestamp"] - d.get("period_start", d["timestamp"])

        features = np.zeros(N_FEATURES, dtype=np.float32)

        # Price (0-4)
        features[0] = np.clip(up_price, 0.0, 1.0)
        features[1] = np.clip(down_price, 0.0, 1.0)
        features[2] = np.clip(up_price + down_price - 1.0, -0.1, 0.1) * 10.0
        features[3] = 0.0
        features[4] = 0.0

        # Volatility (5-9)
        features[5] = 0.0
        features[6] = 0.0
        features[7] = 0.0
        features[8] = 0.0
        features[9] = 0.0

        # Cross-market (10-13)
        features[10] = np.clip(d.get("ta_macd_hist", 0.0) * 5.0, -1.0, 1.0)
        features[11] = np.clip(d.get("ta_momentum_10", 0.0) * 5.0, -1.0, 1.0)
        features[12] = np.clip(d.get("ta_vol_ratio", 1.0) - 1.0, -1.0, 1.0)
        features[13] = np.clip(d.get("ta_rsi", 0.5) - 0.5, -0.5, 0.5) * 2.0

        # Time (14)
        remaining = max(0, 900 - elapsed)
        features[14] = remaining / 900.0

        # Position (15-17)
        features[15] = 1.0 if self.position is not None else 0.0
        if self.position is not None:
            features[16] = float(self.position.side)
            current_price = up_price if self.position.side == 1 else down_price
            unrealized = (current_price - self.position.entry_price) * self.position.shares
            features[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            features[16] = 0.0
            features[17] = 0.0

        # Regime (18-19)
        features[18] = 0.0
        features[19] = 0.0

        # Order Book (20-34)
        features[20] = np.clip(d.get("ob_imbalance", 0.0), -1.0, 1.0)
        features[21] = np.clip(d.get("ob_spread_bps", 2.0) / 10.0, 0.0, 1.0)
        features[22] = np.clip(d.get("ob_bid_depth_5", 0.0) / 100.0, 0.0, 1.0)
        features[23] = np.clip(d.get("ob_ask_depth_5", 0.0) / 100.0, 0.0, 1.0)
        features[24] = np.clip(d.get("ob_bid_depth_20", 0.0) / 500.0, 0.0, 1.0)
        features[25] = np.clip(d.get("ob_ask_depth_20", 0.0) / 500.0, 0.0, 1.0)
        features[26] = np.clip(d.get("ob_depth_imbalance_5", 0.0), -1.0, 1.0)
        features[27] = np.clip(d.get("ob_depth_imbalance_20", 0.0), -1.0, 1.0)
        features[28] = np.clip(d.get("ob_wall_bid", 0.0), 0.0, 1.0)
        features[29] = np.clip(d.get("ob_wall_ask", 0.0), 0.0, 1.0)
        features[30] = np.clip(d.get("ob_slope_bid", 0.0), -1.0, 1.0)
        features[31] = np.clip(d.get("ob_slope_ask", 0.0), -1.0, 1.0)
        features[32] = np.clip(d.get("ob_spread", 0.0) * 20.0, 0.0, 1.0)
        features[33] = np.clip(d.get("ob_bid_depth_10", 0.0) / 200.0, 0.0, 1.0)
        features[34] = np.clip(d.get("ob_ask_depth_10", 0.0) / 200.0, 0.0, 1.0)

        # Trade Flow (35-42)
        features[35] = np.clip(d.get("tf_buy_ratio", 0.5), 0.0, 1.0)
        features[36] = np.clip(d.get("tf_flow_imbalance", 0.0), -1.0, 1.0)
        features[37] = np.clip(d.get("tf_large_trades", 0.0), 0.0, 1.0)
        features[38] = np.clip(d.get("tf_avg_size", 0.0) * 10.0, 0.0, 1.0)
        features[39] = np.clip(d.get("tf_size_variance", 0.5), 0.0, 1.0)
        features[40] = np.clip(d.get("tf_aggression", 0.5), 0.0, 1.0)
        features[41] = np.clip(d.get("tf_buy_volume", 0.0) / 1e6, 0.0, 1.0)
        features[42] = np.clip(d.get("tf_sell_volume", 0.0) / 1e6, 0.0, 1.0)

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
            features[43 + i] = np.clip(float(val), -1.0, 1.0)

        # ML Forecast (75-82)
        features[75] = np.clip(d.get("ml_prob_up", 0.5), 0.0, 1.0)
        features[76] = np.clip(d.get("ml_prob_down", 0.5), 0.0, 1.0)
        features[77] = np.clip(d.get("ml_confidence", 0.0), 0.0, 1.0)
        features[78] = np.clip(d.get("ml_prediction", 0.5), 0.0, 1.0)
        features[79] = np.clip(d.get("ml_edge", 0.0), -1.0, 1.0)
        features[80] = np.clip(d.get("ml_signal_strength", 0.0), -1.0, 1.0)
        features[81] = np.clip(d.get("ml_raw_xgb", 0.5), 0.0, 1.0)
        features[82] = np.clip(d.get("ml_raw_lgb", 0.5), 0.0, 1.0)

        return features

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
        self.position = Position(side=side, entry_price=price, size_usd=size_usd, shares=shares, entry_step=self.current_step)
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

    def _execute_action(self, action):
        reward = 0.0
        idx = self.current_data_idx
        d = self.raw_data[idx]
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
                    if self.capital > 0 and self.position_size_pct > 0:
                        reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= MAX_HOLD_STEPS:
                exit_price = up_price if self.position.side == 1 else down_price
                pnl, is_win = self._close_position(exit_price)
                if self.capital > 0 and self.position_size_pct > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        return reward

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

        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = int(self.rng.integers(10, start_range))

        for _ in range(5):
            self._get_observation()
            self.current_data_idx += 1
            self.current_step += 1

        obs = self._get_observation()
        return obs, {"capital": self.capital, "step": self.current_step}

    def step(self, action):
        assert self.action_space.contains(action)
        trade_reward = self._execute_action(action)
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
                trade_reward += pnl / (self.capital * self.position_size_pct + 1e-8)

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
        return obs, trade_reward, terminated, truncated, info


def backtest(model_path, data_path, n_episodes=100):
    print("=" * 60)
    print(f"  Backtest: {model_path}")
    print(f"  Data: {data_path}")
    print(f"  Episodes: {n_episodes}")
    print("=" * 60)

    model = PPO.load(model_path)
    print(f"Model expects {model.observation_space.shape[0]} features")

    env = PolymarketEnv83(data_path=data_path, seed=42)

    episode_results = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        result = {
            'episode': ep,
            'final_capital': info['capital'],
            'total_pnl': info['total_pnl'],
            'total_return_pct': (info['capital'] - 1000.0) / 1000.0 * 100,
            'trade_count': info['trade_count'],
            'wins': info['wins'],
            'losses': info['losses'],
            'episode_trades': info['episode_trades'],
        }
        episode_results.append(result)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")

    capitals = [r['final_capital'] for r in episode_results]
    pnls = [r['total_pnl'] for r in episode_results]
    returns = [r['total_return_pct'] for r in episode_results]
    wins = sum(r['wins'] for r in episode_results)
    losses = sum(r['losses'] for r in episode_results)
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
    avg_win = np.mean(winning_pnls) if winning_pnls else 0.0
    avg_loss = np.mean(losing_pnls) if losing_pnls else 0.0

    ep_wins = sum(1 for r in returns if r > 0)
    ep_win_rate = ep_wins / len(returns) * 100

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'=' * 60}")
    print(f"\nPerformance:")
    print(f"  Avg Capital:      ${np.mean(capitals):.2f}")
    print(f"  Avg PnL:           ${np.mean(pnls):.2f}")
    print(f"  Avg Return:        {np.mean(returns):.2f}%")
    print(f"  Median Return:     {np.median(returns):.2f}%")
    print(f"  Std Return:        {np.std(returns):.2f}%")
    print(f"  Min Capital:       ${min(capitals):.2f}")
    print(f"  Max Capital:       ${max(capitals):.2f}")

    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:      {sharpe:.2f}")
    print(f"  Max Drawdown:      {max_dd:.2f}%")
    print(f"  Profit Factor:     {profit_factor:.2f}")

    print(f"\nTrading Stats:")
    print(f"  Total Trades:      {total_trades}")
    print(f"  Win Rate (trades): {wins/total_trades*100:.1f}%")
    print(f"  Win Rate (episodes): {ep_win_rate:.1f}%")
    print(f"  Avg Win:           ${avg_win:.2f}")
    print(f"  Avg Loss:          ${avg_loss:.2f}")
    print(f"  Avg Trades/Ep:      {total_trades/n_episodes:.1f}")

    print(f"\nReturn Distribution:")
    bins = [-100, -10, -5, -2, 0, 2, 5, 10, 20, 50, 100, 1000]
    for i in range(len(bins) - 1):
        count = sum(1 for r in returns if bins[i] <= r < bins[i+1])
        pct = count / len(returns) * 100
        bar = '█' * int(pct / 2)
        print(f"  [{bins[i]:>5}%, {bins[i+1]:>5}%): {count:>3} ({pct:>5.1f}%) {bar}")

    return {
        'avg_capital': float(np.mean(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'avg_return': float(np.mean(returns)),
        'median_return': float(np.median(returns)),
        'std_return': float(np.std(returns)),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'profit_factor': float(profit_factor),
        'win_rate_trades': float(wins / total_trades * 100),
        'win_rate_episodes': float(ep_win_rate),
        'total_trades': total_trades,
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'min_capital': float(min(capitals)),
        'max_capital': float(max(capitals)),
    }


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else str(
        WORKSPACE / "rl_polymarket" / "models" / "ppo_v7_btc_steps500000.zip"
    )
    data_path = sys.argv[2] if len(sys.argv) > 2 else str(
        WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v3.jsonl"
    )

    stats = backtest(model_path, data_path, n_episodes=100)

    output_path = WORKSPACE / "rl_polymarket" / "models" / "backtest_v7.json"
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {output_path}")
