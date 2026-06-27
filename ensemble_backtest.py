#!/usr/bin/env python3
"""
Ensemble backtest: combine v7 (83 feat) and v8 (95 feat) models.

Strategy:
  - Both models predict action
  - If they agree → execute
  - If they disagree → HOLD (conservative)
  - Also test: always follow v8 (more features = more informed)
  - Also test: confidence-weighted (if both BUY, strong signal)

This tests whether ensemble reduces variance and improves Sharpe.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

WORKSPACE = Path("/home/antonov5/.openclaw/workspace")

N_FEATURES_V7 = 83
N_FEATURES_V8 = 95
TIMESTEPS_PER_PERIOD = 90
TAKER_FEE_RATE = 0.025
POSITION_SIZE_PCT = 0.02
MIN_HOLD_STEPS = 3
MAX_HOLD_STEPS = 20
COOLDOWN_STEPS = 3
OVERTRADE_PENALTY = 0.002


class PolymarketEnvEnsemble(gym.Env):
    """Environment that supports both 83 and 95 feature models."""

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, data_segment, n_features=N_FEATURES_V8, initial_capital=1000.0,
                 position_size_pct=POSITION_SIZE_PCT, taker_fee=TAKER_FEE_RATE,
                 max_steps_per_episode=TIMESTEPS_PER_PERIOD,
                 min_hold_steps=MIN_HOLD_STEPS, seed=None):
        super().__init__()
        self.raw_data = data_segment
        self.n_features = n_features
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.min_hold_steps = min_hold_steps
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(n_features,), dtype=np.float32
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

        mid = float(up_price)
        self.price_history.append(mid)
        if len(self.price_history) > 6:
            self.price_history = self.price_history[-6:]
        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > 5:
                self.return_history = self.return_history[-5:]

        n = self.n_features
        f = np.zeros(n, dtype=np.float32)

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
            f[16] = float(self.position_side)
            cp = up_price if self.position_side == 1 else down_price
            unrealized = (cp - self.position_entry) * self.position_shares
            f[17] = np.clip(unrealized / self.position_size, -1.0, 1.0)
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

        # ML Forecast (75-82) — only if n >= 83
        if n >= 83:
            f[75] = np.clip(d.get("ml_prob_up", 0.5), 0.0, 1.0)
            f[76] = np.clip(d.get("ml_prob_down", 0.5), 0.0, 1.0)
            f[77] = np.clip(d.get("ml_confidence", 0.0), 0.0, 1.0)
            f[78] = np.clip(d.get("ml_prediction", 0.5), 0.0, 1.0)
            f[79] = np.clip(d.get("ml_edge", 0.0), -1.0, 1.0)
            f[80] = np.clip(d.get("ml_signal_strength", 0.0), -1.0, 1.0)
            f[81] = np.clip(d.get("ml_raw_xgb", 0.5), 0.0, 1.0)
            f[82] = np.clip(d.get("ml_raw_lgb", 0.5), 0.0, 1.0)

        # Multi-Horizon ML (83-94) — only if n >= 95
        if n >= 95:
            for h, base in [(1, 83), (3, 86), (5, 89), (10, 92)]:
                f[base] = np.clip(d.get(f"ml_h{h}_prob_up", 0.5), 0.0, 1.0)
                f[base + 1] = np.clip(d.get(f"ml_h{h}_confidence", 0.0), 0.0, 1.0)
                f[base + 2] = np.clip(d.get(f"ml_h{h}_edge", 0.0), -1.0, 1.0)

        return f

    @property
    def position_side(self):
        return self._pos_side if self.position else 0

    @property
    def position_entry(self):
        return self._pos_entry if self.position else 0

    @property
    def position_shares(self):
        return self._pos_shares if self.position else 0

    @property
    def position_size(self):
        return self._pos_size if self.position else 1

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
        self.position = True
        self._pos_side = side
        self._pos_entry = price
        self._pos_shares = shares
        self._pos_size = size_usd
        self.capital -= fee
        self.trade_count += 1
        return True

    def _close_position(self, exit_price):
        if self.position is None:
            return 0.0, False
        pnl = (exit_price - self._pos_entry) * self._pos_shares
        exit_fee = self._pos_size * self.taker_fee
        pnl -= exit_fee
        is_win = pnl > 0
        self.capital += self._pos_size + pnl
        self.total_pnl += pnl
        if is_win:
            self.wins += 1
        else:
            self.losses += 1
        self.position = None
        self._pos_side = 0
        self._pos_entry = 0
        self._pos_shares = 0
        self._pos_size = 0
        self.cooldown_counter = 0
        self.episode_trades += 1
        return pnl, is_win

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.position = None
        self._pos_side = 0
        self._pos_entry = 0
        self._pos_shares = 0
        self._pos_size = 0
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
                steps_held = self.current_step - self._pos_entry_step if hasattr(self, '_pos_entry_step') else 999
                if steps_held >= self.min_hold_steps:
                    exit_price = up_price if self._pos_side == 1 else down_price
                    pnl, is_win = self._close_position(exit_price)
                    if self.capital > 0:
                        reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        # Auto-close
        if self.position is not None:
            steps_held = self.current_step - self._pos_entry_step if hasattr(self, '_pos_entry_step') else 0
            if steps_held >= MAX_HOLD_STEPS:
                exit_price = up_price if self._pos_side == 1 else down_price
                pnl, is_win = self._close_position(exit_price)
                if self.capital > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        self.peak_capital = max(self.peak_capital, self.capital)
        self.current_step += 1
        self.current_data_idx += 1
        if self.cooldown_counter < COOLDOWN_STEPS:
            self.cooldown_counter += 1

        terminated = self.current_data_idx >= len(self.raw_data) - 1
        truncated = self.current_step >= self.max_steps

        if (terminated or truncated) and self.position is not None:
            idx = min(self.current_data_idx, len(self.raw_data) - 1)
            exit_price = self.raw_data[idx]["up_price"] if self._pos_side == 1 else self.raw_data[idx]["down_price"]
            pnl, _ = self._close_position(exit_price)
            if self.capital > 0:
                reward += pnl / (self.capital * self.position_size_pct + 1e-8)

        obs = self._get_observation()
        info = {
            "capital": self.capital, "total_pnl": self.total_pnl,
            "trade_count": self.trade_count, "wins": self.wins,
            "losses": self.losses, "step": self.current_step,
            "episode_trades": self.episode_trades,
        }
        return obs, reward, terminated, truncated, info


def load_data(data_path):
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


def ensemble_backtest(
    model7_path, model8_path, data_path,
    n_episodes=100, seed=42, mode="agree"
):
    """
    Modes:
      'v7' — only v7 model
      'v8' — only v8 model
      'agree' — both must agree, else HOLD
      'v8_lead' — v8 leads, v7 confirms (if v7 disagrees → HOLD)
    """
    print(f"\n{'='*60}")
    print(f"  Ensemble Backtest — Mode: {mode}")
    print(f"  v7: {model7_path}")
    print(f"  v8: {model8_path}")
    print(f"  Episodes: {n_episodes}")
    print(f"{'='*60}")

    model7 = PPO.load(model7_path) if os.path.exists(model7_path) else None
    model8 = PPO.load(model8_path) if os.path.exists(model8_path) else None

    data = load_data(data_path)
    results = []

    for ep in range(n_episodes):
        # Use v8 env (95 feat) — v7 will just use first 83
        env = PolymarketEnvEnsemble(data_segment=data, n_features=95, seed=seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            # Get actions from both models
            if model8:
                action8, _ = model8.predict(obs, deterministic=True)
            else:
                action8 = 0

            if model7:
                # v7 uses first 83 features
                obs7 = obs[:83]
                action7, _ = model7.predict(obs7, deterministic=True)
            else:
                action7 = 0

            # Ensemble decision
            if mode == "v8":
                action = int(action8)
            elif mode == "v7":
                action = int(action7)
            elif mode == "agree":
                if action7 == action8:
                    action = int(action7)
                else:
                    action = 0  # HOLD on disagreement
            elif mode == "v8_lead":
                if action8 == 0:
                    action = 0
                elif action7 == action8:
                    action = int(action8)
                else:
                    action = 0  # HOLD if v7 disagrees
            else:
                action = int(action8)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        results.append(info)

    capitals = [r['capital'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    returns = [(c - 1000.0) / 1000.0 * 100 for c in capitals]
    wins = sum(r['wins'] for r in results)
    losses = sum(r['losses'] for r in results)
    total_trades = wins + losses

    sharpe = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0.0

    print(f"\n  Results:")
    print(f"    Avg Return:      {np.mean(returns):.2f}%")
    print(f"    Avg PnL:         ${np.mean(pnls):.2f}")
    print(f"    Std Return:      {np.std(returns):.2f}%")
    print(f"    Sharpe:          {sharpe:.2f}")
    print(f"    Win Rate:        {wins/total_trades*100:.1f}%" if total_trades > 0 else "    Win Rate: N/A")
    print(f"    Total Trades:    {total_trades}")
    print(f"    Min Capital:     ${min(capitals):.2f}")
    print(f"    Max Capital:     ${max(capitals):.2f}")

    return {
        'mode': mode,
        'avg_return': float(np.mean(returns)),
        'avg_pnl': float(np.mean(pnls)),
        'std_return': float(np.std(returns)),
        'sharpe': float(sharpe),
        'win_rate': float(wins / total_trades * 100) if total_trades > 0 else 0,
        'total_trades': total_trades,
    }


if __name__ == "__main__":
    model7 = str(WORKSPACE / "rl_polymarket" / "models" / "ppo_v7_btc_steps500000.zip")
    model8 = str(WORKSPACE / "rl_polymarket" / "models" / "ppo_v8_btc_steps500000.zip")
    data = str(WORKSPACE / "rl_polymarket" / "data" / "expanded_snapshots_v4.jsonl")

    all_results = {}
    for mode in ["v7", "v8", "agree", "v8_lead"]:
        result = ensemble_backtest(model7, model8, data, n_episodes=100, mode=mode)
        all_results[mode] = result

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  ENSEMBLE COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Mode':<12} {'Return':>8} {'PnL':>8} {'Sharpe':>8} {'WR':>8} {'Trades':>8}")
    print(f"  {'-'*56}")
    for mode, r in all_results.items():
        print(f"  {mode:<12} {r['avg_return']:>7.2f}% ${r['avg_pnl']:>6.2f} {r['sharpe']:>7.2f} {r['win_rate']:>6.1f}% {r['total_trades']:>7}")

    # Save
    output_path = WORKSPACE / "rl_polymarket" / "models" / "ensemble_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved to {output_path}")
