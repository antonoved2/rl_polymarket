#!/usr/bin/env python3
"""
PolymarketEnv v7 — Risk-aware training environment.

Same 95 features as v6, but with:
  - Drawdown penalty in reward
  - Volatility-adjusted position sizing
  - Consecutive loss penalty
  - Risk-adjusted reward (Sharpe-like)

This teaches the model to not just maximize PnL, but to manage risk.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

N_FEATURES = 95
TIMESTEPS_PER_PERIOD = 90
TAKER_FEE_RATE = 0.025
POSITION_SIZE_PCT = 0.02
MIN_HOLD_STEPS = 3
MAX_HOLD_STEPS = 20
COOLDOWN_STEPS = 3
OVERTRADE_PENALTY = 0.002

# Risk parameters
MAX_DRAWSOWN_PENALTY = 0.005  # penalty per step when in drawdown
CONSECUTIVE_LOSS_PENALTY = 0.01  # penalty for consecutive losses
RISK_FREE_RATE = 0.0  # for Sharpe calculation

_data_cache: Dict[tuple, List[Dict]] = {}


@dataclass
class Position:
    side: int
    entry_price: float
    size_usd: float
    shares: float
    entry_step: int


class FeatureExtractor:
    """Extract base features (price, volatility, regime)."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.return_history: List[float] = []

    def update(self, up_price: float) -> np.ndarray:
        mid_price = float(up_price)
        self.price_history.append(mid_price)
        if len(self.price_history) > self.lookback + 1:
            self.price_history = self.price_history[-(self.lookback + 1):]

        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.lookback:
                self.return_history = self.return_history[-self.lookback:]

        features = np.zeros(N_FEATURES, dtype=np.float32)

        # Price (0-4)
        features[0] = np.clip(up_price, 0.0, 1.0)
        features[1] = np.clip(1.0 - up_price, 0.0, 1.0)
        features[2] = np.clip(up_price + (1.0 - up_price) - 1.0, -0.1, 0.1) * 10.0
        if len(self.price_history) >= 6:
            features[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        if len(self.price_history) >= 2:
            features[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # Volatility (5-9)
        if len(self.price_history) >= 6:
            features[5] = np.clip((self.price_history[-1] - self.price_history[-6]) * 20.0, -1.0, 1.0)
        if len(self.return_history) >= 2:
            features[6] = np.clip(abs(self.return_history[-1]) * 50.0, 0.0, 1.0)
        if len(self.return_history) >= 3:
            features[7] = 1.0 if abs(self.return_history[-1]) > 0.05 else 0.0
        if len(self.return_history) >= 3:
            features[8] = np.clip((self.return_history[-1] - self.return_history[-3]) * 50.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            features[9] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # Position (15-17)
        features[15] = 0.0
        features[16] = 0.0
        features[17] = 0.0

        # Regime (18-19)
        if len(self.price_history) >= 5:
            total_move = abs(self.price_history[-1] - self.price_history[-5])
            total_range = sum(abs(self.return_history[-i]) for i in range(min(5, len(self.return_history))))
            if total_range > 0:
                features[18] = np.clip(total_move / total_range * 2.0 - 1.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            features[19] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


class PolymarketEnvV7(gym.Env):
    """
    Risk-aware PPO environment for Polymarket.
    95 features, 4 actions, with drawdown penalty.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        data_path: str = "/opt/rl_trader/data/expanded_snapshots.jsonl",
        asset: str = "btc",
        initial_capital: float = 1000.0,
        position_size_pct: float = POSITION_SIZE_PCT,
        taker_fee: float = TAKER_FEE_RATE,
        max_steps_per_episode: int = TIMESTEPS_PER_PERIOD,
        min_hold_steps: int = MIN_HOLD_STEPS,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.asset = asset
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.min_hold_steps = min_hold_steps
        self.rng = np.random.default_rng(seed)

        cache_key = (data_path, asset)
        if cache_key not in _data_cache:
            _data_cache[cache_key] = self._load_data(data_path, asset)
        self.raw_data = _data_cache[cache_key]
        self.feature_extractor = FeatureExtractor(lookback=5)

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
        self.consecutive_losses = 0
        self.step_returns = []

    def _load_data(self, path: str, asset: str) -> List[Dict]:
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

    def _get_observation(self) -> np.ndarray:
        if self.current_data_idx >= len(self.raw_data):
            self.current_data_idx = len(self.raw_data) - 1
        d = self.raw_data[self.current_data_idx]

        up_price = d["up_price"]
        down_price = d["down_price"]
        elapsed = d["timestamp"] - d.get("period_start", d["timestamp"])

        features = self.feature_extractor.update(up_price)

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
            cp = up_price if self.position.side == 1 else down_price
            unrealized = (cp - self.position.entry_price) * self.position.shares
            features[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            features[16] = 0.0
            features[17] = 0.0

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

        # Multi-Horizon ML (83-94)
        for h, base in [(1, 83), (3, 86), (5, 89), (10, 92)]:
            features[base] = np.clip(d.get(f"ml_h{h}_prob_up", 0.5), 0.0, 1.0)
            features[base + 1] = np.clip(d.get(f"ml_h{h}_confidence", 0.0), 0.0, 1.0)
            features[base + 2] = np.clip(d.get(f"ml_h{h}_edge", 0.0), -1.0, 1.0)

        return features

    def _open_position(self, side: int, price: float) -> bool:
        if self.position is not None:
            return False
        if self.cooldown_counter > 0 and self.cooldown_counter < COOLDOWN_STEPS:
            return False
        if price <= 0.01 or price >= 0.99:
            return False

        # Dynamic position size based on drawdown
        dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        size_mult = 0.5 if dd > 0.10 else 1.0
        size_usd = self.capital * self.position_size_pct * size_mult

        if size_usd < 1.0:
            return False

        shares = size_usd / price
        fee = size_usd * self.taker_fee

        self.position = Position(
            side=side, entry_price=price, size_usd=size_usd,
            shares=shares, entry_step=self.current_step,
        )
        self.capital -= fee
        self.trade_count += 1
        return True

    def _close_position(self, exit_price: float) -> Tuple[float, bool]:
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
            self.consecutive_losses = 0
        else:
            self.losses += 1
            self.consecutive_losses += 1
        self.position = None
        self.cooldown_counter = 0
        self.episode_trades += 1
        return pnl, is_win

    def _execute_action(self, action: int) -> float:
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

        # Auto-close after MAX_HOLD_STEPS
        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= MAX_HOLD_STEPS:
                exit_price = up_price if self.position.side == 1 else down_price
                pnl, is_win = self._close_position(exit_price)
                if self.capital > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)

        # === Risk penalties ===
        # Drawdown penalty
        current_dd = (self.peak_capital - self.capital) / self.peak_capital if self.peak_capital > 0 else 0.0
        if current_dd > 0.05:  # penalty starts at 5% drawdown
            reward -= MAX_DRAWSOWN_PENALTY * (current_dd / 0.05)

        # Consecutive loss penalty
        if self.consecutive_losses >= 3:
            reward -= CONSECUTIVE_LOSS_PENALTY * (self.consecutive_losses - 2)

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
        self.consecutive_losses = 0
        self.step_returns = []
        self.feature_extractor.reset()

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
            trade_reward -= 1.0
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
