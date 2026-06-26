"""
PolymarketEnv v4 — Model learns fair price, trades on edge.

Architecture:
  Model (PPO) outputs action: 0=HOLD, 1=BUY_UP, 2=BUY_DOWN, 3=SELL
  
  "fair price" is NOT computed by formula. Instead, the model must learn
  to estimate it from the observation features (TA, momentum, etc).
  
  The observation includes all 45 features from v3, plus:
    - normalized time remaining
    - spread
  
  The model implicitly learns fair price by predicting which direction
  the price will move. Reward = actual PnL.
  
  Key: NO deterministic fair price. NO hard edge thresholds.
  Model learns everything from PnL feedback.
  
Actions: 0=HOLD, 1=BUY_UP, 2=BUY_DOWN, 3=SELL_CLOSE
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


ASSETS = ["btc", "eth", "sol"]
TIMESTEPS_PER_PERIOD = 90
TAKER_FEE_RATE = 0.025
POSITION_SIZE_PCT = 0.02
MIN_HOLD_STEPS = 3
MAX_HOLD_STEPS = 20
COOLDOWN_STEPS = 3
OVERTRADE_PENALTY = 0.002
N_FEATURES = 45

_data_cache: Dict[Tuple[str, str], List[Dict]] = {}


@dataclass
class Position:
    side: int          # 1 = UP, -1 = DOWN
    entry_price: float
    size_usd: float
    shares: float
    entry_step: int


class FeatureExtractor:
    """Extract 45 features from market state — same as v3."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.return_history: List[float] = []

    def update(self, up_price: float) -> np.ndarray:
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

        features = np.zeros(N_FEATURES, dtype=np.float32)

        # === Price features (0-4) ===
        features[0] = np.clip(up_price, 0.0, 1.0)
        features[1] = np.clip(1.0 - up_price, 0.0, 1.0)  # down_price
        features[2] = np.clip(up_price + (1.0 - up_price) - 1.0, -0.1, 0.1) * 10.0  # spread

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

        # === Cross-market placeholder (10-13) — filled below ===
        # These are filled by the env from Binance data

        # === Time (14) ===
        # Filled by env

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

        # === TA indicators (20-44) — filled by env ===

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


class PolymarketEnvV4(gym.Env):
    """
    PPO environment for Polymarket 15-minute binary prediction markets.
    Model learns from scratch — no deterministic fair price.
    
    Actions: 0=HOLD, 1=BUY_UP, 2=BUY_DOWN, 3=SELL_CLOSE
    Reward: normalized PnL from closed positions
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

        self.action_space = spaces.Discrete(4)  # HOLD=0, BUY_UP=1, BUY_DOWN=2, SELL=3
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(N_FEATURES,), dtype=np.float32
        )

        self.capital: float = 0.0
        self.position: Optional[Position] = None
        self.current_step: int = 0
        self.current_data_idx: int = 0
        self.peak_capital: float = 0.0
        self.trade_count: int = 0
        self.total_pnl: float = 0.0
        self.wins: int = 0
        self.losses: int = 0
        self.cooldown_counter: int = 0
        self.episode_trades: int = 0

    def _load_data(self, path: str, asset: str) -> List[Dict]:
        """Load and parse expanded snapshots."""
        data = []
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
        try:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    snap = json.loads(line)
                    for market_key, market_data in snap.get("markets", {}).items():
                        if market_key.startswith(f"{asset}-updown-15m-"):
                            binance_key = f"{asset.upper()}USDT"
                            binance_data = snap.get("binance", {}).get(binance_key, {})
                            entry = {
                                "timestamp": snap["timestamp"],
                                "period_start": snap.get("period_start", snap["timestamp"]),
                                "market_key": market_key,
                                "up_price": market_data.get("up", 0.5),
                                "down_price": market_data.get("down", 0.5),
                                "binance_price": binance_data.get("price", 0.0),
                            }
                            for field in ta_fields:
                                entry[f"ta_{field}"] = binance_data.get(field, 0.0)
                            data.append(entry)
        except FileNotFoundError:
            print(f"[EnvV4] Data file not found: {path}")
            return []

        data.sort(key=lambda x: x["timestamp"])
        print(f"[EnvV4] Loaded {len(data)} snapshots for {asset}")
        return data

    def _get_observation(self) -> np.ndarray:
        """Build 45-feature observation from current data index."""
        if self.current_data_idx >= len(self.raw_data):
            self.current_data_idx = len(self.raw_data) - 1
        d = self.raw_data[self.current_data_idx]

        up_price = d["up_price"]
        down_price = d["down_price"]
        elapsed = d["timestamp"] - d.get("period_start", d["timestamp"])

        # Update feature extractor
        features = self.feature_extractor.update(up_price)

        # === Cross-market (10-13) from Binance ===
        features[10] = np.clip(d.get("ta_macd_hist", 0.0) * 5.0, -1.0, 1.0)
        features[11] = np.clip(d.get("ta_momentum_10", 0.0) * 5.0, -1.0, 1.0)
        features[12] = np.clip(d.get("ta_vol_ratio", 1.0) - 1.0, -1.0, 1.0)
        features[13] = np.clip(d.get("ta_rsi", 0.5) - 0.5, -0.5, 0.5) * 2.0

        # === Time (14) ===
        remaining = max(0, 900 - elapsed)
        features[14] = remaining / 900.0

        # === Position (15-17) ===
        features[15] = 1.0 if self.position is not None else 0.0
        if self.position is not None:
            features[16] = float(self.position.side)
            current_price = up_price if self.position.side == 1 else down_price
            unrealized = (current_price - self.position.entry_price) * self.position.shares
            features[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            features[16] = 0.0
            features[17] = 0.0

        # === Regime (18-19) already set by extractor ===
        # 18 = trend strength, 19 = volatility

        # === TA indicators (20-44) ===
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
            val = d.get(f"ta_{field}", 0.0)
            features[20 + i] = np.clip(float(val), -1.0, 1.0)

        return features

    def _open_position(self, side: int, price: float) -> bool:
        """Open a position. side=1 for UP, side=-1 for DOWN."""
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

        self.position = Position(
            side=side,
            entry_price=price,
            size_usd=size_usd,
            shares=shares,
            entry_step=self.current_step,
        )
        self.capital -= fee
        self.trade_count += 1
        return True

    def _close_position(self, exit_price: float) -> Tuple[float, bool]:
        """Close current position. Returns (pnl, is_win)."""
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

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """
        Execute action with helper info in observation.
        No hard rules — model learns when to buy/sell from PnL.
        Only basic safety: can't sell without position, can't buy with position.
        """
        reward = 0.0
        trade_executed = False
        idx = self.current_data_idx

        d = self.raw_data[idx]
        up_price = d["up_price"]
        down_price = d["down_price"]

        if action == 0:  # HOLD
            pass

        elif action == 1:  # BUY UP
            if self.position is None:
                if self._open_position(side=1, price=up_price):
                    trade_executed = True
                    reward -= OVERTRADE_PENALTY

        elif action == 2:  # BUY DOWN
            if self.position is None:
                if self._open_position(side=-1, price=down_price):
                    trade_executed = True
                    reward -= OVERTRADE_PENALTY

        elif action == 3:  # SELL — close position
            if self.position is not None:
                steps_held = self.current_step - self.position.entry_step
                if steps_held >= self.min_hold_steps:
                    exit_price = up_price if self.position.side == 1 else down_price
                    pnl, is_win = self._close_position(exit_price)
                    if self.capital > 0 and self.position_size_pct > 0:
                        reward = pnl / (self.capital * self.position_size_pct + 1e-8)
                    trade_executed = True

        # Auto-close after MAX_HOLD_STEPS (force model to re-evaluate)
        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= MAX_HOLD_STEPS:
                exit_price = up_price if self.position.side == 1 else down_price
                pnl, is_win = self._close_position(exit_price)
                if self.capital > 0 and self.position_size_pct > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)
                trade_executed = True

        return reward, trade_executed

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
        self.feature_extractor.reset()

        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = int(self.rng.integers(10, start_range))

        # Warm up feature extractor
        for _ in range(5):
            self._get_observation()
            self.current_data_idx += 1
            self.current_step += 1

        obs = self._get_observation()
        return obs, {"capital": self.capital, "step": self.current_step}

    def step(self, action):
        assert self.action_space.contains(action)

        trade_reward, _ = self._execute_action(action)
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

        # Force close at end of episode
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

    def get_episode_stats(self):
        total_trades = self.wins + self.losses
        return {
            "final_capital": self.capital,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / total_trades if total_trades > 0 else 0,
            "peak_capital": self.peak_capital,
        }
