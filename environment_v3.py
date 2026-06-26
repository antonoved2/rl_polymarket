"""
PolymarketEnv v3 — RL environment for Polymarket 15-minute binary prediction markets.

Key design:
- Action 0: HOLD (do nothing)
- Action 1: BUY UP token (bet that outcome = YES)
- Action 2: BUY DOWN token (bet that outcome = NO)

Position mechanics:
- Each token costs $price (0-1 range, like a probability)
- If your side wins: token → $1.00, profit = (1.0 - entry_price) * shares
- If your side loses: token → $0.00, loss = entry_price * shares
- Taker fee: 2.5% on entry AND exit
- Min hold: 5 steps (75 seconds), then can exit at market price

Reward shaping:
- Realized PnL as primary reward signal
- Small penalty for holding too long (time cost)
- No artificial bias toward any side
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass


ASSETS = ["btc", "eth", "sol"]
TIMESTEPS_PER_PERIOD = 90  # 15 min / 10 sec per step
TAKER_FEE_RATE = 0.025
POSITION_SIZE_PCT = 0.10
MIN_HOLD_STEPS = 5


@dataclass
class Position:
    side: int          # 1 = UP, -1 = DOWN
    entry_price: float # price per token ($0-1)
    size_usd: float    # total USD invested
    shares: float      # number of tokens
    entry_step: int    # step when opened


N_FEATURES = 45  # 20 base + 25 TA

_data_cache: Dict[Tuple[str, str], List[Dict]] = {}


@dataclass
class MarketState:
    timestamp: int
    period_start: int
    up_price: float
    down_price: float
    binance_price: float = 0.0
    binance_return_1m: float = 0.0
    binance_return_5m: float = 0.0
    volatility_5m: float = 0.0
    # TA indicators (normalized)
    ta_ma_cross_5_20: float = 0.0
    ta_ma_cross_10_20: float = 0.0
    ta_ma_cross_ema_12_26: float = 0.0
    ta_price_vs_sma20: float = 0.0
    ta_price_vs_ema50: float = 0.0
    ta_rsi: float = 0.5
    ta_macd_line: float = 0.0
    ta_macd_signal: float = 0.0
    ta_macd_hist: float = 0.0
    ta_bb_width: float = 0.0
    ta_bb_pct_b: float = 0.5
    ta_bb_upper: float = 0.5
    ta_bb_lower: float = 0.5
    ta_atr_pct: float = 0.0
    ta_stoch_k: float = 0.5
    ta_stoch_d: float = 0.5
    ta_vol_ratio: float = 1.0
    ta_obv: float = 0.0
    ta_momentum_5: float = 0.0
    ta_momentum_10: float = 0.0
    ta_sma_5_n: float = 0.0
    ta_sma_10_n: float = 0.0
    ta_sma_20_n: float = 0.0
    ta_ema_12_n: float = 0.0
    ta_ema_26_n: float = 0.0


class FeatureExtractor:
    """Extracts 45 normalized features from market state."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.return_history: List[float] = []

    def update(self, state: MarketState) -> np.ndarray:
        mid_price = float(state.up_price)
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
        features[0] = np.clip(state.up_price, 0.0, 1.0)
        features[1] = np.clip(state.down_price, 0.0, 1.0)
        features[2] = np.clip((state.up_price + state.down_price - 1.0) * 10.0, -1.0, 1.0)

        if len(self.price_history) >= 6:
            features[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        if len(self.price_history) >= 2:
            features[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # === Order Book features (5-9) ===
        base_spread = 0.005 + 0.02 * (1.0 - abs(state.up_price - 0.5) * 2)
        vol_factor = 1.0 + state.volatility_5m * 50.0
        spread = min(base_spread * vol_factor, 0.05)
        features[5] = np.clip(spread * 20.0, 0.0, 1.0)

        if len(self.price_history) >= 6:
            features[6] = np.clip((self.price_history[-1] - self.price_history[-6]) * 20.0, -1.0, 1.0)
        if len(self.return_history) >= 2:
            features[7] = np.clip(abs(self.return_history[-1]) * 50.0, 0.0, 1.0)
        if len(self.return_history) >= 3:
            features[8] = 1.0 if abs(self.return_history[-1]) > 0.05 else 0.0
        if len(self.return_history) >= 3:
            features[9] = np.clip((self.return_history[-1] - self.return_history[-3]) * 50.0, -1.0, 1.0)

        # === Cross-market features (10-13) ===
        features[10] = np.clip(state.binance_return_1m * 100.0, -1.0, 1.0)
        features[11] = np.clip(state.binance_return_5m * 100.0, -1.0, 1.0)
        if len(self.return_history) >= 3:
            features[12] = np.clip(np.std(self.return_history) * 100.0, 0.0, 1.0)
        else:
            features[12] = 0.0
        features[13] = np.clip(state.volatility_5m * 100.0, 0.0, 1.0)

        # === Time feature (14) ===
        elapsed = state.timestamp - state.period_start
        remaining = max(0, 900 - elapsed)
        features[14] = remaining / 900.0

        # === Position features (15-17) ===
        features[15] = 0.0  # has_position
        features[16] = 0.0  # position_side
        features[17] = 0.0  # unrealized_pnl_pct

        # === Market regime (18-19) ===
        if len(self.price_history) >= 5:
            total_move = abs(self.price_history[-1] - self.price_history[-5])
            total_range = sum(abs(self.return_history[-i]) for i in range(min(5, len(self.return_history))))
            if total_range > 0:
                features[18] = np.clip(total_move / total_range * 2.0 - 1.0, -1.0, 1.0)
        if len(self.return_history) >= 5:
            features[19] = np.clip(np.std(self.return_history[-5:]) * 200.0, 0.0, 1.0)

        # === TA indicators from Binance (20-44) ===
        ta_fields = [
            "ma_cross_5_20", "ma_cross_10_20", "ma_cross_ema_12_26",
            "price_vs_sma20", "price_vs_ema50",
            "rsi",
            "macd_line", "macd_signal", "macd_hist",
            "bb_width", "bb_pct_b", "bb_upper", "bb_lower",
            "atr_pct",
            "stoch_k", "stoch_d",
            "vol_ratio", "obv",
            "momentum_5", "momentum_10",
            "sma_5_n", "sma_10_n", "sma_20_n", "ema_12_n", "ema_26_n",
        ]
        for i, field in enumerate(ta_fields):
            val = getattr(state, f"ta_{field}", None)
            if val is not None:
                features[20 + i] = np.clip(float(val), -1.0, 1.0)

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


class PolymarketEnvV3(gym.Env):
    """
    Gymnasium environment for Polymarket 15-minute binary markets.

    Actions:
        0 = HOLD (do nothing, keep existing position)
        1 = BUY UP (purchase UP token, betting on YES)
        2 = BUY DOWN (purchase DOWN token, betting on NO)

    Position PnL:
        Entry: pay price per token + 2.5% fee
        Exit: receive exit_price per token - 2.5% fee
        If side wins: exit_price = 1.0
        If side loses: exit_price = 0.0
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
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

        self.action_space = spaces.Discrete(3)  # HOLD=0, BUY_UP=1, BUY_DOWN=2
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

    def _load_data(self, path: str, asset: str) -> List[Dict]:
        """Load and parse expanded snapshots for a specific asset."""
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
            print(f"[Env] Data file not found: {path}")
            return []

        data.sort(key=lambda x: x["timestamp"])
        print(f"[Env] Loaded {len(data)} snapshots for {asset}")
        return data

    def _get_market_state(self, idx: int) -> MarketState:
        """Convert raw data entry to MarketState."""
        if idx >= len(self.raw_data):
            idx = len(self.raw_data) - 1
        if idx < 0:
            idx = 0
        d = self.raw_data[idx]
        state = MarketState(
            timestamp=d["timestamp"],
            period_start=d["period_start"],
            up_price=d["up_price"],
            down_price=d["down_price"],
            binance_price=d["binance_price"],
        )
        # Compute returns
        if idx > 0:
            prev_price = self.raw_data[idx - 1]["binance_price"]
            if prev_price > 0:
                state.binance_return_1m = (d["binance_price"] - prev_price) / prev_price
        if idx >= 5:
            prev_price = self.raw_data[idx - 5]["binance_price"]
            if prev_price > 0:
                state.binance_return_5m = (d["binance_price"] - prev_price) / prev_price
        if idx >= 10:
            prices = [self.raw_data[i]["binance_price"] for i in range(idx - 10, idx + 1)]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            if returns:
                state.volatility_5m = float(np.std(returns))
        # TA indicators
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
        for field in ta_fields:
            val = d.get(f"ta_{field}", 0.0)
            setattr(state, f"ta_{field}", val)
        return state

    def _get_observation(self, features: np.ndarray) -> np.ndarray:
        """Add position info to features and return observation."""
        obs = features.copy()
        obs[15] = 1.0 if self.position is not None else 0.0
        if self.position is not None:
            obs[16] = float(self.position.side)  # 1.0 or -1.0
            idx = min(self.current_data_idx, len(self.raw_data) - 1)
            current_up = self.raw_data[idx]["up_price"]
            current_down = self.raw_data[idx]["down_price"]
            current_price = current_up if self.position.side == 1 else current_down
            # Unrealized PnL as fraction of position size
            unrealized = (current_price - self.position.entry_price) * self.position.shares
            obs[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            obs[16] = 0.0
            obs[17] = 0.0
        return obs

    def _open_position(self, side: int, price: float) -> bool:
        """Open a position. side=1 for UP, side=-1 for DOWN."""
        if self.position is not None:
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
        # PnL calculation:
        # For UP: if exit=1.0, profit = (1.0 - entry) * shares
        # For DOWN: if exit=1.0 (meaning DOWN won), profit = (1.0 - entry) * shares
        # If exit=0.0 (side lost), loss = -entry * shares
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
        return pnl, is_win

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """
        Execute action. Returns (reward, trade_executed).
        Reward is based on realized PnL from closing positions.
        """
        reward = 0.0
        trade_executed = False
        idx = self.current_data_idx

        if action == 0:  # HOLD
            pass
        elif action == 1:  # BUY UP
            if self.position is None:
                up_price = self.raw_data[idx]["up_price"]
                if self._open_position(side=1, price=up_price):
                    trade_executed = True
        elif action == 2:  # BUY DOWN
            if self.position is None:
                down_price = self.raw_data[idx]["down_price"]
                if self._open_position(side=-1, price=down_price):
                    trade_executed = True

        # Auto-close position after min_hold_steps
        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= self.min_hold_steps:
                # Close at current market price
                current_up = self.raw_data[idx]["up_price"]
                current_down = self.raw_data[idx]["down_price"]
                exit_price = current_up if self.position.side == 1 else current_down
                pnl, is_win = self._close_position(exit_price)

                # Reward = normalized PnL
                if self.position_size_pct > 0:
                    reward = pnl / (self.capital * self.position_size_pct + 1e-8)
                else:
                    reward = 0.0

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
        self.feature_extractor.reset()

        # Start at a random point in the data
        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = int(self.rng.integers(10, start_range))

        # Warm up feature extractor
        for _ in range(5):
            state = self._get_market_state(self.current_data_idx)
            self.feature_extractor.update(state)
            self.current_data_idx += 1
            self.current_step += 1

        state = self._get_market_state(self.current_data_idx)
        features = self.feature_extractor.update(state)
        obs = self._get_observation(features)
        return obs, {"capital": self.capital, "step": self.current_step}

    def step(self, action):
        assert self.action_space.contains(action)

        trade_reward, _ = self._execute_action(action)
        self.peak_capital = max(self.peak_capital, self.capital)

        self.current_step += 1
        self.current_data_idx += 1

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
            current_up = self.raw_data[idx]["up_price"]
            current_down = self.raw_data[idx]["down_price"]
            exit_price = current_up if self.position.side == 1 else current_down
            pnl, _ = self._close_position(exit_price)
            if self.capital > 0:
                trade_reward += pnl / (self.capital * self.position_size_pct + 1e-8)

        state = self._get_market_state(self.current_data_idx)
        features = self.feature_extractor.update(state)
        obs = self._get_observation(features)

        info = {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "step": self.current_step,
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


if __name__ == "__main__":
    print("=" * 60)
    print("PolymarketEnvV3 — test run")
    print("=" * 60)
    env = PolymarketEnvV3(asset="btc", initial_capital=1000.0)
    obs, info = env.reset()
    print(f"Obs shape: {obs.shape}, Action space: {env.action_space}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step}: capital=${info['capital']:.2f}, pnl=${info['total_pnl']:.2f}, "
                  f"trades={info['trade_count']}, w={info['wins']}, l={info['losses']}")
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"\nStats: {stats}")
