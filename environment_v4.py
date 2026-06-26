"""
PolymarketEnv v4 — Edge-based statistical arbitrage.

Core idea:
  Model learns to estimate "fair price" from TA indicators + momentum,
  compares it to Polymarket market price, and trades the difference (edge).

  - If PM price < fair_price * (1 + threshold) → BUY UP (undervalued)
  - If PM price > fair_price * (1 - threshold) → BUY DOWN (overvalued)
  - If edge disappears (PM price converges to fair) → SELL (exit)

The model does NOT predict price direction — it detects mispricing
and bets on mean-reversion to fair value.

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
POSITION_SIZE_PCT = 0.10
MIN_HOLD_STEPS = 3
COOLDOWN_STEPS = 5
OVERTRADE_PENALTY = 0.002

# Edge thresholds
EDGE_MIN = 0.03       # minimum 3% edge to enter
EDGE_EXIT = 0.005     # exit when edge < 0.5%

N_FEATURES = 45        # same 45 features as v3

_data_cache: Dict[Tuple[str, str], List[Dict]] = {}


@dataclass
class Position:
    side: int          # 1 = UP, -1 = DOWN
    entry_price: float
    size_usd: float
    shares: float
    entry_step: int
    entry_fair_price: float  # fair price at entry
    entry_edge: float        # edge at entry


def compute_fair_price(state_dict: dict) -> float:
    """
    Compute "fair price" from TA indicators and momentum.
    This is a deterministic function — model uses it as a reference.
    
    Components:
    - Mean-reversion from Bollinger Bands (BB %B)
    - Momentum from MACD histogram
    - Trend from MA cross
    - RSI-based correction
    """
    # Start from 0.5 (neutral)
    fair = 0.5
    
    # Bollinger Band mean-reversion
    # If price is below lower BB, fair value is higher (mean-reversion up)
    # bb_pct_b < 0 → price below lower BB → fair > market
    bb_pct_b = state_dict.get("bb_pct_b", 0.5)
    fair += (0.5 - bb_pct_b) * 0.15  # ±0.075 from BB
    
    # MACD histogram — momentum signal
    macd_hist = state_dict.get("macd_hist", 0.0)
    fair += np.clip(macd_hist * 5.0, -0.05, 0.05)
    
    # MA cross — trend
    ma_cross = state_dict.get("ma_cross_5_20", 0.0)
    fair += np.clip(ma_cross * 3.0, -0.03, 0.03)
    
    # RSI — overbought/oversold mean-reversion
    rsi = state_dict.get("rsi", 0.5)
    fair += (0.5 - rsi) * 0.1  # ±0.05 from RSI
    
    # Momentum 10
    momentum = state_dict.get("momentum_10", 0.0)
    fair += np.clip(momentum * 2.0, -0.02, 0.02)
    
    return np.clip(fair, 0.05, 0.95)


def compute_edge(pm_price: float, fair_price: float) -> float:
    """
    Edge = how much PM price deviates from fair price, as a fraction.
    Positive edge = PM price is below fair (undervalued) → should go up
    Negative edge = PM price is above fair (overvalued) → should go down
    """
    if fair_price <= 0 or fair_price >= 1:
        return 0.0
    return (fair_price - pm_price) / fair_price


class PolymarketEnvV4(gym.Env):
    """
    Edge-based environment.
    
    The observation includes the computed fair price and edge,
    so the model can learn when edge is real vs noise.
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
        self.last_fair_price: float = 0.5
        self.last_edge: float = 0.0

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

    def _get_state_dict(self, idx: int) -> dict:
        """Get raw data entry as dict for fair price computation."""
        if idx >= len(self.raw_data):
            idx = len(self.raw_data) - 1
        if idx < 0:
            idx = 0
        return self.raw_data[idx]

    def _get_observation(self, idx: int) -> np.ndarray:
        """
        Build 45-feature observation.
        Includes fair price and edge as additional info in the observation.
        """
        d = self._get_state_dict(idx)
        obs = np.zeros(N_FEATURES, dtype=np.float32)

        up_price = d["up_price"]
        down_price = d["down_price"]

        # Compute fair price from TA
        ta_dict = {}
        for field in ["bb_pct_b", "macd_hist", "ma_cross_5_20", "rsi", "momentum_10",
                       "ma_cross_10_20", "price_vs_sma20", "price_vs_ema50",
                       "bb_width", "vol_ratio", "momentum_5", "macd_line"]:
            ta_dict[field] = d.get(f"ta_{field}", 0.0)

        fair_price = compute_fair_price(ta_dict)
        edge = compute_edge(up_price, fair_price)

        # === Price features (0-4) ===
        obs[0] = np.clip(up_price, 0.0, 1.0)
        obs[1] = np.clip(down_price, 0.0, 1.0)
        obs[2] = np.clip((up_price + down_price - 1.0) * 10.0, -1.0, 1.0)
        obs[3] = np.clip(edge, -0.5, 0.5) * 2.0  # normalized edge
        obs[4] = np.clip(fair_price - 0.5, -0.45, 0.45) * 2.0  # fair price deviation

        # === Order book (5-9) ===
        spread = up_price + down_price - 1.0
        obs[5] = np.clip(spread * 20.0, -1.0, 1.0)
        obs[6] = np.clip(edge * 5.0, -1.0, 1.0)  # edge magnified
        obs[7] = np.clip(abs(edge) * 10.0, 0.0, 1.0)  # edge magnitude
        obs[8] = 1.0 if abs(edge) > EDGE_MIN else 0.0  # has significant edge
        obs[9] = np.clip((fair_price - up_price) * 5.0, -1.0, 1.0)  # fair vs market

        # === Cross-market (10-13) ===
        obs[10] = np.clip(d.get("ta_macd_hist", 0.0) * 10.0, -1.0, 1.0)
        obs[11] = np.clip(d.get("ta_momentum_10", 0.0) * 5.0, -1.0, 1.0)
        obs[12] = np.clip(d.get("ta_vol_ratio", 1.0) - 1.0, -1.0, 1.0)
        obs[13] = np.clip(d.get("ta_rsi", 0.5) - 0.5, -0.5, 0.5) * 2.0

        # === Time (14) ===
        elapsed = d["timestamp"] - d.get("period_start", d["timestamp"])
        remaining = max(0, 900 - elapsed)
        obs[14] = remaining / 900.0

        # === Position (15-17) ===
        obs[15] = 1.0 if self.position is not None else 0.0
        if self.position is not None:
            obs[16] = float(self.position.side)
            current_price = up_price if self.position.side == 1 else down_price
            unrealized = (current_price - self.position.entry_price) * self.position.shares
            obs[17] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            obs[16] = 0.0
            obs[17] = 0.0

        # === Regime (18-19) ===
        obs[18] = np.clip(d.get("ta_ma_cross_5_20", 0.0) * 5.0, -1.0, 1.0)
        obs[19] = np.clip(d.get("ta_bb_width", 0.0) * 20.0, 0.0, 1.0)

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
            obs[20 + i] = np.clip(float(val), -1.0, 1.0)

        return obs

    def _open_position(self, side: int, price: float, fair_price: float, edge: float) -> bool:
        """Open a position. side=1 for UP, side=-1 for DOWN."""
        if self.position is not None:
            return False
        if self.cooldown_counter > 0 and self.cooldown_counter < COOLDOWN_STEPS:
            return False
        if price < 0.05 or price > 0.95:
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
            entry_fair_price=fair_price,
            entry_edge=edge,
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
        return pnl, is_win

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """
        Execute action with edge-based logic.
        
        BUY_UP: only if edge > EDGE_MIN (UP is undervalued)
        BUY_DOWN: only if edge < -EDGE_MIN (DOWN is undervalued = UP overvalued)
        SELL: only if edge has collapsed (mean-reversion happened)
        """
        reward = 0.0
        trade_executed = False
        idx = self.current_data_idx

        d = self._get_state_dict(idx)
        up_price = d["up_price"]
        down_price = d["down_price"]

        # Compute current fair price and edge
        ta_dict = {field: d.get(f"ta_{field}", 0.0) for field in
                   ["bb_pct_b", "macd_hist", "ma_cross_5_20", "rsi", "momentum_10"]}
        fair_price = compute_fair_price(ta_dict)
        edge = compute_edge(up_price, fair_price)
        self.last_fair_price = fair_price
        self.last_edge = edge

        if action == 0:  # HOLD
            pass

        elif action == 1:  # BUY UP — only if UP is undervalued (positive edge)
            if self.position is None and edge > EDGE_MIN:
                if self._open_position(side=1, price=up_price, fair_price=fair_price, edge=edge):
                    trade_executed = True
                    reward -= OVERTRADE_PENALTY

        elif action == 2:  # BUY DOWN — only if UP is overvalued (negative edge)
            if self.position is None and edge < -EDGE_MIN:
                if self._open_position(side=-1, price=down_price, fair_price=fair_price, edge=-edge):
                    trade_executed = True
                    reward -= OVERTRADE_PENALTY

        elif action == 3:  # SELL — exit if edge has collapsed
            if self.position is not None:
                steps_held = self.current_step - self.position.entry_step
                if steps_held >= self.min_hold_steps:
                    # Exit when edge has mostly disappeared
                    current_edge_abs = abs(edge)
                    if current_edge_abs < EDGE_EXIT:
                        exit_price = up_price if self.position.side == 1 else down_price
                        pnl, is_win = self._close_position(exit_price)
                        if self.position_size_pct > 0:
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
        self.last_fair_price = 0.5
        self.last_edge = 0.0

        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = int(self.rng.integers(10, start_range))

        # Warm up
        for _ in range(3):
            self._get_observation(self.current_data_idx)
            self.current_data_idx += 1
            self.current_step += 1

        obs = self._get_observation(self.current_data_idx)
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

        obs = self._get_observation(self.current_data_idx)

        info = {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "step": self.current_step,
            "fair_price": self.last_fair_price,
            "edge": self.last_edge,
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
