"""
PolymarketEnv v2 — расширенная среда с CLOB-подобными фичами и multi-asset.

Улучшения:
- 20 фичей (было 14): добавлены bid/ask spread, depth imbalance, trade intensity
- Multi-asset: одна среда для BTC+ETH+SOL с общим капиталом
- Реалистичные order book фичи на основе динамики цен
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


@dataclass
class Position:
    side: int
    entry_price: float
    size_usd: float
    shares: float
    entry_step: int
    asset: str


@dataclass
class MarketState:
    timestamp: int
    period_start: int
    asset: str
    up_price: float
    down_price: float
    binance_price: float = 0.0
    binance_return_1m: float = 0.0
    binance_return_5m: float = 0.0
    volatility_5m: float = 0.0
    # CLOB-like features (synthetic, derived from price dynamics)
    up_bid: float = 0.0
    up_ask: float = 0.0
    down_bid: float = 0.0
    down_ask: float = 0.0
    bid_ask_spread: float = 0.0
    depth_imbalance: float = 0.0
    trade_intensity: float = 0.0
    large_trade_flag: float = 0.0


class FeatureExtractor:
    """Извлекает 20 нормализованных фичей."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.return_history: List[float] = []
        self.spread_history: List[float] = []
        self.volume_history: List[float] = []

    def update(self, state: MarketState) -> np.ndarray:
        mid_price = float(state.up_price)
        self.price_history.append(mid_price)
        max_len = self.lookback + 1
        if len(self.price_history) > max_len:
            self.price_history = self.price_history[len(self.price_history) - max_len:]

        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.lookback:
                self.return_history = self.return_history[len(self.return_history) - self.lookback:]

        # Track spread
        self.spread_history.append(state.bid_ask_spread)
        if len(self.spread_history) > self.lookback:
            self.spread_history = self.spread_history[len(self.spread_history) - self.lookback:]

        # Track volume (trade intensity)
        self.volume_history.append(state.trade_intensity)
        if len(self.volume_history) > self.lookback:
            self.volume_history = self.volume_history[len(self.volume_history) - self.lookback:]

        features = np.zeros(20, dtype=np.float32)

        # === Price features (0-4) ===
        features[0] = np.clip(state.up_price, 0.0, 1.0)
        features[1] = np.clip(state.down_price, 0.0, 1.0)
        features[2] = np.clip((state.up_price + state.down_price - 1.0) * 10.0, -1.0, 1.0)

        # Momentum 5 steps
        if len(self.price_history) >= 6:
            features[3] = np.clip((self.price_history[-1] - self.price_history[-6]) * 10.0, -1.0, 1.0)
        # Momentum all
        if len(self.price_history) >= 2:
            features[4] = np.clip((self.price_history[-1] - self.price_history[0]) * 5.0, -1.0, 1.0)

        # === Order Book features (5-9) ===
        # Bid-ask spread
        features[5] = np.clip(state.bid_ask_spread * 20.0, 0.0, 1.0)
        # Depth imbalance
        features[6] = np.clip(state.depth_imbalance, -1.0, 1.0)
        # Trade intensity
        features[7] = np.clip(state.trade_intensity * 5.0, 0.0, 1.0)
        # Large trade flag
        features[8] = np.clip(state.large_trade_flag, 0.0, 1.0)
        # Spread momentum (is spread widening?)
        if len(self.spread_history) >= 3:
            features[9] = np.clip((self.spread_history[-1] - self.spread_history[-3]) * 50.0, -1.0, 1.0)

        # === Cross-market features (10-13) ===
        # Binance return 1m
        features[10] = np.clip(state.binance_return_1m * 100.0, -1.0, 1.0)
        # Binance return 5m
        features[11] = np.clip(state.binance_return_5m * 100.0, -1.0, 1.0)
        # Volatility
        if len(self.return_history) >= 3:
            features[12] = np.clip(np.std(self.return_history) * 100.0, 0.0, 1.0)
        # Volume momentum
        if len(self.volume_history) >= 3:
            features[13] = np.clip((self.volume_history[-1] - np.mean(self.volume_history[:-1])) * 10.0, -1.0, 1.0)

        # === Time feature (14) ===
        elapsed = state.timestamp - state.period_start
        remaining = max(0, 900 - elapsed)
        features[14] = remaining / 900.0

        # === Position features (15-18) ===
        features[15] = 0.0  # has_position (set by env)
        features[16] = 0.0  # position_side
        features[17] = 0.0  # position_pnl
        features[18] = 0.0  # position_asset_onehot (set by env)

        # === Market regime (19) ===
        # High volatility regime
        if len(self.return_history) >= 5:
            recent_vol = np.std(self.return_history[-5:])
            features[19] = np.clip(recent_vol * 200.0, 0.0, 1.0)

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()
        self.spread_history.clear()
        self.volume_history.clear()


class PolymarketMultiEnv(gym.Env):
    """
    Multi-asset Gymnasium environment.

    Агент торгует на нескольких активах (BTC, ETH, SOL) одновременно.
    Общий капитал, позиции по каждому активу.

    Action: HOLD(0), BUY(1), SELL(2) — применяется к выбранному активу
    Нужно указать какой актив через action space: 3 actions × 3 assets = 9 действий

    Или проще: action = asset_idx * 3 + action_type
    0: BTC HOLD, 1: BTC BUY, 2: BTC SELL
    3: ETH HOLD, 4: ETH BUY, 5: ETH SELL
    6: SOL HOLD, 7: SOL BUY, 8: SOL SELL
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
        assets: List[str] = None,
        initial_capital: float = 1000.0,
        position_size_pct: float = POSITION_SIZE_PCT,
        taker_fee: float = TAKER_FEE_RATE,
        max_steps_per_episode: int = TIMESTEPS_PER_PERIOD,
        drawdown_penalty: float = 0.1,
        trade_penalty: float = 0.001,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.assets = assets or ASSETS
        self.n_assets = len(self.assets)
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.drawdown_penalty = drawdown_penalty
        self.trade_penalty = trade_penalty
        self.rng = np.random.default_rng(seed)

        # Load data for all assets
        self.raw_data = {}
        for asset in self.assets:
            self.raw_data[asset] = self._load_data(data_path, asset)

        self.feature_extractors = {a: FeatureExtractor(lookback=5) for a in self.assets}

        # Action space: n_assets * 3 (HOLD/BUY/SELL per asset)
        self.action_space = spaces.Discrete(self.n_assets * 3)

        # Observation: 20 features per asset + 3 position features + 1 capital feature
        obs_size = self.n_assets * 20 + self.n_assets * 3 + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # State
        self.capital = 0.0
        self.positions = {}  # asset -> Position
        self.current_step = 0
        self.current_data_idx = {}
        self.peak_capital = 0.0
        self.trade_count = 0
        self.total_pnl = 0.0

    def _load_data(self, path: str, asset: str) -> List[Dict]:
        data = []
        with open(path) as f:
            for line in f:
                snap = json.loads(line.strip())
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
                        data.append(entry)
        data.sort(key=lambda x: x["timestamp"])
        print(f"[Env] Loaded {len(data)} snapshots for {asset}")
        return data

    def _get_market_state(self, asset: str, idx: int) -> MarketState:
        data = self.raw_data[asset]
        if idx >= len(data):
            idx = len(data) - 1

        d = data[idx]
        state = MarketState(
            timestamp=d["timestamp"],
            period_start=d["period_start"],
            asset=asset,
            up_price=d["up_price"],
            down_price=d["down_price"],
            binance_price=d["binance_price"],
        )

        # Binance returns
        if idx > 0:
            prev = data[idx - 1]["binance_price"]
            if prev > 0:
                state.binance_return_1m = (d["binance_price"] - prev) / prev
        if idx >= 5:
            prev = data[idx - 5]["binance_price"]
            if prev > 0:
                state.binance_return_5m = (d["binance_price"] - prev) / prev

        # Volatility
        if idx >= 10:
            prices = [data[i]["binance_price"] for i in range(idx - 10, idx + 1)]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            if returns:
                state.volatility_5m = float(np.std(returns))

        # Generate synthetic CLOB-like features from price dynamics
        self._generate_clob_features(state, data, idx)

        return state

    def _generate_clob_features(self, state: MarketState, data: List[Dict], idx: int):
        """Генерирует реалистичные order book фичи на основе динамики цен."""
        up = state.up_price
        down = state.down_price

        # Bid-ask spread: wider near 0.5 (uncertain), narrower near 0/1
        # Also wider with higher volatility
        base_spread = 0.005 + 0.02 * (1.0 - abs(up - 0.5) * 2)  # 0.5% to 2.5%
        vol_factor = 1.0 + state.volatility_5m * 50.0  # higher vol = wider spread
        spread = min(base_spread * vol_factor, 0.05)  # cap at 5%

        state.up_bid = max(up - spread / 2, 0.001)
        state.up_ask = min(up + spread / 2, 0.999)
        state.down_bid = max(down - spread / 2, 0.001)
        state.down_ask = min(down + spread / 2, 0.999)
        state.bid_ask_spread = spread

        # Depth imbalance: based on momentum
        # If price going up, more depth on bid (buyers aggressive)
        if idx >= 5:
            momentum = up - data[idx - 5]["up_price"]
            # Normalize: positive momentum -> positive imbalance (more bids)
            state.depth_imbalance = np.clip(momentum * 20.0, -1.0, 1.0)
        else:
            state.depth_imbalance = 0.0

        # Trade intensity: based on absolute price change
        if idx > 0:
            abs_change = abs(up - data[idx - 1]["up_price"])
            state.trade_intensity = min(abs_change * 50.0, 1.0)
        else:
            state.trade_intensity = 0.0

        # Large trade flag: sudden price movement
        if idx > 0:
            abs_change = abs(up - data[idx - 1]["up_price"])
            state.large_trade_flag = 1.0 if abs_change > 0.05 else 0.0
        else:
            state.large_trade_flag = 0.0

    def _get_observation(self, features_per_asset: Dict[str, np.ndarray]) -> np.ndarray:
        """Собирает полный observation из фичей всех активов."""
        obs_parts = []

        # Features per asset (20 each)
        for asset in self.assets:
            obs_parts.append(features_per_asset[asset])

        # Position features per asset (3 each: has_pos, side, pnl)
        for asset in self.assets:
            pos_feat = np.zeros(3, dtype=np.float32)
            if asset in self.positions:
                pos = self.positions[asset]
                pos_feat[0] = 1.0
                pos_feat[1] = float(pos.side)
                # Unrealized P&L
                data = self.raw_data[asset]
                idx = min(self.current_data_idx.get(asset, 0), len(data) - 1)
                current_up = data[idx]["up_price"]
                current_down = data[idx]["down_price"]
                if pos.side == 1:
                    current_price = current_up
                else:
                    current_price = current_down
                unrealized = (current_price - pos.entry_price) * pos.shares
                pos_feat[2] = np.clip(unrealized / pos.size_usd, -1.0, 1.0)
            obs_parts.append(pos_feat)

        # Capital feature
        capital_feat = np.array([
            np.clip((self.capital - self.initial_capital) / self.initial_capital, -1.0, 1.0)
        ], dtype=np.float32)
        obs_parts.append(capital_feat)

        return np.concatenate(obs_parts).astype(np.float32)

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """Выполняет действие."""
        asset_idx = action // 3
        action_type = action % 3

        if asset_idx >= self.n_assets:
            return 0.0, False

        asset = self.assets[asset_idx]
        reward = 0.0
        trade_executed = False

        if action_type == 0:  # HOLD
            pass

        elif action_type == 1:  # BUY
            if asset not in self.positions:
                data = self.raw_data[asset]
                idx = self.current_data_idx.get(asset, 0)
                if idx < len(data):
                    up_price = data[idx]["up_price"]
                    if 0.01 < up_price < 0.99:
                        size_usd = self.capital * self.position_size_pct
                        shares = size_usd / up_price
                        fee = size_usd * self.taker_fee
                        self.positions[asset] = Position(
                            side=1, entry_price=up_price, size_usd=size_usd,
                            shares=shares, entry_step=self.current_step, asset=asset,
                        )
                        self.capital -= fee
                        self.trade_count += 1
                        trade_executed = True

        elif action_type == 2:  # SELL
            if asset not in self.positions:
                data = self.raw_data[asset]
                idx = self.current_data_idx.get(asset, 0)
                if idx < len(data):
                    down_price = data[idx]["down_price"]
                    if 0.01 < down_price < 0.99:
                        size_usd = self.capital * self.position_size_pct
                        shares = size_usd / down_price
                        fee = size_usd * self.taker_fee
                        self.positions[asset] = Position(
                            side=-1, entry_price=down_price, size_usd=size_usd,
                            shares=shares, entry_step=self.current_step, asset=asset,
                        )
                        self.capital -= fee
                        self.trade_count += 1
                        trade_executed = True

        # Close positions (after min 5 steps)
        for asset in list(self.positions.keys()):
            pos = self.positions[asset]
            steps_held = self.current_step - pos.entry_step
            if steps_held >= 5:
                data = self.raw_data[asset]
                idx = min(self.current_data_idx.get(asset, 0), len(data) - 1)
                current_up = data[idx]["up_price"]
                current_down = data[idx]["down_price"]

                if pos.side == 1:
                    exit_price = current_up
                else:
                    exit_price = current_down

                pnl = (exit_price - pos.entry_price) * pos.shares
                exit_fee = pos.size_usd * self.taker_fee
                pnl -= exit_fee

                self.capital += pos.size_usd + pnl
                self.total_pnl += pnl
                reward = pnl / pos.size_usd
                del self.positions[asset]
                trade_executed = True

        return reward, trade_executed

    def _compute_reward(self, trade_reward: float) -> float:
        reward = trade_reward
        if self.capital < self.peak_capital:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            reward -= self.drawdown_penalty * dd
        reward -= self.trade_penalty * self.trade_count
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.capital = self.initial_capital
        self.positions = {}
        self.current_step = 0
        self.peak_capital = self.initial_capital
        self.trade_count = 0
        self.total_pnl = 0.0

        for asset in self.assets:
            self.feature_extractors[asset].reset()
            start_range = max(10, len(self.raw_data[asset]) - self.max_steps - 10)
            self.current_data_idx[asset] = int(self.rng.integers(10, start_range))

        # Warmup
        for _ in range(5):
            for asset in self.assets:
                state = self._get_market_state(asset, self.current_data_idx[asset])
                self.feature_extractors[asset].update(state)
                self.current_data_idx[asset] += 1
            self.current_step += 1

        features = {}
        for asset in self.assets:
            state = self._get_market_state(asset, self.current_data_idx[asset])
            features[asset] = self.feature_extractors[asset].update(state)

        obs = self._get_observation(features)
        return obs, {"capital": self.capital, "step": self.current_step}

    def step(self, action):
        assert self.action_space.contains(action)

        trade_reward, _ = self._execute_action(action)
        self.peak_capital = max(self.peak_capital, self.capital)
        reward = self._compute_reward(trade_reward)

        self.current_step += 1

        terminated = False
        truncated = False

        # Advance all assets
        for asset in self.assets:
            self.current_data_idx[asset] += 1
            if self.current_data_idx[asset] >= len(self.raw_data[asset]) - 1:
                terminated = True

        if self.current_step >= self.max_steps:
            truncated = True

        if self.capital <= 0:
            reward -= 1.0
            terminated = True

        # Close positions at end
        if (terminated or truncated):
            for asset in list(self.positions.keys()):
                pos = self.positions[asset]
                data = self.raw_data[asset]
                idx = min(self.current_data_idx.get(asset, 0), len(data) - 1)
                current_up = data[idx]["up_price"]
                current_down = data[idx]["down_price"]
                exit_price = current_up if pos.side == 1 else current_down
                pnl = (exit_price - pos.entry_price) * pos.shares
                exit_fee = pos.size_usd * self.taker_fee
                pnl -= exit_fee
                self.capital += pos.size_usd + pnl
                self.total_pnl += pnl
                del self.positions[asset]

        features = {}
        for asset in self.assets:
            state = self._get_market_state(asset, self.current_data_idx[asset])
            features[asset] = self.feature_extractors[asset].update(state)

        obs = self._get_observation(features)

        info = {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "step": self.current_step,
            "n_positions": len(self.positions),
        }

        return obs, reward, terminated, truncated, info

    def get_episode_stats(self):
        return {
            "final_capital": self.capital,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "trade_count": self.trade_count,
            "peak_capital": self.peak_capital,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("PolymarketMultiEnv — тест")
    print("=" * 60)

    env = PolymarketMultiEnv(initial_capital=1000.0)
    obs, info = env.reset()

    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space} (9 actions: 3 assets × 3)")
    print(f"Initial capital: ${env.initial_capital}")

    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f"Step {step}: capital=${info['capital']:.2f}, pnl=${info['total_pnl']:.2f}, "
                  f"trades={info['trade_count']}, positions={info['n_positions']}")
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"\nStats: {stats}")
    print(f"Total reward: {total_reward:.4f}")
