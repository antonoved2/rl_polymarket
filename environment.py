"""
PolymarketEnv — Gymnasium environment для RL-агента.

Среда симулирует торговлю на Polymarket 15-минутных бинарных рынках.
Агент наблюдает за рынком, принимает решения (HOLD/BUY/SELL),
и получает награду на основе реального P&L.

Данные: исторические снапшоты Polymarket + Binance.
Эпизод: N шагов внутри одного 15-минутного периода или
         полный период до резолва.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass, field


# ===== Константы =====
ASSETS = ["btc", "eth", "sol"]
TIMESTEPS_PER_PERIOD = 90  # 90 шагов × 10 сек ≈ 15 мин
TICK_INTERVAL_SEC = 10     # интервал между шагами в симуляции

# Polymarket 2026 fee structure (taker fee)
# При 50/50 odds: ~3.15% (0.0315)
# При 90/10 odds: ~0.75% (0.0075)
TAKER_FEE_RATE = 0.025     # консервативная оценка 2.5% средний тейкер фик
MAKER_FEE_RATE = 0.0       # maker fee = 0 на Polymarket

# Размер позиции (% от капитала)
POSITION_SIZE_PCT = 0.10   # 10% от капитала на сделку
MAX_POSITIONS = 1          # максимум 1 открытая позиция одновременно


@dataclass
class Position:
    """Открытая позиция."""
    side: int           # 1 = UP (YES), -1 = DOWN (NO)
    entry_price: float  # цена входа (0-1)
    size_usd: float     # размер позиции в USD
    shares: float       # количество токенов
    entry_step: int     # шаг входа


@dataclass
class MarketState:
    """Состояние рынка на один шаг."""
    timestamp: int
    period_start: int
    up_price: float
    down_price: float
    up_bid: float = 0.0
    up_ask: float = 0.0
    down_bid: float = 0.0
    down_ask: float = 0.0
    binance_price: float = 0.0
    binance_return_1m: float = 0.0
    binance_return_5m: float = 0.0
    volatility_5m: float = 0.0
    orderbook_imbalance: float = 0.0


class FeatureExtractor:
    """Извлекает нормализованные фичи из рыночных данных."""

    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.price_history: List[float] = []
        self.return_history: List[float] = []

    def update(self, state: MarketState) -> np.ndarray:
        """Обновляет историю и возвращает 14 нормализованных фичей."""
        mid_price = float(state.up_price)
        self.price_history.append(mid_price)
        max_len = self.lookback + 1
        if len(self.price_history) > max_len:
            self.price_history = self.price_history[len(self.price_history) - max_len:]

        # Returns
        if len(self.price_history) >= 2:
            ret = self.price_history[-1] - self.price_history[-2]
            self.return_history.append(ret)
            if len(self.return_history) > self.lookback:
                self.return_history = self.return_history[len(self.return_history) - self.lookback:]

        features = np.zeros(14, dtype=np.float32)

        # 1. UP price (нормализовано к [0,1] — уже вероятность)
        features[0] = np.clip(state.up_price, 0.0, 1.0)

        # 2. DOWN price
        features[1] = np.clip(state.down_price, 0.0, 1.0)

        # 3. UP+DOWN spread (mispricing: >0 = YES переоценен, <0 = недооценен)
        features[2] = np.clip(state.up_price + state.down_price - 1.0, -0.1, 0.1) * 10.0

        # 4. Price momentum (5 шагов)
        if len(self.price_history) >= 6:
            features[3] = (self.price_history[-1] - self.price_history[-6]) * 10.0
        features[3] = np.clip(features[3], -1.0, 1.0)

        # 5. Price momentum (все доступные шаги)
        if len(self.price_history) >= 2:
            features[4] = (self.price_history[-1] - self.price_history[0]) * 5.0
        features[4] = np.clip(features[4], -1.0, 1.0)

        # 6. Bid-ask spread
        spread = (state.up_ask - state.up_bid) if state.up_ask > state.up_bid else 0.02
        features[5] = np.clip(spread * 20.0, 0.0, 1.0)  # нормализуем: 5% spread = 1.0

        # 7. Binance return 1m
        features[6] = np.clip(state.binance_return_1m * 100.0, -1.0, 1.0)

        # 8. Binance return 5m
        features[7] = np.clip(state.binance_return_5m * 100.0, -1.0, 1.0)

        # 9. Volatility (5m)
        if len(self.return_history) >= 3:
            features[8] = np.clip(np.std(self.return_history) * 100.0, 0.0, 1.0)
        else:
            features[8] = 0.0

        # 10. Time to expiry (1.0 = начало, 0.0 = конец)
        elapsed = state.timestamp - state.period_start
        remaining = max(0, 900 - elapsed)
        features[9] = remaining / 900.0

        # 11. Has position
        features[10] = 0.0  # будет установлен в environment

        # 12. Position side
        features[11] = 0.0  # будет установлен в environment

        # 13. Position P&L (нереализованный)
        features[12] = 0.0  # будет установлен в environment

        # 14. Order book imbalance
        features[13] = np.clip(state.orderbook_imbalance, -1.0, 1.0)

        return features

    def reset(self):
        self.price_history.clear()
        self.return_history.clear()


class PolymarketEnv(gym.Env):
    """
    Gymnasium environment для торговли Polymarket 15m бинарными рынками.

    Action space: {HOLD=0, BUY=1, SELL=2}
    Observation space: 14 нормализованных float32 фичей
    Reward: %P&L - drawdown_penalty - trade_count_penalty
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
        drawdown_penalty: float = 0.1,
        trade_penalty: float = 0.001,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.asset = asset
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.taker_fee = taker_fee
        self.max_steps = max_steps_per_episode
        self.drawdown_penalty = drawdown_penalty
        self.trade_penalty = trade_penalty
        self.rng = np.random.default_rng(seed)

        # Загрузка данных
        self.raw_data = self._load_data(data_path, asset)
        self.feature_extractor = FeatureExtractor(lookback=5)

        # Определение spaces
        # Action: HOLD(0), BUY(1), SELL(2)
        self.action_space = spaces.Discrete(3)

        # Observation: 14 фичей в [-1, 1]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )

        # Состояние эпизода
        self.capital: float = 0.0
        self.position: Optional[Position] = None
        self.current_step: int = 0
        self.current_data_idx: int = 0
        self.peak_capital: float = 0.0
        self.trade_count: int = 0
        self.total_pnl: float = 0.0
        self.episode_history: List[Dict] = []

    def _load_data(self, path: str, asset: str) -> List[Dict]:
        """Загружает и фильтрует данные по активу."""
        data = []
        with open(path) as f:
            for line in f:
                snap = json.loads(line.strip())
                # Фильтруем рынки по активу
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

        # Сортируем по timestamp
        data.sort(key=lambda x: x["timestamp"])
        print(f"[Env] Loaded {len(data)} snapshots for {asset}")
        return data

    def _get_market_state(self, idx: int) -> MarketState:
        """Создаёт MarketState из сырых данных."""
        if idx >= len(self.raw_data):
            idx = len(self.raw_data) - 1

        d = self.raw_data[idx]
        state = MarketState(
            timestamp=d["timestamp"],
            period_start=d["period_start"],
            up_price=d["up_price"],
            down_price=d["down_price"],
            binance_price=d["binance_price"],
        )

        # Вычисляем returns из binance цены
        if idx > 0:
            prev_price = self.raw_data[idx - 1]["binance_price"]
            if prev_price > 0:
                state.binance_return_1m = (d["binance_price"] - prev_price) / prev_price

        if idx >= 5:
            prev_price = self.raw_data[idx - 5]["binance_price"]
            if prev_price > 0:
                state.binance_return_5m = (d["binance_price"] - prev_price) / prev_price

        # Volatility из последних 10 шагов
        if idx >= 10:
            prices = [self.raw_data[i]["binance_price"] for i in range(idx - 10, idx + 1)]
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices)) if prices[i-1] > 0]
            if returns:
                state.volatility_5m = float(np.std(returns))

        return state

    def _get_observation(self, features: np.ndarray) -> np.ndarray:
        """Добавляет position-dependent фичи к observation."""
        obs = features.copy()

        # 11. Has position
        obs[10] = 1.0 if self.position is not None else 0.0

        # 12. Position side
        if self.position is not None:
            obs[11] = float(self.position.side)  # 1.0 или -1.0
        else:
            obs[11] = 0.0

        # 13. Position P&L
        if self.position is not None:
            current_up = self.raw_data[self.current_data_idx]["up_price"]
            current_down = self.raw_data[self.current_data_idx]["down_price"]
            if self.position.side == 1:  # UP
                current_price = current_up
            else:  # DOWN
                current_price = current_down
            unrealized = (current_price - self.position.entry_price) * self.position.shares
            obs[12] = np.clip(unrealized / self.position.size_usd, -1.0, 1.0)
        else:
            obs[12] = 0.0

        return obs

    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """
        Выполняет действие, возвращает (reward, trade_executed).
        """
        reward = 0.0
        trade_executed = False
        idx = self.current_data_idx

        if action == 0:  # HOLD
            pass

        elif action == 1:  # BUY (ставим на UP)
            if self.position is None:
                up_price = self.raw_data[idx]["up_price"]
                if up_price > 0.01 and up_price < 0.99:  # не покупаем при крайних ценах
                    size_usd = self.capital * self.position_size_pct
                    shares = size_usd / up_price if up_price > 0 else 0
                    fee = size_usd * self.taker_fee
                    self.position = Position(
                        side=1,
                        entry_price=up_price,
                        size_usd=size_usd,
                        shares=shares,
                        entry_step=self.current_step,
                    )
                    self.capital -= fee  # вычитаем fee
                    self.trade_count += 1
                    trade_executed = True

        elif action == 2:  # SELL (ставим на DOWN)
            if self.position is None:
                down_price = self.raw_data[idx]["down_price"]
                if down_price > 0.01 and down_price < 0.99:
                    size_usd = self.capital * self.position_size_pct
                    shares = size_usd / down_price if down_price > 0 else 0
                    fee = size_usd * self.taker_fee
                    self.position = Position(
                        side=-1,
                        entry_price=down_price,
                        size_usd=size_usd,
                        shares=shares,
                        entry_step=self.current_step,
                    )
                    self.capital -= fee
                    self.trade_count += 1
                    trade_executed = True

        # Закрытие позиции (если позиция открыта и прошло достаточно шагов)
        if self.position is not None:
            steps_held = self.current_step - self.position.entry_step
            if steps_held >= 5:  # минимум 5 шагов держим
                close_action = action  # любое действие закрывает после минимума
                current_up = self.raw_data[idx]["up_price"]
                current_down = self.raw_data[idx]["down_price"]

                if self.position.side == 1:  # UP position
                    exit_price = current_up
                    pnl = (exit_price - self.position.entry_price) * self.position.shares
                else:  # DOWN position
                    exit_price = current_down
                    pnl = (exit_price - self.position.entry_price) * self.position.shares

                # Вычитаем fee на выходе
                exit_fee = self.position.size_usd * self.taker_fee
                pnl -= exit_fee

                self.capital += self.position.size_usd + pnl
                self.total_pnl += pnl
                reward = pnl / self.position.size_usd  # % return
                self.position = None
                trade_executed = True

        return reward, trade_executed

    def _compute_reward(self, trade_reward: float) -> float:
        """Итоговая reward function с штрафами."""
        reward = trade_reward

        # Штраф за drawdown
        if self.capital < self.peak_capital:
            dd = (self.peak_capital - self.capital) / self.peak_capital
            reward -= self.drawdown_penalty * dd

        # Штраф за частые сделки (fees!)
        reward -= self.trade_penalty * self.trade_count

        return reward

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Сброс эпизода."""
        super().reset(seed=seed)

        self.capital = self.initial_capital
        self.position = None
        self.current_step = 0
        self.peak_capital = self.initial_capital
        self.trade_count = 0
        self.total_pnl = 0.0
        self.episode_history = []
        self.feature_extractor.reset()

        # Случайный стартовый индекс (но с запасом для lookback)
        start_range = max(10, len(self.raw_data) - self.max_steps - 10)
        self.current_data_idx = self.rng.integers(10, start_range)

        # Пропускаем первые шаги для накопления истории
        for _ in range(5):
            state = self._get_market_state(self.current_data_idx)
            self.feature_extractor.update(state)
            self.current_data_idx += 1
            self.current_step += 1

        # Начальное наблюдение
        state = self._get_market_state(self.current_data_idx)
        features = self.feature_extractor.update(state)
        obs = self._get_observation(features)

        info = {
            "capital": self.capital,
            "step": self.current_step,
            "data_idx": self.current_data_idx,
        }
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Один шаг в среде."""
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Выполняем действие
        trade_reward, trade_executed = self._execute_action(action)

        # Обновляем peak capital
        self.peak_capital = max(self.peak_capital, self.capital)

        # Вычисляем reward
        reward = self._compute_reward(trade_reward)

        # Записываем в историю
        self.episode_history.append({
            "step": self.current_step,
            "action": action,
            "capital": self.capital,
            "reward": reward,
            "trade_executed": trade_executed,
            "has_position": self.position is not None,
        })

        # Переходим к следующему шагу
        self.current_step += 1
        self.current_data_idx += 1

        # Проверяем конец эпизода
        terminated = False
        truncated = False

        # Конец данных
        if self.current_data_idx >= len(self.raw_data) - 1:
            terminated = True

        # Достигли максимального числа шагов
        if self.current_step >= self.max_steps:
            truncated = True

        # Банкротство
        if self.capital <= 0:
            reward -= 1.0  # большой штраф за банкротство
            terminated = True

        # Закрываем позицию при конце эпизода
        if (terminated or truncated) and self.position is not None:
            idx = min(self.current_data_idx, len(self.raw_data) - 1)
            current_up = self.raw_data[idx]["up_price"]
            current_down = self.raw_data[idx]["down_price"]
            if self.position.side == 1:
                exit_price = current_up
            else:
                exit_price = current_down
            pnl = (exit_price - self.position.entry_price) * self.position.shares
            exit_fee = self.position.size_usd * self.taker_fee
            pnl -= exit_fee
            self.capital += self.position.size_usd + pnl
            self.total_pnl += pnl
            self.position = None

        # Новое наблюдение
        state = self._get_market_state(self.current_data_idx)
        features = self.feature_extractor.update(state)
        obs = self._get_observation(features)

        info = {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "step": self.current_step,
            "has_position": self.position is not None,
        }

        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Простой рендер состояния."""
        if mode == "human":
            pos_str = f"POS({'UP' if self.position and self.position.side == 1 else 'DOWN'})" if self.position else "FLAT"
            print(
                f"Step {self.current_step:3d} | "
                f"Capital: ${self.capital:8.2f} | "
                f"P&L: ${self.total_pnl:8.2f} | "
                f"Trades: {self.trade_count:3d} | "
                f"{pos_str}"
            )

    def get_episode_stats(self) -> Dict[str, Any]:
        """Статистика эпизода для логирования."""
        return {
            "final_capital": self.capital,
            "total_pnl": self.total_pnl,
            "total_return_pct": (self.capital - self.initial_capital) / self.initial_capital * 100,
            "trade_count": self.trade_count,
            "peak_capital": self.peak_capital,
            "max_drawdown_pct": (self.peak_capital - self.capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0,
            "steps": self.current_step,
        }


# ===== Тест среды =====
if __name__ == "__main__":
    print("=" * 60)
    print("PolymarketEnv — тест среды")
    print("=" * 60)

    env = PolymarketEnv(asset="btc", initial_capital=1000.0)
    obs, info = env.reset()

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation: {obs}")
    print(f"Action space: {env.action_space}")
    print(f"Initial capital: ${env.initial_capital}")
    print(f"Data points: {len(env.raw_data)}")

    # Случайные действия для теста
    total_reward = 0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            env.render()
        if terminated or truncated:
            break

    stats = env.get_episode_stats()
    print(f"\n{'=' * 60}")
    print(f"Episode stats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  total_reward: {total_reward:.4f}")
    print(f"{'=' * 60}")
