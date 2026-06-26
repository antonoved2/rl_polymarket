"""
Backtesting — сравнение RL агента с XGBoost baseline и случайной стратегией.

Использует walk-forward validation для честного сравнения.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from environment import PolymarketEnv


def backtest_strategy(
    strategy_name: str,
    env: PolymarketEnv,
    n_episodes: int = 100,
    seed: int = 42,
    predict_fn=None,
) -> Dict:
    """Тестирует стратегию на среде."""
    results = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        total_reward = 0

        while not done:
            if predict_fn is not None:
                action = predict_fn(obs, env)
            else:
                action = env.action_space.sample()  # случайная

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        stats = env.get_episode_stats()
        stats['total_reward'] = total_reward
        results.append(stats)

    # Агрегация
    capitals = [r['final_capital'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    trades = [r['trade_count'] for r in results]
    returns = [r['total_return_pct'] for r in results]

    summary = {
        'strategy': strategy_name,
        'n_episodes': n_episodes,
        'avg_capital': float(np.mean(capitals)),
        'std_capital': float(np.std(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'std_pnl': float(np.std(pnls)),
        'avg_return_pct': float(np.mean(returns)),
        'std_return_pct': float(np.std(returns)),
        'avg_trades': float(np.mean(trades)),
        'profitable_episodes': sum(1 for p in pnls if p > 0),
        'win_rate': sum(1 for p in pnls if p > 0) / len(pnls),
        'max_return': float(max(returns)),
        'min_return': float(min(returns)),
        'median_return': float(np.median(returns)),
        'sharpe_approx': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
    }
    return summary


def random_strategy(obs, env):
    """Случайная стратегия."""
    return env.action_space.sample()


def buy_and_hold_strategy(obs, env):
    """Стратегия: всегда покупаем UP в начале, держим до конца."""
    if obs[10] == 0.0:  # no position
        return 1  # BUY
    return 0  # HOLD


def momentum_strategy(obs, env):
    """Простая momentum стратегия: покупаем если momentum > 0."""
    if obs[10] != 0.0:  # has position
        return 0  # HOLD
    # momentum в f[3] — нормализован к [-1, 1]
    if obs[3] > 0.1:
        return 1  # BUY
    elif obs[3] < -0.1:
        return 2  # SELL
    return 0  # HOLD


def run_comparison(
    asset: str = "btc",
    data_path: str = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl",
    n_episodes: int = 100,
    seed: int = 42,
    rl_model_path: str = None,
):
    """Запускает сравнение стратегий."""

    print("=" * 70)
    print(f"Backtest Comparison — {asset.upper()}")
    print(f"Episodes: {n_episodes}")
    print("=" * 70)

    strategies = []

    # 1. Случайная стратегия
    print("\n[1/4] Random Strategy...")
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=seed)
    result = backtest_strategy("Random", env, n_episodes, seed, random_strategy)
    strategies.append(result)
    print(f"  Avg Return: {result['avg_return_pct']:.2f}% | "
          f"Win Rate: {result['win_rate']:.1%} | "
          f"Avg P&L: ${result['avg_pnl']:.2f}")

    # 2. Buy & Hold
    print("\n[2/4] Buy & Hold (UP)...")
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=seed)
    result = backtest_strategy("Buy&Hold", env, n_episodes, seed, buy_and_hold_strategy)
    strategies.append(result)
    print(f"  Avg Return: {result['avg_return_pct']:.2f}% | "
          f"Win Rate: {result['win_rate']:.1%} | "
          f"Avg P&L: ${result['avg_pnl']:.2f}")

    # 3. Momentum
    print("\n[3/4] Momentum Strategy...")
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=seed)
    result = backtest_strategy("Momentum", env, n_episodes, seed, momentum_strategy)
    strategies.append(result)
    print(f"  Avg Return: {result['avg_return_pct']:.2f}% | "
          f"Win Rate: {result['win_rate']:.1%} | "
          f"Avg P&L: ${result['avg_pnl']:.2f}")

    # 4. RL Agent
    if rl_model_path:
        print(f"\n[4/4] RL Agent ({rl_model_path})...")
        from stable_baselines3 import PPO
        model = PPO.load(rl_model_path)

        def rl_predict(obs, env):
            action, _ = model.predict(obs, deterministic=True)
            return int(action)

        env = PolymarketEnv(asset=asset, data_path=data_path, seed=seed)
        result = backtest_strategy("RL_PPO", env, n_episodes, seed, rl_predict)
        strategies.append(result)
        print(f"  Avg Return: {result['avg_return_pct']:.2f}% | "
              f"Win Rate: {result['win_rate']:.1%} | "
              f"Avg P&L: ${result['avg_pnl']:.2f}")
    else:
        print("\n[4/4] RL Agent — SKIPPED (no model path)")

    # Итоговая таблица
    print("\n" + "=" * 70)
    print(f"{'Strategy':<15} {'Avg Return':>12} {'Win Rate':>10} {'Avg P&L':>12} {'Sharpe':>8}")
    print("-" * 70)
    for s in strategies:
        print(f"{s['strategy']:<15} {s['avg_return_pct']:>10.2f}% {s['win_rate']:>9.1%} "
              f"${s['avg_pnl']:>10.2f} {s['sharpe_approx']:>7.2f}")
    print("=" * 70)

    # Сохраняем результаты
    results_path = Path(__file__).parent / "models" / f"backtest_{asset}.json"
    with open(results_path, "w") as f:
        json.dump(strategies, f, indent=2)
    print(f"\n[Saved] {results_path}")

    return strategies


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="btc")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--rl-model", type=str, default=None)
    args = parser.parse_args()

    run_comparison(
        asset=args.asset,
        n_episodes=args.n_episodes,
        rl_model_path=args.rl_model,
    )
