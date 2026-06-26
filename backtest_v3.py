"""
Расширенный бэктест v3 — детальный анализ всех моделей.

Тестирует все обученные модели на большом количестве эпизодов
с walk-forward валидацией.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from environment import PolymarketEnv
from environment_v2 import PolymarketMultiEnv
from stable_baselines3 import PPO


def backtest_model(model, env_class, model_name, n_episodes=200, seed=42, **env_kwargs):
    """Тестирует модель на среде."""
    results = []

    for ep in range(n_episodes):
        env = env_class(seed=seed + ep, **env_kwargs)
        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        stats = env.get_episode_stats()
        results.append(stats)

    capitals = [r['final_capital'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    trades = [r['trade_count'] for r in results]
    returns = [r['total_return_pct'] for r in results]
    wins = [1 for p in pnls if p > 0]

    summary = {
        'model': model_name,
        'n_episodes': n_episodes,
        'avg_capital': float(np.mean(capitals)),
        'std_capital': float(np.std(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'std_pnl': float(np.std(pnls)),
        'avg_return_pct': float(np.mean(returns)),
        'std_return_pct': float(np.std(returns)),
        'avg_trades': float(np.mean(trades)),
        'win_rate': float(len(wins) / len(pnls)),
        'max_return': float(max(returns)),
        'min_return': float(min(returns)),
        'median_return': float(np.median(returns)),
        'sharpe': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
    }
    return summary


def compare_strategies(asset, model_path, n_episodes=200):
    """Сравнивает все стратегии для одного актива."""
    from backtest import random_strategy, buy_and_hold_strategy, momentum_strategy

    data_path = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl"
    strategies = []

    # 1. Random
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=42)
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            action = random_strategy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats = env.get_episode_stats()
        results.append(stats)
    strategies.append(aggregate_results("Random", results))

    # 2. Buy & Hold
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=42)
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            action = buy_and_hold_strategy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats = env.get_episode_stats()
        results.append(stats)
    strategies.append(aggregate_results("Buy&Hold", results))

    # 3. Momentum
    env = PolymarketEnv(asset=asset, data_path=data_path, seed=42)
    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        while not done:
            action = momentum_strategy(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        stats = env.get_episode_stats()
        results.append(stats)
    strategies.append(aggregate_results("Momentum", results))

    # 4. RL PPO (v1)
    if model_path and Path(model_path).exists():
        model = PPO.load(model_path)
        env = PolymarketEnv(asset=asset, data_path=data_path, seed=42)
        results = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                done = terminated or truncated
            stats = env.get_episode_stats()
            results.append(stats)
        strategies.append(aggregate_results("RL_PPO_v1", results))

    return strategies


def aggregate_results(name, results):
    """Агрегирует результаты эпизодов."""
    capitals = [r['final_capital'] for r in results]
    pnls = [r['total_pnl'] for r in results]
    trades = [r['trade_count'] for r in results]
    returns = [r['total_return_pct'] for r in results]

    return {
        'strategy': name,
        'n_episodes': len(results),
        'avg_capital': float(np.mean(capitals)),
        'std_capital': float(np.std(capitals)),
        'avg_pnl': float(np.mean(pnls)),
        'std_pnl': float(np.std(pnls)),
        'avg_return_pct': float(np.mean(returns)),
        'std_return_pct': float(np.std(returns)),
        'avg_trades': float(np.mean(trades)),
        'win_rate': float(sum(1 for p in pnls if p > 0) / len(pnls)),
        'max_return': float(max(returns)),
        'min_return': float(min(returns)),
        'median_return': float(np.median(returns)),
        'sharpe': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
    }


def print_comparison(strategies):
    """Красивый вывод сравнения."""
    print("\n" + "=" * 80)
    print(f"{'Strategy':<15} {'Avg Return':>12} {'Win Rate':>10} {'Avg P&L':>12} {'Sharpe':>8} {'Trades':>8}")
    print("-" * 80)
    for s in strategies:
        print(f"{s['strategy']:<15} {s['avg_return_pct']:>10.2f}% {s['win_rate']:>9.1%} "
              f"${s['avg_pnl']:>10.2f} {s['sharpe']:>7.2f} {s['avg_trades']:>7.1f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=str, default="btc")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model or f"models/ppo_{args.asset}_steps50000"
    print(f"\nBacktest comparison for {args.asset.upper()}")
    print(f"Model: {model_path}")
    print(f"Episodes: {args.n_episodes}")

    strategies = compare_strategies(args.asset, model_path, args.n_episodes)
    print_comparison(strategies)

    # Сохраняем
    out_path = Path("models") / f"backtest_v3_{args.asset}.json"
    with open(out_path, "w") as f:
        json.dump(strategies, f, indent=2)
    print(f"\n[Saved] {out_path}")
