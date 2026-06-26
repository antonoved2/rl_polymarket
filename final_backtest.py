"""
Финальный бэктест — сравнение всех обученных моделей.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from environment import PolymarketEnv
from stable_baselines3 import PPO


def backtest_model(model, asset, n_episodes=300, seed=42):
    """Тестирует модель на большом количестве эпизодов."""
    data_path = "/home/antonov5/.openclaw/workspace/data_collector/data/expanded/expanded_snapshots.jsonl"
    results = []

    for ep in range(n_episodes):
        env = PolymarketEnv(asset=asset, data_path=data_path, seed=seed + ep)
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        stats = env.get_episode_stats()
        results.append(stats)

        if (ep + 1) % 50 == 0:
            pnls_so_far = [r['total_pnl'] for r in results]
            wins_so_far = sum(1 for p in pnls_so_far if p > 0)
            print(f"  Episode {ep+1}/{n_episodes} | WR: {wins_so_far/len(results)*100:.1f}%")

    pnls = [r['total_pnl'] for r in results]
    returns = [r['total_return_pct'] for r in results]
    trades = [r['trade_count'] for r in results]
    wins = sum(1 for p in pnls if p > 0)

    return {
        'avg_return_pct': float(np.mean(returns)),
        'std_return_pct': float(np.std(returns)),
        'avg_pnl': float(np.mean(pnls)),
        'std_pnl': float(np.std(pnls)),
        'win_rate': wins / len(pnls),
        'avg_trades': float(np.mean(trades)),
        'sharpe': float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0,
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'median_return': float(np.median(returns)),
        'n_episodes': n_episodes,
    }


def main():
    models = [
        ("ppo_btc_steps50000", "btc", "PPO v1 BTC (50K)"),
        ("ppo_v3_btc_steps150000", "btc", "PPO v3 BTC (150K)"),
        ("ppo_v3_eth_steps150000", "eth", "PPO v3 ETH (150K)"),
        ("ppo_v3_sol_steps150000", "sol", "PPO v3 SOL (150K)"),
        ("ppo_eth_steps50000", "eth", "PPO v1 ETH (50K)"),
        ("ppo_sol_steps50000", "sol", "PPO v1 SOL (50K)"),
    ]

    all_results = []

    for model_name, asset, display_name in models:
        model_path = f"models/{model_name}"
        if not Path(model_path).exists():
            print(f"[SKIP] {display_name} — модель не найдена")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {display_name}")
        print(f"{'='*60}")

        model = PPO.load(model_path)
        result = backtest_model(model, asset, n_episodes=300)
        result['name'] = display_name
        result['asset'] = asset
        all_results.append(result)

        print(f"\n  Win Rate: {result['win_rate']*100:.1f}%")
        print(f"  Avg Return: {result['avg_return_pct']:.2f}%")
        print(f"  Avg P&L: ${result['avg_pnl']:.2f}")
        print(f"  Sharpe: {result['sharpe']:.2f}")

    # Итоговая таблица
    print(f"\n{'='*80}")
    print(f"{'Model':<25} {'Asset':<5} {'WR':>8} {'Avg Ret':>10} {'P&L':>12} {'Sharpe':>8}")
    print(f"{'-'*80}")
    for r in all_results:
        print(f"{r['name']:<25} {r['asset']:<5} {r['win_rate']*100:>7.1f}% "
              f"{r['avg_return_pct']:>9.2f}% ${r['avg_pnl']:>10.2f} {r['sharpe']:>7.2f}")
    print(f"{'='*80}")

    # Сохраняем
    with open("models/final_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] models/final_comparison.json")


if __name__ == "__main__":
    main()
