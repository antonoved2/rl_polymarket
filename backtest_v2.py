"""Backtest multi-asset модели."""
import sys, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from environment_v2 import PolymarketMultiEnv
from stable_baselines3 import PPO


def run_backtest(model_path, n_episodes=100, seed=42):
    model = PPO.load(model_path)
    env = PolymarketMultiEnv(initial_capital=1000.0, seed=seed)

    results = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
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

    print(f"\n=== Multi-Asset PPO Backtest ({n_episodes} episodes) ===")
    print(f"Avg Capital: ${np.mean(capitals):.2f} ± {np.std(capitals):.2f}")
    print(f"Avg P&L: ${np.mean(pnls):.2f} ± {np.std(pnls):.2f}")
    print(f"Avg Return: {np.mean(returns):.2f}% ± {np.std(returns):.2f}%")
    print(f"Avg Trades: {np.mean(trades):.1f}")
    print(f"Profitable: {sum(1 for p in pnls if p > 0)}/{len(pnls)}")
    print(f"Max Return: {max(returns):.2f}%")
    print(f"Min Return: {min(returns):.2f}%")
    print(f"Sharpe: {np.mean(returns)/np.std(returns):.2f}" if np.std(returns) > 0 else "Sharpe: N/A")


if __name__ == "__main__":
    run_backtest("models/ppo_multi_steps100000", n_episodes=100)
