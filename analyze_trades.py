#!/usr/bin/env python3
"""
Analyze PPO v6 model trade patterns.
Runs the model on historical data and collects detailed statistics.
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from stable_baselines3 import PPO
from environment_v3 import PolymarketEnvV3, N_FEATURES

FEATURE_NAMES = [
    "up_price", "down_price", "spread_x10", "momentum_6s", "cum_return",
    "spread_norm", "momentum_6s_2", "abs_return", "big_move", "return_accel",
    "binance_ret_1m", "binance_ret_5m", "volatility", "volatility_5m",
    "time_remaining",
    "has_position", "position_side", "unrealized_pnl",
    "regime_trend", "regime_vol",
    "ma_cross_5_20", "ma_cross_10_20", "ma_cross_ema_12_26",
    "price_vs_sma20", "price_vs_ema50",
    "rsi", "macd_line", "macd_signal", "macd_hist",
    "bb_width", "bb_pct_b", "bb_upper", "bb_lower",
    "atr_pct", "stoch_k", "stoch_d",
    "vol_ratio", "obv",
    "momentum_5", "momentum_10",
    "sma_5", "sma_10", "sma_20", "ema_12", "ema_26",
]


def analyze_model(model_path, asset="btc", n_episodes=200):
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    all_entries = []  # entries (when model opens a position)
    all_holds = []    # holds (when model decides not to enter)
    all_data = []     # all timesteps

    for ep in range(n_episodes):
        env = PolymarketEnvV3(
            data_path="/opt/rl_trader/data/expanded_snapshots.jsonl",
            asset=asset,
            initial_capital=1000.0,
            position_size_pct=0.10,
            taker_fee=0.025,
            max_steps_per_episode=90,
            min_hold_steps=3,
            seed=ep * 7 + 13,
        )

        obs, _ = env.reset(seed=ep * 7 + 13)
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)

            idx = env.current_data_idx
            raw = env.raw_data[idx]
            elapsed = raw["timestamp"] - raw.get("period_start", raw["timestamp"])

            entry_data = {
                "episode": ep,
                "step": step,
                "action": action,
                "obs": obs.copy(),
                "up_price": raw["up_price"],
                "down_price": raw["down_price"],
                "elapsed": elapsed,
                "has_position": env.position is not None,
                "capital": env.capital,
            }
            all_data.append(entry_data)

            # Entry: action != 0 and no position before
            if action != 0 and not env.position:
                all_entries.append(entry_data)

            # Hold: action == 0 and no position
            if action == 0 and not env.position:
                all_holds.append(entry_data)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1

    # ═══════════════════════════════════════════════════════════════════
    # Analysis
    # ═══════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print(f"  TRADE ANALYSIS - PPO v6 ({n_episodes} episodes)")
    print(f"  Total timesteps: {len(all_data)}")
    print(f"  Entries: {len(all_entries)}")
    print(f"  Holds (no pos): {len(all_holds)}")
    print(f"{'='*60}")

    # 1. Action distribution
    actions = [d["action"] for d in all_data]
    hold_c = actions.count(0)
    up_c = actions.count(1)
    down_c = actions.count(2)
    sell_c = actions.count(3)
    total = len(actions)

    print(f"\n📊 ACTION DISTRIBUTION (all timesteps):")
    print(f"  HOLD:     {hold_c:6d} ({hold_c/total*100:.1f}%)")
    print(f"  BUY_UP:   {up_c:6d} ({up_c/total*100:.1f}%)")
    print(f"  BUY_DOWN: {down_c:6d} ({down_c/total*100:.1f}%)")
    print(f"  SELL:     {sell_c:6d} ({sell_c/total*100:.1f}%)")

    # 2. Entry direction
    sell_actions = [d for d in all_data if d["action"] == 3]
    
    if all_entries:
        up_entries = [e for e in all_entries if e["action"] == 1]
        down_entries = [e for e in all_entries if e["action"] == 2]

        print(f"\n📈 ENTRY DIRECTION:")
        print(f"  BUY_UP:   {len(up_entries)} ({len(up_entries)/len(all_entries)*100:.1f}%)")
        print(f"  BUY_DOWN: {len(down_entries)} ({len(down_entries)/len(all_entries)*100:.1f}%)")
        
        print(f"\n🔴 SELL ACTION (model-initiated exits):")
        print(f"  Total SELL: {sell_actions}")
        if sell_actions:
            print(f"  SELL with position: {sum(1 for s in sell_actions if s['has_position'])}")
            print(f"  SELL without pos:   {sum(1 for s in sell_actions if not s['has_position'])}")

        # 3. Feature analysis - entries vs holds
        print(f"\n🔍 FEATURE VALUES: ENTRY vs HOLD (top discriminators):")

        entry_obs = np.array([e["obs"] for e in all_entries])
        hold_obs = np.array([h["obs"] for h in all_holds]) if all_holds else np.zeros((1, N_FEATURES))
        all_obs = np.array([d["obs"] for d in all_data])

        # Remove position features (15-17) from analysis - they're state-dependent
        analysis_features = [i for i in range(N_FEATURES) if i not in [15, 16, 17]]

        differences = []
        for i in analysis_features:
            all_mean = np.mean(all_obs[:, i])
            all_std = np.std(all_obs[:, i])
            entry_mean = np.mean(entry_obs[:, i]) if len(entry_obs) > 0 else 0
            hold_mean = np.mean(hold_obs[:, i]) if len(hold_obs) > 0 else 0

            if all_std > 0:
                # How different are entries from the average timestep
                diff = (entry_mean - all_mean) / all_std
            else:
                diff = 0
            differences.append((i, diff, all_mean, entry_mean, hold_mean))

        differences.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"  {'Feature':20s} {'Entry':>10s} {'Hold':>10s} {'All':>10s} {'Diff':>8s}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*8}")
        for i, diff, all_m, entry_m, hold_m in differences[:20]:
            name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
            marker = " ◀" if abs(diff) > 0.3 else ""
            print(f"  {name:20s} {entry_m:+10.3f} {hold_m:+10.3f} {all_m:+10.3f} {diff:+7.2f}σ{marker}")

        # 4. Entry price analysis
        print(f"\n💰 ENTRY PRICE ANALYSIS:")
        up_prices = [e["up_price"] for e in up_entries]
        down_prices = [e["down_price"] for e in down_entries]

        if up_prices:
            print(f"  BUY_UP entries (n={len(up_prices)}):")
            print(f"    UP price:   mean={np.mean(up_prices):.3f}  std={np.std(up_prices):.3f}  "
                  f"min={np.min(up_prices):.3f}  max={np.max(up_prices):.3f}")
            # Price buckets
            buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            counts = [sum(1 for p in up_prices if buckets[j] <= p < buckets[j+1]) for j in range(len(buckets)-1)]
            print(f"    Price buckets: ", end="")
            for j in range(len(buckets)-1):
                print(f"[{buckets[j]:.1f}-{buckets[j+1]:.1f})={counts[j]} ", end="")
            print()

        if down_prices:
            print(f"  BUY_DOWN entries (n={len(down_prices)}):")
            print(f"    DOWN price: mean={np.mean(down_prices):.3f}  std={np.std(down_prices):.3f}  "
                  f"min={np.min(down_prices):.3f}  max={np.max(down_prices):.3f}")
            buckets = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            counts = [sum(1 for p in down_prices if buckets[j] <= p < buckets[j+1]) for j in range(len(buckets)-1)]
            print(f"    Price buckets: ", end="")
            for j in range(len(buckets)-1):
                print(f"[{buckets[j]:.1f}-{buckets[j+1]:.1f})={counts[j]} ", end="")
            print()

        # 5. Entry timing
        print(f"\n⏱️  ENTRY TIMING (seconds into 15-min period):")
        elapsed_times = [e["elapsed"] for e in all_entries]
        print(f"  Mean:   {np.mean(elapsed_times):.0f}s")
        print(f"  Median: {np.median(elapsed_times):.0f}s")
        print(f"  Min:    {np.min(elapsed_times):.0f}s")
        print(f"  Max:    {np.max(elapsed_times):.0f}s")

        q1 = sum(1 for t in elapsed_times if t < 225)
        q2 = sum(1 for t in elapsed_times if 225 <= t < 450)
        q3 = sum(1 for t in elapsed_times if 450 <= t < 675)
        q4 = sum(1 for t in elapsed_times if t >= 675)
        print(f"  Q1 (0-3.75m):   {q1:4d} ({q1/len(elapsed_times)*100:.1f}%)")
        print(f"  Q2 (3.75-7.5m): {q2:4d} ({q2/len(elapsed_times)*100:.1f}%)")
        print(f"  Q3 (7.5-11.25m):{q3:4d} ({q3/len(elapsed_times)*100:.1f}%)")
        print(f"  Q4 (11.25-15m): {q4:4d} ({q4/len(elapsed_times)*100:.1f}%)")

        # 6. Key indicator values at entry
        print(f"\n📐 KEY INDICATORS AT ENTRY:")

        def print_stats(vals, name, pct=False):
            if not vals:
                print(f"  {name}: N/A")
                return
            if pct:
                print(f"  {name:20s}: mean={np.mean(vals)*100:.1f}%  std={np.std(vals)*100:.1f}%  "
                      f"min={np.min(vals)*100:.1f}%  max={np.max(vals)*100:.1f}%")
            else:
                print(f"  {name:20s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                      f"min={np.min(vals):.3f}  max={np.max(vals):.3f}")

        # RSI
        rsi_vals = [e["obs"][24] for e in all_entries]
        print_stats(rsi_vals, "RSI (norm)")

        # MACD hist
        macd_vals = [e["obs"][28] for e in all_entries]
        print_stats(macd_vals, "MACD hist")

        # BB pct_b
        bb_vals = [e["obs"][30] for e in all_entries]
        print_stats(bb_vals, "BB %B")

        # Vol ratio
        vol_vals = [e["obs"][36] for e in all_entries]
        print_stats(vol_vals, "Volume ratio")

        # Momentum 5
        mom_vals = [e["obs"][38] for e in all_entries]
        print_stats(mom_vals, "Momentum 5")

        # MA cross 5/20
        ma_vals = [e["obs"][20] for e in all_entries]
        print_stats(ma_vals, "MA cross 5/20")

        # 7. Entries by side - different patterns?
        print(f"\n🔄 ENTRY PATTERNS BY SIDE:")

        for side_name, side_entries in [("BUY_UP", up_entries), ("BUY_DOWN", down_entries)]:
            if not side_entries:
                continue
            print(f"\n  {side_name} (n={len(side_entries)}):")

            # RSI
            rsi = [e["obs"][24] for e in side_entries]
            print(f"    RSI:        mean={np.mean(rsi):.3f}  (0.5=neutral, >0.7=overbought, <0.3=oversold)")

            # MACD
            macd = [e["obs"][28] for e in side_entries]
            print(f"    MACD hist:  mean={np.mean(macd):+.4f}  ({'bullish' if np.mean(macd) > 0 else 'bearish'})")

            # Momentum
            mom = [e["obs"][38] for e in side_entries]
            print(f"    Momentum 5: mean={np.mean(mom):+.4f}  ({'rising' if np.mean(mom) > 0 else 'falling'})")

            # MA cross
            ma = [e["obs"][20] for e in side_entries]
            print(f"    MA 5/20:    mean={np.mean(ma):+.4f}  ({'above' if np.mean(ma) > 0 else 'below'})")

            # BB %B
            bb = [e["obs"][30] for e in side_entries]
            print(f"    BB %B:      mean={np.mean(bb):.3f}  ({'upper band' if np.mean(bb) > 0.7 else 'lower band' if np.mean(bb) < 0.3 else 'middle'})")

            # Time
            times = [e["elapsed"] for e in side_entries]
            print(f"    Entry time: mean={np.mean(times):.0f}s  ({'early' if np.mean(times) < 300 else 'late'} in period)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/opt/rl_trader/models/ppo_v5_btc_steps500000"
    analyze_model(model_path, n_episodes=200)
