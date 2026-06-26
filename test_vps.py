#!/usr/bin/env python3
"""Quick test for VPS deployment."""
import sys
sys.path.insert(0, "/opt/rl_trader")

from rl_live_trader import FeatureExtractor, MarketDataFetcher, TradeLogger
from stable_baselines3 import PPO
import numpy as np

print("=== RL Live Trader Test ===")

# Test 1: Feature Extractor
print("\n[1] Feature Extractor...")
fe = FeatureExtractor()
for i in range(10):
    f = fe.update(0.5 + i*0.01, 0.5 - i*0.01, 60000.0, ta_data={})
assert f.shape == (45,), f"Expected 45 features, got {f.shape}"
print(f"  OK: {f.shape}")

# Test 2: Market Data
print("\n[2] Market Data Fetcher...")
mf = MarketDataFetcher("btc")
pd = mf.get_current_period()
print(f"  Found {len(pd)} periods")
for p, info in pd.items():
    up = info.get("up_price", 0)
    down = info.get("down_price", 0)
    print(f"  Period {p}: UP={up:.4f} DOWN={down:.4f}")
    assert up > 0 and down > 0, "Prices should be positive"

# Test 3: Binance + TA
print("\n[3] Binance Klines + TA...")
klines = mf.get_binance_klines()
if klines:
    closes = [k["c"] for k in klines]
    ta = mf.compute_ta(closes)
    print(f"  Got {len(klines)} klines, {len(ta)} TA indicators")
    assert len(ta) > 20, "Should have 25 TA indicators"
else:
    print("  No klines (API might be slow)")

# Test 4: Model Loading
print("\n[4] PPO Model Loading...")
model = PPO.load("/opt/rl_trader/model.zip")
print(f"  Model loaded: {type(model.policy).__name__}")

# Test 5: Full Observation
print("\n[5] Full Observation...")
obs = fe.update(0.55, 0.45, 60000.0, ta_data=ta)
obs[14] = 0.5  # time remaining
print(f"  Observation shape: {obs.shape}")
assert obs.shape == (45,)

# Test 6: Model Prediction
print("\n[6] Model Prediction...")
action, _ = model.predict(obs, deterministic=True)
print(f"  Action: {action} (0=HOLD, 1=BUY_UP, 2=BUY_DOWN)")
assert action in [0, 1, 2], f"Invalid action: {action}"

# Test 7: Trade Logger
print("\n[7] Trade Logger...")
tl = TradeLogger("/opt/rl_trader/trade_logs")
test_trade = {
    "timestamp": 1719300000,
    "period": 1719298200,
    "side": "UP",
    "entry_price": 0.55,
    "exit_price": 0.60,
    "shares": 100.0,
    "size_usd": 55.0,
    "pnl": 5.0,
    "capital_after": 1005.0,
}
tl.log_trade(test_trade)
stats = tl.get_total_stats()
print(f"  Total trades: {stats['total_trades']}")
print(f"  Win rate: {stats['win_rate']*100:.0f}%")
print(f"  Total PnL: ${stats['total_pnl']:.2f}")

print("\n=== ALL TESTS PASSED ===")
