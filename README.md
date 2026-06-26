# RL Polymarket — Reinforcement Learning Trading Agent

## Архитектура

```
PolymarketEnv (Gymnasium)
├── State (14 features) — нормализованные
├── Action: HOLD(0) / BUY(1) / SELL(2)
├── Reward: %P&L - drawdown_penalty - fee_penalty
└── Episode: один 15-минутный период (или N шагов внутри)
```

## Данные
- Polymarket Gamma API: UP/DOWN цены (bid/ask/mid)
- Binance WebSocket: spot price, returns, order flow
- Chainlink: resolution source (для симуляции)

## Фичи (14)
1. up_price — текущая UP цена (mid)
2. down_price — текущая DOWN цена (mid)
3. up_down_spread — up + down - 1.0 (mispricing)
4. price_momentum_5m — изменение цены за 5 мин
5. price_momentum_10m — изменение цены за 10 мин
6. bid_ask_spread — спред на Polymarket
7. binance_return_1m — доходность за 1 мин
8. binance_return_5m — доходность за 5 мин
9. volatility_5m — волатильность за 5 мин
10. time_to_expiry — время до резолва (нормализованное)
11. has_position — есть ли позиция (0/1)
12. position_side — сторона позиции (-1/0/1)
13. position_pnl — нереализованный P&L
14. prob_model — оценка вероятности (Black-Scholes)

## Reward Function
```
reward = realized_pnl_pct - 0.1 * max_drawdown - 0.001 * n_trades
```

## Обучение
- PPO (Stable-Baselines3)
- 1M шагов на симуляции
- Walk-forward валидация
- Paper trading → Live

## Структура
```
rl_polymarket/
├── README.md
├── environment.py     # Gymnasium env
├── features.py         # Feature extraction
├── reward.py           # Reward function
├── train.py            # PPO training script
├── backtest.py         # Backtesting
├── evaluate.py         # Evaluation on out-of-sample
└── requirements.txt
```
