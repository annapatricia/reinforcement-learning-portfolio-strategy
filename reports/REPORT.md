# RL Portfolio Agent — MVP Results

## Objective
Build a simple offline Reinforcement Learning (PPO) agent to allocate a portfolio across 3 assets (SPY, QQQ, TLT) using historical daily returns.

## Baseline
Equal-weight portfolio (1/3 each asset).

## Metrics
- Sharpe Ratio (annualized)
- Max Drawdown

## Results (local MVP)

- Baseline Sharpe: 0.719
- Baseline Max Drawdown: -30.06%
- RL Sharpe: 0.977
- RL Max Drawdown: -28.18%

## Plots
![Equity Curve Baseline vs RL](figures/equity_curve_baseline_vs_rl.png)

## Notes / Limitations
- This MVP uses a simplified reward: next-day portfolio return.
- No transaction costs, slippage, or turnover penalty yet.
- No train/test split yet (next step).
