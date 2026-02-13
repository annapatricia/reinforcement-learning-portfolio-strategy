import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.data import download_prices
from src.env_portfolio import PortfolioEnv


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    pr = returns.values @ w
    return pd.Series(pr, index=returns.index)


def sharpe_ratio(port_ret: pd.Series) -> float:
    if port_ret.std() == 0:
        return float("nan")
    return float(np.sqrt(252) * port_ret.mean() / port_ret.std())


def max_drawdown(port_ret: pd.Series) -> float:
    equity = (1 + port_ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def evaluate_rl(model_path: str, returns_df: pd.DataFrame, window: int = 20) -> pd.Series:
    env = PortfolioEnv(returns_df.values, window=window)
    model = PPO.load(model_path)

    obs, _ = env.reset()
    done = False
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated

    idx = returns_df.index[window: window + len(rewards)]
    return pd.Series(rewards, index=idx)


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "TLT"]
    prices = download_prices(tickers, start="2020-01-01")
    rets = compute_returns(prices)

    # Split
    train_rets = rets.loc[: "2023-12-31"]
    test_rets = rets.loc["2024-01-01":]

    print("Test period:", test_rets.index.min(), "to", test_rets.index.max())

    # Baseline on test
    w_eq = np.array([1/3, 1/3, 1/3])
    base_test = portfolio_returns(test_rets, w_eq)

    sr_base = sharpe_ratio(base_test)
    mdd_base = max_drawdown(base_test)

    # RL on test
    rl_test = evaluate_rl("reports/ppo_portfolio_train_only", test_rets)

    sr_rl = sharpe_ratio(rl_test)
    mdd_rl = max_drawdown(rl_test)

    print("\n=== TEST RESULTS ===")
    print("Baseline Sharpe:", round(sr_base, 3))
    print("RL Sharpe:", round(sr_rl, 3))
    print("Baseline MaxDD:", round(mdd_base, 3))
    print("RL MaxDD:", round(mdd_rl, 3))
