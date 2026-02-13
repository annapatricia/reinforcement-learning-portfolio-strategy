from stable_baselines3 import PPO

from src.data import download_prices
from src.evaluate import compute_returns
from src.env_portfolio import PortfolioEnv


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "TLT"]
    prices = download_prices(tickers, start="2020-01-01")
    rets = compute_returns(prices)

    # Train split (até final de 2023)
    train_rets = rets.loc[: "2023-12-31"]

    env = PortfolioEnv(train_rets.values, window=20)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=30_000)

    model.save("reports/ppo_portfolio_train_only")
    print("Saved model: reports/ppo_portfolio_train_only")
