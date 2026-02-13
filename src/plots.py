from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from src.data import download_prices
from src.evaluate import compute_returns, portfolio_returns, evaluate_rl


def plot_equity_curve(series_dict, title, outpath):
    plt.figure()
    for name, port_ret in series_dict.items():
        equity = (1 + port_ret).cumprod()
        plt.plot(equity.index, equity.values, label=name)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.tight_layout()

    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()


if __name__ == "__main__":
    tickers = ["SPY", "QQQ", "TLT"]
    prices = download_prices(tickers, start="2020-01-01")
    rets = compute_returns(prices)

    # Split
    train_rets = rets.loc[: "2023-12-31"]
    test_rets = rets.loc["2024-01-01":]

    # Baseline test
    w_eq = np.array([1/3, 1/3, 1/3])
    base_test = portfolio_returns(test_rets, w_eq)

    # RL test
    rl_test = evaluate_rl("reports/ppo_portfolio_train_only", test_rets)

    # Alinhar índices
    base_test = base_test.loc[rl_test.index]

    plot_equity_curve(
        {"Baseline (Test)": base_test, "RL (Test)": rl_test},
        "Equity Curve — Test Period Only",
        "reports/figures/equity_curve_test_only.png",
    )

    print("Saved: reports/figures/equity_curve_test_only.png")
