import numpy as np
import gymnasium as gym
from gymnasium import spaces


class PortfolioEnv(gym.Env):
    """
    Ambiente simples de alocação de portfólio.

    Observação (obs): retornos passados (window x n_assets)
    Ação (action): pesos contínuos (n_assets) que normalizamos para somar 1
    Recompensa (reward): retorno do portfólio no dia seguinte
    """

    metadata = {"render_modes": []}

    def __init__(self, returns: np.ndarray, window: int = 20):
        super().__init__()
        assert returns.ndim == 2, "returns must be (T, n_assets)"
        self.returns = returns
        self.window = window
        self.T, self.n_assets = returns.shape

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window, self.n_assets), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )

        self.t = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = self.window
        obs = self.returns[self.t - self.window : self.t].astype(np.float32)
        return obs, {}

    def step(self, action):
        action = np.array(action, dtype=np.float32)
        action = np.clip(action, 0.0, 1.0)

        s = float(action.sum())
        if s <= 1e-8:
            action = np.ones_like(action) / self.n_assets
        else:
            action = action / s

        # retorno do dia t
        r_t = float(self.returns[self.t] @ action)

        self.t += 1
        terminated = self.t >= self.T
        truncated = False

        obs = self.returns[self.t - self.window : self.t].astype(np.float32)
        info = {"weights": action, "daily_return": r_t}

        return obs, r_t, terminated, truncated, info
