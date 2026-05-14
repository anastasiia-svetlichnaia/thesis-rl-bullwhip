import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleInventoryEnv(gym.Env):
    def __init__(
            self,
            stability_lambda=0.0,
            holding_cost_rate=0.5,
            stockout_cost_rate=2.0,
            include_demand_in_state=False,
            normalize_action_space=False,
        ):
        super().__init__()

        self.include_demand_in_state = include_demand_in_state
        self.normalize_action_space=normalize_action_space

        self.max_order = 50.0

        obs_shape = (2,) if include_demand_in_state else (1,)

        self.observation_space = spaces.Box(low=-100, high=100, shape=obs_shape, dtype=np.float32)
        if normalize_action_space:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        else:
            self.action_space = spaces.Box(
                low=0.0,
                high=self.max_order,
                shape=(1,),
                dtype=np.float32
            )

        self.max_steps = 50
        self.current_step = 0
        self.inventory = 20
        self.last_demand = 0

        self.stability_lambda = stability_lambda
        self.previous_order = 0.0
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate

    def _get_obs(self):
        if self.include_demand_in_state:
            return np.array([self.inventory, self.last_demand], dtype=np.float32)
        return np.array([self.inventory], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.inventory = self.np_random.integers(5, 30)
        self.current_step = 0
        self.previous_order = 0.0
        self.last_demand = 0
        return self._get_obs(), {}

    def step(self, action):
        raw_action = float(action[0])

        if self.normalize_action_space:
            raw_action = np.clip(raw_action, -1.0, 1.0)
            order = (raw_action + 1.0) / 2.0 * self.max_order
        else:
            order = np.clip(raw_action, 0.0, self.max_order)

        demand = self.np_random.integers(0, 20)
        self.last_demand = demand

        # demand is served from current inventory before new order arrives
        available_inventory = self.inventory

        fulfilled_demand = min(max(available_inventory, 0), demand)
        unmet_demand = demand - fulfilled_demand

        # updating inventory after demand and order
        self.inventory = self.inventory + order - fulfilled_demand

        holding_cost = max(self.inventory, 0) * self.holding_cost_rate
        stockout_cost = unmet_demand * self.stockout_cost_rate

        stability_penalty = self.stability_lambda * abs(order - self.previous_order)

        reward = -(holding_cost + stockout_cost + stability_penalty)

        if self.inventory < 0:
            reward -= 50
        reward -= 10 * unmet_demand

        self.previous_order = order
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "demand": demand,
            "inventory": self.inventory,
            "order": order,
            "fulfilled_demand": fulfilled_demand,
            "unmet_demand": unmet_demand,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "stability_penalty": stability_penalty,
        }

        return self._get_obs(), reward, terminated, truncated, info