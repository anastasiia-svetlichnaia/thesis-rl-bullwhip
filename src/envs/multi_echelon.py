import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiEchelonInventoryEnv(gym.Env):
    def __init__(
        self,
        stability_lambda=0.0,
        holding_cost_rate=0.5,
        stockout_cost_rate=20.0,
        include_demand_in_state=True,
        normalize_action_space=True,
        supplier_replenishment=15.0,
    ):
        super().__init__()

        self.include_demand_in_state = include_demand_in_state
        self.normalize_action_space = normalize_action_space

        self.max_order = 50.0
        self.max_steps = 50

        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        self.stability_lambda = stability_lambda
        self.supplier_replenishment = supplier_replenishment

        obs_shape = (3,) if include_demand_in_state else (2,)

        self.observation_space = spaces.Box(
            low=-200,
            high=200,
            shape=obs_shape,
            dtype=np.float32,
        )

        if normalize_action_space:
            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=0.0,
                high=self.max_order,
                shape=(1,),
                dtype=np.float32,
            )

        self.current_step = 0
        self.retailer_inventory = 20.0
        self.supplier_inventory = 50.0
        self.last_customer_demand = 0.0
        self.previous_retailer_order = 0.0

    def _get_obs(self):
        if self.include_demand_in_state:
            return np.array(
                [
                    self.retailer_inventory,
                    self.supplier_inventory,
                    self.last_customer_demand,
                ],
                dtype=np.float32,
            )

        return np.array(
            [
                self.retailer_inventory,
                self.supplier_inventory,
            ],
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.retailer_inventory = float(self.np_random.integers(10, 30))
        self.supplier_inventory = float(self.np_random.integers(40, 80))
        self.last_customer_demand = 0.0
        self.previous_retailer_order = 0.0

        return self._get_obs(), {}

    def step(self, action):
        raw_action = float(action[0])

        if self.normalize_action_space:
            raw_action = np.clip(raw_action, -1.0, 1.0)
            retailer_order = (raw_action + 1.0) / 2.0 * self.max_order
        else:
            retailer_order = np.clip(raw_action, 0.0, self.max_order)

        customer_demand = float(self.np_random.integers(0, 20))
        self.last_customer_demand = customer_demand

        # Step 1 : retailer serves customer demand from available inventory
        fulfilled_customer_demand = min(max(self.retailer_inventory, 0.0), customer_demand)
        unmet_customer_demand = customer_demand - fulfilled_customer_demand

        # Step 2 : retailer places order to supplier
        supplier_shipment = min(max(self.supplier_inventory, 0.0), retailer_order)
        unfulfilled_retailer_order = retailer_order - supplier_shipment

        # Step 3 : updating inventories
        self.retailer_inventory = self.retailer_inventory - fulfilled_customer_demand + supplier_shipment
        self.supplier_inventory = self.supplier_inventory - supplier_shipment + self.supplier_replenishment

        # Step 4 : costs
        retailer_holding_cost = max(self.retailer_inventory, 0.0) * self.holding_cost_rate
        supplier_holding_cost = max(self.supplier_inventory, 0.0) * self.holding_cost_rate
        holding_cost = retailer_holding_cost + supplier_holding_cost

        stockout_cost = unmet_customer_demand * self.stockout_cost_rate

        stability_penalty = self.stability_lambda * abs(
            retailer_order - self.previous_retailer_order
        )

        reward = -(holding_cost + stockout_cost + stability_penalty)

        if self.retailer_inventory <= 0:
            reward -= 50

        reward -= 10 * unmet_customer_demand

        self.previous_retailer_order = retailer_order
        self.current_step += 1

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "demand": customer_demand,
            "order": retailer_order,
            "inventory": self.retailer_inventory,

            "customer_demand": customer_demand,
            "retailer_order": retailer_order,
            "supplier_shipment": supplier_shipment,
            "retailer_inventory": self.retailer_inventory,
            "supplier_inventory": self.supplier_inventory,
            "fulfilled_customer_demand": fulfilled_customer_demand,
            "unmet_demand": unmet_customer_demand,
            "unfulfilled_retailer_order": unfulfilled_retailer_order,

            "holding_cost": holding_cost,
            "retailer_holding_cost": retailer_holding_cost,
            "supplier_holding_cost": supplier_holding_cost,
            "stockout_cost": stockout_cost,
            "stability_penalty": stability_penalty,
        }

        return self._get_obs(), reward, terminated, truncated, info