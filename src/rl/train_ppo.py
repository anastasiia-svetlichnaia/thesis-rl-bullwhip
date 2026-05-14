from stable_baselines3 import PPO
from src.envs.inventory_env import SimpleInventoryEnv
from src.envs.multi_echelon import MultiEchelonInventoryEnv


def train_ppo(
    env_class=SimpleInventoryEnv,
    stability_lambda=0.0,
    holding_cost_rate=0.5,
    stockout_cost_rate=2.0,
    total_timesteps=100000,
    include_demand_in_state=False,
    normalize_action_space=False,
):

    env = env_class(
        stability_lambda=stability_lambda,
        holding_cost_rate=holding_cost_rate,
        stockout_cost_rate=stockout_cost_rate,
        include_demand_in_state=include_demand_in_state,
        normalize_action_space=normalize_action_space,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps)

    return model

def ppo_policy(model):
    def policy(obs, info):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy