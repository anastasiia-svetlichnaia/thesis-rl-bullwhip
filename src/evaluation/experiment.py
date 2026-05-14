from src.envs.inventory_env import SimpleInventoryEnv
from src.envs.multi_echelon import MultiEchelonInventoryEnv
from src.evaluation.run_episode import run_episode
from src.evaluation.metrics import bullwhip_effect


def run_policy_experiment(
    policy_name,
    policy_fn,
    seed=42,
    scenario_id="default",
    env_class=SimpleInventoryEnv,
    stability_lambda=0.0,
    holding_cost_rate=0.5,
    stockout_cost_rate=2.0,
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

    df = run_episode(
        env=env,
        policy_fn=policy_fn,
        episode_id=0,
        scenario_id=scenario_id,
        seed=seed,
    )

    bwe_result = bullwhip_effect(df["order"], df["demand"])

    total_demand = df["demand"].sum()
    total_unmet_demand = df["unmet_demand"].sum()

    if total_demand == 0:
        service_level = float("nan")
    else:
        service_level = 1 - (total_unmet_demand / total_demand)

    service_level = max(0.0, min(1.0, service_level))

    metrics = {
        "policy": policy_name,
        "scenario_id": scenario_id,
        "stability_lambda": stability_lambda,
        "bwe": bwe_result["bwe"],
        "order_variance": bwe_result["order_variance"],
        "demand_variance": bwe_result["demand_variance"],
        "total_reward": df["reward"].sum(),
        "total_holding_cost": df["holding_cost"].sum(),
        "total_stockout_cost": df["stockout_cost"].sum(),
        "total_stability_penalty": df["stability_penalty"].sum(),
        "avg_inventory": df["inventory"].mean(),
        "service_level": service_level,
        "holding_cost_rate": holding_cost_rate,
        "stockout_cost_rate": stockout_cost_rate,
        "total_demand": total_demand,
        "total_unmet_demand": total_unmet_demand,
        "service_level": service_level,
        "include_demand_in_state": include_demand_in_state,
        "normalize_action_space": normalize_action_space,
    }

    return df, metrics