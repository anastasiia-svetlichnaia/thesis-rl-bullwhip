import pandas as pd


def run_episode(env, policy_fn, episode_id=0, scenario_id="simple_test", seed=42):
    rows = []

    obs, info = env.reset(seed=seed)
    done = False
    t = 0

    while not done:
        action = policy_fn(obs, info)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        rows.append({
            "scenario_id": scenario_id,
            "episode_id": episode_id,
            "t": t,

            # common columns
            "demand": step_info.get("demand"),
            "order": step_info.get("order"),
            "inventory": step_info.get("inventory"),
            "reward": reward,

            # service columns
            "fulfilled_demand": step_info.get("fulfilled_demand"),
            "unmet_demand": step_info.get("unmet_demand", 0.0),

            # cost columns
            "holding_cost": step_info.get("holding_cost", 0.0),
            "stockout_cost": step_info.get("stockout_cost", 0.0),
            "stability_penalty": step_info.get("stability_penalty", 0.0),

            # multi-echelon columns
            "customer_demand": step_info.get("customer_demand"),
            "retailer_order": step_info.get("retailer_order"),
            "supplier_shipment": step_info.get("supplier_shipment"),
            "retailer_inventory": step_info.get("retailer_inventory"),
            "supplier_inventory": step_info.get("supplier_inventory"),
            "unfulfilled_retailer_order": step_info.get("unfulfilled_retailer_order"),
        })

        obs = next_obs
        info = step_info
        done = terminated or truncated
        t += 1

    return pd.DataFrame(rows)