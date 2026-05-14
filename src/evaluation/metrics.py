from __future__ import annotations

import numpy as np
import pandas as pd


def bullwhip_effect(order_series: pd.Series, demand_series: pd.Series, ddof: int = 1):
    """
    Returns bullwhip effect and underlying variances
    """
    order_var = np.var(order_series, ddof=ddof)
    demand_var = np.var(demand_series, ddof=ddof)

    if demand_var == 0:
        bwe = float("nan")
    else:
        bwe = float(order_var / demand_var)

    return {
        "bwe": bwe,
        "order_variance": float(order_var),
        "demand_variance": float(demand_var),
    }