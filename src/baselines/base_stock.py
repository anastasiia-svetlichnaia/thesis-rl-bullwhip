def base_stock_policy(S):
    def policy(obs, info):
        inventory = obs[0]

        order = S - inventory

        # no negative orders
        if order < 0:
            order = 0

        return [order]

    return policy