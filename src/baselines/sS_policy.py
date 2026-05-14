def sS_policy(s, S):
    def policy(obs, info):
        inventory = obs[0]

        if inventory <= s:
            order = S - inventory
        else:
            order = 0

        if order < 0:
            order = 0

        return [order]

    return policy