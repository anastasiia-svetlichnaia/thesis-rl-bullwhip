import matplotlib.pyplot as plt


def plot_demand_vs_order(df, title="Demand vs Order", save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(df["t"], df["demand"], label="Demand")
    plt.plot(df["t"], df["order"], label="Order")
    plt.xlabel("Time")
    plt.ylabel("Quantity")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()