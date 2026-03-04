import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import os

def load_elbo_values(filepath):
    values = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("Mean"):
                continue
            values.append(float(line))
    return np.array(values)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    flow_vals = load_elbo_values(os.path.join(base_dir, "elbo_values_flow.txt"))
    gauss_vals = load_elbo_values(os.path.join(base_dir, "elbo_values_gaussian.txt"))
    mog_vals = load_elbo_values(os.path.join(base_dir, "elbo_values_mog.txt"))

    models = [
        ("Flow Prior", flow_vals, "#1f77b4"),
        ("Gaussian Prior", gauss_vals, "#ff7f0e"),
        ("MoG Prior", mog_vals, "#2ca02c"),
    ]

    # Build x range covering all data
    all_vals = np.concatenate([flow_vals, gauss_vals, mog_vals])
    x_min = all_vals.min() - 2
    x_max = all_vals.max() + 2
    x = np.linspace(x_min, x_max, 500)

    fig, ax = plt.subplots(figsize=(10, 5))

    for name, vals, color in models:
        mu, std = vals.mean(), vals.std()
        pdf = norm.pdf(x, mu, std)
        ax.plot(x, pdf, label=f"{name} ($\\mu$={mu:.2f}, $\\sigma$={std:.2f})", color=color, linewidth=2)
        ax.fill_between(x, pdf, alpha=0.2, color=color)

    ax.set_xlabel("ELBO", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Distribution of ELBO Values for Three Models", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "elbo_distributions.png"), dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
