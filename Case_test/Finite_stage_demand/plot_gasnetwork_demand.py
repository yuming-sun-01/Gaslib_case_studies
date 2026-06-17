"""Plot Test gas network schematic and cyclic demand profile."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (import_data_from_excel, graph_construction,
                   plot_graph_with_layout, dynamic_demand_profile)

# utils.py sets text.usetex=True at module level; override so matplotlib's
# built-in mathtext handles $\mathrm{...}$ without needing a latex binary
plt.rcParams["text.usetex"] = False

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
NETWORK_FILE = os.path.join(BASE_DIR, "test", "networkData.xlsx")
INPUT_FILE   = os.path.join(BASE_DIR, "test", "inputData.xlsx")

NUM_TIME_PERIODS = 1
TIME_LENGTH      = 7    # 0-6 hrs
T_MAX            = 6.0


def plot_demand_profile(inputData, out_file="plot_demand.pdf"):
    wcons = inputData["wCons"]

    ss_demand = None
    sink_name = None
    for key, vol_dict in sorted(wcons.items()):
        if key.startswith("sink") and 0 in vol_dict:
            ss_demand = float(list(vol_dict[0].values())[0])
            sink_name = key
            break

    if ss_demand is None:
        print("No sink demand data found.")
        return

    ss_arr = np.array([ss_demand] * TIME_LENGTH)
    dyn = dynamic_demand_profile(ss_arr, num_time_periods=NUM_TIME_PERIODS)
    t = np.linspace(0, T_MAX, TIME_LENGTH)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t, dyn, 'b-', linewidth=2.0)
    ax.set_xlabel("Time (hrs)", fontsize=16)
    ax.set_ylabel("Demand (kg/s)", fontsize=16)
    ax.set_title("Cyclic Demand", fontsize=18)
    ax.set_xlim(0, T_MAX)
    margin = (dyn.max() - dyn.min()) * 0.1
    ax.set_ylim(dyn.min() - margin, dyn.max() + margin)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(BASE_DIR, out_file)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Saved {out_path}")
    print(f"  Sink: {sink_name}, ss={ss_demand:.4f} kg/s, "
          f"range=[{dyn.min():.4f}, {dyn.max():.4f}]")


def main():
    os.chdir(BASE_DIR)
    networkData, inputData = import_data_from_excel(NETWORK_FILE, INPUT_FILE)
    G = graph_construction(networkData)

    # Uses plot_graph_with_layout from utils.py (saves gaslib40_schematic.pdf)
    # edge_labels=False: the positions in utils are hardcoded for Gaslib40 node indices
    # rename to test_schematic.pdf afterwards
    plot_graph_with_layout(G, node_labels=True, edge_labels=False)
    src = os.path.join(BASE_DIR, "gaslib40_schematic.pdf")
    dst = os.path.join(BASE_DIR, "test_schematic.pdf")
    os.replace(src, dst)
    print(f"Saved {dst}")

    plot_demand_profile(inputData, "plot_demand.pdf")
    plt.show()


if __name__ == "__main__":
    main()
