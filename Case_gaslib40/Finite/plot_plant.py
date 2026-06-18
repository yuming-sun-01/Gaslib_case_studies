"""Plot compressor beta and interm_p from MPC_test_results.xlsx against
optimal_css_24hrs_test_extended.xlsx on a common time axis.
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

# Configuration
TRANSPARENT = True
FONTSIZE = 16
LINEWIDTH = 2.0

plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = 'serif'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MPC_FILE = os.path.join(OUT_DIR, "final_paper_enmpc_explicit_terminal_each_point_stability_72hrs_soft.xlsx")
CSS_FILE = os.path.join(OUT_DIR, "optimal_css_24hrs_gaslib40_extended.xlsx")
T_PERIOD = 24.0   # CSS reference is 24-hour periodic

LW_MPC = LINEWIDTH
LW_CSS = LINEWIDTH


def _tile_to_horizon(df, T_period, t_max):
    """Tile a periodic CSS DataFrame so its index covers [0, t_max]."""
    if df.index.max() >= t_max:
        return df
    # Take one full period from the start of the file
    base = df[df.index < T_period].copy()
    if base.empty:
        return df
    n_periods = int(t_max // T_period) + 2
    tiled = []
    for k in range(n_periods):
        shifted = base.copy()
        shifted.index = shifted.index + k * T_period
        tiled.append(shifted)
    out = pd.concat(tiled)
    out = out[out.index <= t_max]
    return out


def _load(sheet):
    mpc = pd.read_excel(MPC_FILE, sheet_name=sheet, index_col=0)
    css = pd.read_excel(CSS_FILE, sheet_name=sheet, index_col=0)
    css = _tile_to_horizon(css, T_PERIOD, float(mpc.index.max()))
    return mpc, css


def plot_beta():
    mpc, css = _load("compressor beta")
    cols = list(mpc.columns)
    n = len(cols)

    cmap = plt.get_cmap("tab10" if n <= 10 else "tab20")
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, cols)):
        color = cmap(i % cmap.N)
        ax.plot(mpc.index, mpc[col].to_numpy(), color=color,
                lw=LW_MPC, label=r"$\mathrm{MPC}$", zorder=2)
        if col in css.columns:
            ax.plot(css.index, css[col].to_numpy(),
                    color="k", lw=LW_CSS, linestyle=':',
                    label=r"$\mathrm{CSS\ (extended)}$", zorder=5)
        short = (col.replace("compressor_beta[", "").replace(", :]", "").replace("'", ""))
        ax.set_ylabel(f"$\\mathrm{{{short}}}$")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")

    axes[-1].set_xlabel(r"$\mathrm{Time\ (hrs)}$")
    fig.suptitle(r"$\mathrm{Compressor\ \beta:\ MPC\ vs\ CSS\ (extended)}$", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_beta.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")


def _pipe_key(col):
    m = re.search(r"'(pipe_\d+)'", col)
    return m.group(1) if m else col


def plot_interm_p():
    mpc, css = _load("interm_p")
    cols = list(mpc.columns)

    pipes = {}
    for c in cols:
        pipes.setdefault(_pipe_key(c), []).append(c)
    pipe_names = sorted(pipes.keys(),
                        key=lambda s: int(re.search(r"\d+", s).group()))
    n = len(pipe_names)

    seg_colors = ['salmon', 'b', 'brown', 'g', 'purple']
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.8 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, pipe in zip(axes, pipe_names):
        for i, col in enumerate(pipes[pipe]):
            seg = re.search(r"',\s*(\d+)", col).group(1)
            c = seg_colors[i % len(seg_colors)]
            ax.plot(mpc.index, mpc[col].to_numpy(), color=c, lw=LW_MPC,
                    label=f"$\\mathrm{{MPC\\ seg\\ {seg}}}$", zorder=2)
            if col in css.columns:
                ax.plot(css.index, css[col].to_numpy(), color="black",
                        lw=LW_CSS, linestyle=':',
                        label=f"$\\mathrm{{CSS\\ seg\\ {seg}}}$", zorder=5)
        ax.set_ylabel(f"$\\mathrm{{{pipe}\\ interm_p}}$")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc="best", ncol=2)

    axes[-1].set_xlabel(r"$\mathrm{Time\ (hrs)}$")
    fig.suptitle(r"$\mathrm{Pipe\ interm_p:\ MPC\ vs\ CSS\ (extended)}$", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_interm_p.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")


def plot_sources():
    """Plot the derived (free) boundary variable per source:
       source_1, source_2 → wSource (pSource is the active fix)
       source_3           → pSource (wSource is the active fix)
    """
    mpc_w, css_w = _load("wSource")
    mpc_p, css_p = _load("pSource")

    # which sheet to use per source name
    sheet_map = {
        "source_1": (mpc_w, css_w, "wSource"),
        "source_2": (mpc_w, css_w, "wSource"),
        "source_3": (mpc_p, css_p, "pSource"),
    }

    panels = []  # list of (source_name, attr_name, mpc_col, css_col, mpc_df, css_df)
    for src, (mpc_df, css_df, attr) in sheet_map.items():
        for col in mpc_df.columns:
            if src in col.replace("'", ""):
                panels.append((src, attr, col, col, mpc_df, css_df))
                break

    n = len(panels)
    if n == 0:
        print("[plot_plant] no source columns matched; skipping plot_sources")
        return

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, (src, attr, mc, cc, mdf, cdf)) in enumerate(zip(axes, panels)):
        color = cmap(i % cmap.N)
        ax.plot(mdf.index, mdf[mc].to_numpy(), color=color,
                lw=LW_MPC, label=r"$\mathrm{MPC}$", zorder=2)
        if cc in cdf.columns:
            ax.plot(cdf.index, cdf[cc].to_numpy(),
                    color="k", lw=LW_CSS, linestyle=':',
                    label=r"$\mathrm{CSS\ (extended)}$", zorder=5)
        ax.set_ylabel(f"$\\mathrm{{{attr}\\ [{src}]}}$".replace("_", r"\_"))
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")

    axes[-1].set_xlabel(r"$\mathrm{Time\ (hrs)}$")
    fig.suptitle(r"$\mathrm{Source\ boundaries:\ MPC\ vs\ CSS\ (extended)}$", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_sources.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")


def main():
    plot_beta()
    plot_interm_p()
    plot_sources()
    plt.show()


if __name__ == "__main__":
    main()
