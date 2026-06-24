"""Plot compressor power, interm_p, and wSource from MPC against CSS."""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from infNMPC_options import _import_settings

# Configuration
TRANSPARENT = True
FONTSIZE = 16
LINEWIDTH = 2.0

plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = 'serif'

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTS = _import_settings()
MPC_FILE = os.path.join(OUT_DIR, "MPC_test_results.xlsx")
CSS_FILE = str(OPTS.ocss_file_path)
T_PERIOD = float(OPTS.T_period)

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


def plot_compressor_P():
    mpc, css = _load("compressor power")
    cols = sorted(mpc.columns, key=lambda c: int(re.search(r'\d+', c).group()))
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
        short = (col.replace("compressor_P[", "").replace(", :]", "").replace("'", ""))
        ax.set_ylabel(f"$\\mathrm{{{short}\\ P}}$")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")

    axes[-1].set_xlabel(r"$\mathrm{Time\ (hrs)}$")
    fig.suptitle(r"$\mathrm{Compressor\ P:\ MPC\ vs\ CSS\ (extended)}$", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_P.pdf")
    plt.savefig(out, bbox_inches="tight")
    png_out = out.replace(".pdf", ".png")
    plt.savefig(png_out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")
    print(f"[plot_plant] saved {png_out}")


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
        ax.set_ylabel(f"$\\mathrm{{{pipe}\\ interm\\_p\\ (bar)}}$")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=10, loc="best", ncol=2)

    axes[-1].set_xlabel(r"$\mathrm{Time\ (hrs)}$")
    fig.suptitle(r"$\mathrm{Pipe\ interm_p:\ MPC\ vs\ CSS\ (extended)}$", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_interm_p.pdf")
    plt.savefig(out, bbox_inches="tight")
    png_out = out.replace(".pdf", ".png")
    plt.savefig(png_out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")
    print(f"[plot_plant] saved {png_out}")


def plot_sources():
    """Plot wSource for all 6 sources."""
    mpc, css = _load("wSource")
    cols = sorted(mpc.columns, key=lambda c: int(re.search(r'\d+', c).group()))
    n = len(cols)
    if n == 0:
        print("[plot_plant] no wSource columns found; skipping plot_sources")
        return

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, (ax, col) in enumerate(zip(axes, cols)):
        src = re.search(r"'(source_\d+)'", col).group(1)
        color = cmap(i % cmap.N)
        ax.plot(mpc.index, mpc[col].to_numpy(), color=color,
                lw=LW_MPC, label="MPC", zorder=2)
        if col in css.columns:
            ax.plot(css.index, css[col].to_numpy(),
                    color="k", lw=LW_CSS, linestyle=':',
                    label="CSS (extended)", zorder=5)
        ax.set_ylabel(f"wSource [{src}]")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time (hrs)")
    fig.suptitle("Source wSource: MPC vs CSS (extended)", y=1.002)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, "plot_plant_sources.pdf")
    plt.savefig(out, bbox_inches="tight")
    png_out = out.replace(".pdf", ".png")
    plt.savefig(png_out, bbox_inches="tight")
    print(f"[plot_plant] saved {out}")
    print(f"[plot_plant] saved {png_out}")


def main():
    plot_compressor_P()
    plot_interm_p()
    plot_sources()
    plt.show()


if __name__ == "__main__":
    main()
