from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent

HARD_FILE = "final_paper_enmpc_explicit_terminal_each_point_stability_10_cycles.xlsx"
SOFT_FILE = "final_paper_enmpc_explicit_terminal_each_point_stability_72hrs_soft.xlsx"
OCSS_FILE = "optimal_css_24hrs_gaslib40_extended.xlsx"

X_MAX_HOURS = 72
DPI = 220

RUNS = [
    ("Hard terminal", HARD_FILE, "#1f77b4", "-"),
    ("Soft terminal", SOFT_FILE, "#d62728", "-"),
    ("OCSS steady", OCSS_FILE, "black", ":"),
]


def read_sheet(filename, sheet):
    df = pd.read_excel(HERE / filename, sheet_name=sheet, index_col=0)
    return df.iloc[:X_MAX_HOURS + 1].reset_index(drop=True)


def entity_name(col):
    return col.split("'")[1]


def sort_by_trailing_int(cols):
    def key(col):
        try:
            return int(entity_name(col).split("_")[-1])
        except Exception:
            return 0
    return sorted(cols, key=key)


def common_columns(data, key, contains):
    sets = [
        {c for c in run_data[key].columns if contains in c}
        for run_data in data.values()
    ]
    return sort_by_trailing_int(set.intersection(*sets))


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "legend.framealpha": 0.9,
    })


def plot_grid(data, key, cols, ylabel, title, output_name, ncols=2):
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 2.8 * nrows),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, col in zip(axes.flat, cols):
        for label, _, color, linestyle in RUNS:
            df = data[label][key]
            t = np.arange(len(df))
            ax.plot(t, df[col], color=color, linestyle=linestyle, lw=1.8, label=label)
        ax.set_xlim(0, X_MAX_HOURS)
        ax.set_title(entity_name(col), fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.flat[len(cols):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (h)")

    fig.suptitle(title, fontsize=14)
    out = HERE / output_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out}")


def plot_source_grid(data, specs, title, output_name, ncols=2):
    nrows = int(np.ceil(len(specs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 2.8 * nrows),
        sharex=True, constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for ax, (key, col, panel_title, ylabel) in zip(axes.flat, specs):
        for label, _, color, linestyle in RUNS:
            df = data[label][key]
            t = np.arange(len(df))
            ax.plot(t, df[col], color=color, linestyle=linestyle, lw=1.8, label=label)
        ax.set_xlim(0, X_MAX_HOURS)
        ax.set_title(panel_title, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)

    for ax in axes.flat[len(specs):]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (h)")

    fig.suptitle(title, fontsize=14)
    out = HERE / output_name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out}")


def write_summary(data, p_cols, source_specs):
    ref_p = data["OCSS steady"]["P"][p_cols]
    rows = []

    for label, filename, _, _ in RUNS:
        p = data[label]["P"][p_cols]
        source_errors = []
        for key, col, _, _ in source_specs:
            source_errors.append(abs(data[label][key][col] - data["OCSS steady"][key][col]))
        rows.append({
            "case": label,
            "file": filename,
            "total_compressor_P_sum": float(p.iloc[:-1].to_numpy().sum()),
            "mean_abs_P_error_to_ocss": float((p - ref_p).abs().mean().mean()),
            "mean_abs_source_error_to_ocss": float(pd.concat(source_errors, axis=1).mean().mean()),
        })

    out = HERE / "gaslib40_P_wSource_hard_soft_ocss_summary.xlsx"
    pd.DataFrame(rows).to_excel(out, sheet_name="summary", index=False)
    print(f"Saved: {out}")


def main():
    setup_style()
    data = {
        label: {
            "P": read_sheet(filename, "compressor power"),
            "wSource": read_sheet(filename, "wSource"),
            "pSource": read_sheet(filename, "pSource"),
        }
        for label, filename, _, _ in RUNS
    }

    p_cols = common_columns(data, "P", "compressor_P")
    wsource_cols = common_columns(data, "wSource", "wSource")
    psource_cols = common_columns(data, "pSource", "pSource")
    psource_by_name = {entity_name(c): c for c in psource_cols}
    source_specs = []
    for col in wsource_cols:
        name = entity_name(col)
        if name == "source_3":
            source_specs.append(("pSource", psource_by_name[name], "source_3 pSource", "pSource"))
        else:
            source_specs.append(("wSource", col, f"{name} wSource", "wSource"))

    plot_grid(
        data, "P", p_cols,
        ylabel="Compressor P",
        title="Gaslib40 compressor P: hard terminal vs soft terminal vs OCSS",
        output_name="gaslib40_P_hard_soft_ocss_compare.png",
        ncols=2,
    )
    plot_source_grid(
        data, source_specs,
        title="Gaslib40 sources: wSource for source_1/source_2, pSource for source_3",
        output_name="gaslib40_wSource_hard_soft_ocss_compare.png",
        ncols=2,
    )
    write_summary(data, p_cols, source_specs)


if __name__ == "__main__":
    main()
