"""
plot_paper.py — paper-ready E-NMPC comparison plot for the Gaslib40 network.

Same data and layout as plot.py, but tailored for publication:
    - no subplot titles
    - each panel labelled (a) (b) (c) (d) (e) (f) (row-major)
    - no CPU time / cost annotation under the panels

Total cost and CPU time per scenario are written to `results.txt`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

X_MAX_HOURS = None

EXCLUDE_XLSX = {"css.xlsx", "obj.xlsx"}

OCSS_FILE          = "css.xlsx"
OCSS_CYCLE_HOURS   = 24
OCSS_DROP_LAST     = 1

LEFT_SOURCES  = ["source_1", "source_2"]
RIGHT_SOURCES = ["source_3"]

FLOW_SCALE     = 1.0
PRESSURE_SCALE = 1.0
POWER_SCALE    = 100.0

OUTPUT_PNG  = "enmpc_comparison_paper.png"
OUTPUT_TXT  = "results.txt"
DPI         = 200

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)",
                "(g)", "(h)", "(i)", "(j)", "(k)", "(l)"]

# Times New Roman, falling back to a Times look-alike when it is not installed.
# Detect at runtime so machines that have Times New Roman use it (no warnings
# on machines that don't).
import matplotlib.font_manager as fm
_AVAILABLE = {f.name for f in fm.fontManager.ttflist}
PAPER_FONT = [f for f in ["Times New Roman", "STIXGeneral", "DejaVu Serif"]
              if f in _AVAILABLE] or ["serif"]


def scenario_sort_key(path):
    name = path.name
    if name.startswith("f_enmpc_cycle_"):
        try:
            return (0, int(path.stem.rsplit("_", 1)[-1]), name)
        except ValueError:
            return (0, 999, name)
    if name.startswith("Inf_enmpc"):
        return (1, 0, name)
    return (2, 0, name)


COLUMNS = [
    (p.name, None)
    for p in sorted(
        (x for x in HERE.glob("*.xlsx") if x.name not in EXCLUDE_XLSX),
        key=scenario_sort_key,
    )
]


def compute_cycle_costs(pwr_df, pwr_cols, cycle_hours, dt=3600, scale_factor=1000):
    arr = pwr_df[pwr_cols].iloc[:-1].values
    n = len(arr)
    cycles = []
    for k in range(n // cycle_hours):
        block = arr[k * cycle_hours : (k + 1) * cycle_hours]
        cycles.append(float(block.sum()) * dt / scale_factor)
    return cycles


def compute_cost_kwh(pwr_df, pwr_cols, dt=3600, scale_factor=1000):
    return sum(compute_cycle_costs(pwr_df, pwr_cols, OCSS_CYCLE_HOURS, dt=dt, scale_factor=scale_factor))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_run(path):
    src_w = pd.read_excel(path, sheet_name="wSource")
    src_p = pd.read_excel(path, sheet_name="pSource")
    pwr   = pd.read_excel(path, sheet_name="compressor power")
    beta  = pd.read_excel(path, sheet_name="compressor beta")
    for df in (src_w, src_p, pwr, beta):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return src_w, src_p, pwr, beta


def sort_by_trailing_int(cols):
    def key(c):
        try:
            return int(c.split("'")[1].split("_")[-1])
        except Exception:
            return 0
    return sorted(cols, key=key)


def entity_name(col):
    return col.split("'")[1]


def slice_run(src_w, src_p, pwr, beta, sl):
    if sl is None:
        return src_w, src_p, pwr, beta
    return (src_w.iloc[sl].reset_index(drop=True),
            src_p.iloc[sl].reset_index(drop=True),
            pwr.iloc[sl].reset_index(drop=True),
            beta.iloc[sl].reset_index(drop=True))


def truncate_to_hours(src_w, src_p, pwr, beta, max_hours):
    if max_hours is None:
        return (src_w.reset_index(drop=True),
                src_p.reset_index(drop=True),
                pwr.reset_index(drop=True),
                beta.reset_index(drop=True))
    keep = min(max_hours + 1, len(src_w))
    return (src_w.iloc[:keep].reset_index(drop=True),
            src_p.iloc[:keep].reset_index(drop=True),
            pwr.iloc[:keep].reset_index(drop=True),
            beta.iloc[:keep].reset_index(drop=True))


def tile_ocss(ocss_t, ocss_vals, target_t, cycle_hours):
    phase = np.mod(target_t, cycle_hours)
    t_ext = np.concatenate([ocss_t, [cycle_hours]])
    v_ext = np.concatenate([ocss_vals, [ocss_vals[0]]])
    return np.interp(phase, t_ext, v_ext)


def load_cpu_time_seconds(path, max_steps=None):
    def numeric_iteration_values(df, value_col):
        vals = df.copy()
        if "iteration" in vals.columns:
            vals["iteration"] = pd.to_numeric(vals["iteration"], errors="coerce")
            vals = vals[vals["iteration"].notna()]
            if max_steps is not None:
                vals = vals[(vals["iteration"] >= 0) & (vals["iteration"] < max_steps)]
        elif max_steps is not None:
            vals = vals.iloc[:max_steps]
        return vals[value_col].dropna()

    xl = pd.ExcelFile(path)
    if "runtime" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="runtime")
        if "iteration_time_s" in df.columns and len(df) > 0:
            vals = numeric_iteration_values(df, "iteration_time_s")
            if len(vals) > 0:
                return float(vals.sum())
    if "timing" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="timing")
        if "time_s" in df.columns:
            vals = numeric_iteration_values(df, "time_s")
            if len(vals) > 0:
                return float(vals.sum())
    return None


def format_cpu_time(seconds):
    return f"{seconds:.1f} s"


def add_panel_label(ax, text):
    # Placed just outside (above the top-left corner of) the axes.
    ax.text(-0.08, 1.04, text, transform=ax.transAxes,
            ha="left", va="bottom", fontsize=14, fontweight="bold",
            fontfamily=PAPER_FONT)


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
if not COLUMNS:
    raise FileNotFoundError(f"No scenario xlsx files found in {HERE}")

if X_MAX_HOURS is None:
    X_MAX_HOURS = min(
        len(pd.read_excel(HERE / fn, sheet_name="wSource")) - 1
        for fn, _ in COLUMNS
    )

runs = []
for fn, sl in COLUMNS:
    src_w, src_p, pwr, beta = load_run(HERE / fn)
    src_w, src_p, pwr, beta = slice_run(src_w, src_p, pwr, beta, sl)
    src_w, src_p, pwr, beta = truncate_to_hours(src_w, src_p, pwr, beta, X_MAX_HOURS)
    cpu_seconds = load_cpu_time_seconds(HERE / fn, max_steps=X_MAX_HOURS)
    t = np.arange(len(src_w))
    runs.append((fn, t, src_w, src_p, pwr, beta, cpu_seconds))

ocss_src_w, ocss_src_p, ocss_pwr, ocss_beta = load_run(HERE / OCSS_FILE)
keep = min(OCSS_CYCLE_HOURS, len(ocss_src_w) - OCSS_DROP_LAST)
ocss_src_w = ocss_src_w.iloc[:keep].reset_index(drop=True)
ocss_src_p = ocss_src_p.iloc[:keep].reset_index(drop=True)
ocss_pwr   = ocss_pwr.iloc[:keep].reset_index(drop=True)
ocss_beta  = ocss_beta.iloc[:keep].reset_index(drop=True)
ocss_t     = np.arange(len(ocss_src_w))

wS_cols   = sort_by_trailing_int([c for c in ocss_src_w.columns if "wSource"      in c])
pS_cols   = sort_by_trailing_int([c for c in ocss_src_p.columns if "pSource"      in c])
pwr_cols  = sort_by_trailing_int([c for c in ocss_pwr.columns   if "compressor_P" in c])

wS_by_src = {entity_name(c): c for c in wS_cols}
pS_by_src = {entity_name(c): c for c in pS_cols}
LEFT_SOURCES = [s for s in LEFT_SOURCES if s in wS_by_src]
RIGHT_SOURCES = [s for s in RIGHT_SOURCES if s in pS_by_src]

src_order = list(dict.fromkeys(LEFT_SOURCES + RIGHT_SOURCES))
cmap10    = plt.get_cmap("tab10")
src_color = {name: cmap10(i % 10) for i, name in enumerate(src_order)}
pwr_color = {c: cmap10(i % 10) for i, c in enumerate(pwr_cols)}


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": PAPER_FONT,
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

n_cols = len(runs)
fig, axes = plt.subplots(2, n_cols, figsize=(5.3 * n_cols, 8),
                         constrained_layout=True)
if n_cols == 1:
    axes = axes.reshape(2, 1)

for col_idx, (fn, t, src_w, src_p, pwr, beta, cpu_seconds) in enumerate(runs):
    ax_top = axes[0, col_idx]
    ax_top_r = ax_top.twinx()
    ax_top_r.grid(False)
    ax_bot = axes[1, col_idx]

    handles, labels_ = [], []

    # ----- Top row LEFT axis: source flow; RIGHT axis: source pressure -----
    for name in LEFT_SOURCES:
        if name in wS_by_src and wS_by_src[name] in src_w.columns:
            c = wS_by_src[name]
            color = src_color[name]
            line, = ax_top.plot(t, src_w[c].values * FLOW_SCALE,
                                color=color, lw=1.7, label=name)
            handles.append(line); labels_.append(name)
            y_ocss = tile_ocss(ocss_t, ocss_src_w[c].values * FLOW_SCALE,
                               t, OCSS_CYCLE_HOURS)
            ax_top.plot(t, y_ocss, color=color, lw=1.1, linestyle=":", alpha=0.75)

    for name in RIGHT_SOURCES:
        if name in pS_by_src and pS_by_src[name] in src_p.columns:
            c = pS_by_src[name]
            color = src_color[name]
            line, = ax_top_r.plot(t, src_p[c].values * PRESSURE_SCALE,
                                  color=color, lw=1.7, label=name)
            handles.append(line); labels_.append(name + " (p)")
            y_ocss = tile_ocss(ocss_t, ocss_src_p[c].values * PRESSURE_SCALE,
                               t, OCSS_CYCLE_HOURS)
            ax_top_r.plot(t, y_ocss, color=color, lw=1.1,
                          linestyle=":", alpha=0.75)
    ax_top_r.set_ylabel("Pressure (bar)")

    ocss_proxy = plt.Line2D([], [], color="black", linestyle=":", lw=1.2)
    handles.append(ocss_proxy); labels_.append("OCSS")

    ax_top.set_xlabel("Time (hrs)")
    ax_top.set_ylabel("Flow (kg/s)")
    ax_top.set_xlim(0, X_MAX_HOURS)
    ax_top.legend(handles, labels_, loc="lower right",
                  fontsize=8, framealpha=0.9)
    add_panel_label(ax_top, PANEL_LABELS[col_idx])

    # ----- Bottom row: Compressor energy (kWh) -----
    for c in pwr_cols:
        if c in pwr.columns:
            label = entity_name(c).replace("compressorStation_", "C")
            ax_bot.plot(t, pwr[c].values * POWER_SCALE,
                        color=pwr_color[c], lw=1.7, label=label)
    for c in pwr_cols:
        if c in ocss_pwr.columns:
            y_ocss = tile_ocss(ocss_t, ocss_pwr[c].values * POWER_SCALE,
                               t, OCSS_CYCLE_HOURS)
            ax_bot.plot(t, y_ocss, color=pwr_color[c],
                        lw=1.1, linestyle=":", alpha=0.75)
    ax_bot.plot([], [], color="black", linestyle=":", lw=1.2, label="OCSS")

    ax_bot.set_xlabel("Time (hrs)")
    ax_bot.set_ylabel("Energy (kWh)")
    ax_bot.set_xlim(0, X_MAX_HOURS)
    ax_bot.legend(loc="upper right", fontsize=8, ncol=2, framealpha=0.9)
    add_panel_label(ax_bot, PANEL_LABELS[n_cols + col_idx])

out = HERE / OUTPUT_PNG
fig.savefig(out, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")


# ----------------------------------------------------------------------------
# Total cost and CPU time -> results.txt
# ----------------------------------------------------------------------------
def write_results(path):
    lines = []
    for fn, t, src_w, src_p, pwr, beta, cpu_seconds in runs:
        cost = compute_cost_kwh(pwr, pwr_cols, dt=3600, scale_factor=1)
        cpu_str = format_cpu_time(cpu_seconds) if cpu_seconds is not None else "N/A"
        lines.append(f"{fn}")
        lines.append(f"    Total cost: {cost:.2f}")
        lines.append(f"    CPU time:   {cpu_str}")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


write_results(HERE / OUTPUT_TXT)

plt.show()
