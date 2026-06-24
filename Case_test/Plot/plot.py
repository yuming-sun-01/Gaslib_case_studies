"""
plot.py — 2x3 E-NMPC comparison plot for the Test network using local Excel data.

Place this script in the SAME folder as the Excel files. It reads them via
relative paths and writes a single figure `enmpc_comparison.png` next to it.

Layout (2 rows x 3 columns):
    Row 1 (top):    Source flow on LEFT y-axis (kg/s).
                    Source pressure is constant in this network, so the right
                    pressure axis is omitted.
    Row 2 (bottom): All 3 compressor energies (kWh) with OCSS dotted.

Columns:
    Col 1: f_enmpc_cycle_1.xlsx  (finite-horizon NMPC, 1-cycle horizon)
    Col 2: f_enmpc_cycle_10.xlsx (finite-horizon NMPC, 10-cycle horizon)
    Col 3: Inf_enmpc_felement=1.xlsx (infinite-horizon NMPC, 1-cycle segment)

OCSS reference comes from `css.xlsx` (perfectly periodic single-cycle OCSS,
tiled across the visible time axis).

Total cost and CPU time per scenario are written to `results.txt`.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Configuration -- edit names, slices, units, titles here.
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

# Truncate x-axis (and data) to this many hours: 4 cycles x 6 hours.
X_MAX_HOURS = 24

# (filename, title, slice). slice=None means use all rows.
COLUMNS = [
    ("f_enmpc_cycle_1.xlsx",      "Finite horizon E-NMPC\nHorizon – 1 cycle",          None),
    ("f_enmpc_cycle_10.xlsx",     "Finite horizon E-NMPC\nHorizon – 10 cycles",        None),
    ("Inf_enmpc_felement=1.xlsx", "Infinite horizon E-NMPC\nFinite segment – 1 cycle", None),
]

# OCSS reference: dedicated css.xlsx (perfectly periodic single-cycle OCSS).
OCSS_FILE          = "css.xlsx"
OCSS_CYCLE_HOURS   = 6
OCSS_DROP_LAST     = 1     # drop the duplicated periodic endpoint

# Test network: single source; pSource is constant, so only flow is plotted.
LEFT_SOURCES  = ["source_1"]
RIGHT_SOURCES = []         # no pressure on right axis

# Unit scaling driven by model.py: scale['p']=1e5, scale['P']=1e5, scale['w']=1.
FLOW_SCALE     = 1.0
PRESSURE_SCALE = 1.0
POWER_SCALE    = 100.0    # compressor_P [internal] * 100 = kW = kWh per 1-hour step

# Output
OUTPUT_PNG  = "enmpc_comparison.png"
OUTPUT_TXT  = "results.txt"
DPI         = 200


def compute_cycle_costs(pwr_df, pwr_cols, cycle_hours, dt=3600, scale_factor=1000):
    """Return per-cycle costs using the OBJ_compressor_power formula from model.py:

    Obj = sum(P_internal * dt) / scale_factor   (excludes the last timestep).
    """
    arr = pwr_df[pwr_cols].iloc[:-1].values    # drop final timestep
    n = len(arr)
    cycles = []
    for k in range(n // cycle_hours):
        block = arr[k * cycle_hours : (k + 1) * cycle_hours]
        cycles.append(float(block.sum()) * dt / scale_factor)
    return cycles


def compute_cost_kwh(pwr_df, pwr_cols, dt=3600, scale_factor=1000):
    """Total cost over all complete cycles."""
    return sum(compute_cycle_costs(pwr_df, pwr_cols, OCSS_CYCLE_HOURS, dt=dt, scale_factor=scale_factor))


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_run(path):
    """Load wSource, pSource, compressor power, compressor beta sheets."""
    src_w = pd.read_excel(path, sheet_name="wSource")
    src_p = pd.read_excel(path, sheet_name="pSource")
    pwr   = pd.read_excel(path, sheet_name="compressor power")
    beta  = pd.read_excel(path, sheet_name="compressor beta")
    for df in (src_w, src_p, pwr, beta):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return src_w, src_p, pwr, beta


def sort_by_trailing_int(cols):
    """Sort variable columns by the trailing integer in the entity name."""
    def key(c):
        try:
            return int(c.split("'")[1].split("_")[-1])
        except Exception:
            return 0
    return sorted(cols, key=key)


def entity_name(col):
    """Extract 'source_1' from \"wSource['source_1', :]\" etc."""
    return col.split("'")[1]


def slice_run(src_w, src_p, pwr, beta, sl):
    if sl is None:
        return src_w, src_p, pwr, beta
    return (src_w.iloc[sl].reset_index(drop=True),
            src_p.iloc[sl].reset_index(drop=True),
            pwr.iloc[sl].reset_index(drop=True),
            beta.iloc[sl].reset_index(drop=True))


def truncate_to_hours(src_w, src_p, pwr, beta, max_hours):
    """Keep rows 0..max_hours (inclusive) — one extra row for the endpoint."""
    keep = min(max_hours + 1, len(src_w))
    return (src_w.iloc[:keep].reset_index(drop=True),
            src_p.iloc[:keep].reset_index(drop=True),
            pwr.iloc[:keep].reset_index(drop=True),
            beta.iloc[:keep].reset_index(drop=True))


def tile_ocss(ocss_t, ocss_vals, target_t, cycle_hours):
    """Periodic interpolation: value at t+cycle_hours equals value at t."""
    phase = np.mod(target_t, cycle_hours)
    t_ext = np.concatenate([ocss_t, [cycle_hours]])
    v_ext = np.concatenate([ocss_vals, [ocss_vals[0]]])
    return np.interp(phase, t_ext, v_ext)


def load_cpu_time_seconds(path):
    """Read total CPU time in seconds from the NMPC Excel file.
    Sums iteration 0..23 from the `runtime` sheet (preferred),
    or falls back to `timing.total_runtime_s[0]`.
    """
    xl = pd.ExcelFile(path)
    if "runtime" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="runtime")
        if {"iteration", "iteration_time_s"}.issubset(df.columns) and len(df) > 0:
            iterations = pd.to_numeric(df["iteration"], errors="coerce")
            vals = pd.to_numeric(df["iteration_time_s"], errors="coerce")
            vals = vals[(iterations >= 0) & (iterations <= 23)].dropna()
            if len(vals) > 0:
                return float(vals.sum())
    if "timing" in xl.sheet_names:
        df = pd.read_excel(path, sheet_name="timing")
        if "total_runtime_s" in df.columns:
            val = df["total_runtime_s"].iloc[0]
            if pd.notna(val):
                return float(val)
    return None


def format_cpu_time(seconds):
    """Test network: seconds."""
    return f"{seconds:.1f} s"


# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
runs = []
for fn, title, sl in COLUMNS:
    src_w, src_p, pwr, beta = load_run(HERE / fn)
    src_w, src_p, pwr, beta = slice_run(src_w, src_p, pwr, beta, sl)
    src_w, src_p, pwr, beta = truncate_to_hours(src_w, src_p, pwr, beta, X_MAX_HOURS)
    cpu_seconds = load_cpu_time_seconds(HERE / fn)
    t = np.arange(len(src_w))
    runs.append((title, t, src_w, src_p, pwr, beta, cpu_seconds))

# OCSS reference: one clean cycle
ocss_src_w, ocss_src_p, ocss_pwr, ocss_beta = load_run(HERE / OCSS_FILE)
# css.xlsx may already span multiple periods; keep only the first cycle
keep = min(OCSS_CYCLE_HOURS, len(ocss_src_w) - OCSS_DROP_LAST)
ocss_src_w = ocss_src_w.iloc[:keep].reset_index(drop=True)
ocss_src_p = ocss_src_p.iloc[:keep].reset_index(drop=True)
ocss_pwr   = ocss_pwr.iloc[:keep].reset_index(drop=True)
ocss_beta  = ocss_beta.iloc[:keep].reset_index(drop=True)
ocss_t     = np.arange(len(ocss_src_w))

# Column lists, sorted by trailing integer for consistent ordering / colors
wS_cols   = sort_by_trailing_int([c for c in ocss_src_w.columns if "wSource"         in c])
pS_cols   = sort_by_trailing_int([c for c in ocss_src_p.columns if "pSource"         in c])
pwr_cols  = sort_by_trailing_int([c for c in ocss_pwr.columns   if "compressor_P"    in c])
beta_cols = sort_by_trailing_int([c for c in ocss_beta.columns  if "compressor_beta" in c])

# Map entity_name -> wSource and pSource column strings
wS_by_src = {entity_name(c): c for c in wS_cols}
pS_by_src = {entity_name(c): c for c in pS_cols}

# Colors per source (consistent across left/right axes and OCSS lines)
src_order = LEFT_SOURCES + RIGHT_SOURCES
cmap10    = plt.get_cmap("tab10")
src_color = {name: cmap10(i % 10) for i, name in enumerate(src_order)}

# Colors per compressor
pwr_color = {c: cmap10(i % 10) for i, c in enumerate(pwr_cols)}


# ----------------------------------------------------------------------------
# Plot: single 2xN comparison figure
# ----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
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

for col_idx, (title, t, src_w, src_p, pwr, beta, cpu_seconds) in enumerate(runs):
    ax_top = axes[0, col_idx]
    ax_bot = axes[1, col_idx]

    handles, labels_ = [], []

    # ----- Top row LEFT axis: flow for LEFT_SOURCES -----
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

    # OCSS legend entry (neutral)
    ocss_proxy = plt.Line2D([], [], color="black", linestyle=":", lw=1.2)
    handles.append(ocss_proxy); labels_.append("OCSS")

    ax_top.set_title(title, fontsize=12)
    ax_top.set_xlabel("Time (hrs)")
    ax_top.set_ylabel("Flow (kg/s)")
    ax_top.set_xlim(0, X_MAX_HOURS)
    ax_top.legend(handles, labels_, loc="upper right",
                  fontsize=8, framealpha=0.9)

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

# Save
out = HERE / OUTPUT_PNG
fig.savefig(out, dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")


# ----------------------------------------------------------------------------
# Total cost and CPU time -> results.txt
# ----------------------------------------------------------------------------
def write_results(path):
    lines = []
    for (title, t, src_w, src_p, pwr, beta, cpu_seconds), (fn, _, _) in zip(runs, COLUMNS):
        cost = compute_cost_kwh(pwr, pwr_cols, dt=3600, scale_factor=1)
        cpu_str = format_cpu_time(cpu_seconds) if cpu_seconds is not None else "N/A"
        flat_title = title.replace("\n", " — ")
        lines.append(f"{flat_title} [{fn}]")
        lines.append(f"    Total cost: {cost:.2f}")
        lines.append(f"    CPU time:   {cpu_str}")
        lines.append("")
    path.write_text("\n".join(lines))
    print(f"Saved: {path}")


write_results(HERE / OUTPUT_TXT)

plt.show()
