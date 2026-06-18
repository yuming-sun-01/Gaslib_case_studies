# .py file needed: utils, model, controller_factory, infinite_block, infNMPC_options
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.common.timing import HierarchicalTimer

from utils import (
    import_data_from_excel,
    write_data_to_excel,
    dynamic_demand_calculation,
)
from model import buildNonLinearModel, init_network_default
from controller_factory import run_model
from utils import _norm
from infNMPC_options import _import_settings
from stability import apply_stability_constraint

# Module-level cache: avoid re-reading the CSS workbook on every MPC step.
_css_wb_cache: dict = {}

# =============================================================================
# Demand loading — pre-computed array (no upper limit on simulation length)
# =============================================================================
def _build_demand_data(m_steady, simulation_steps, opts):
    """Pre-compute a demand profile using dynamic_demand_calculation on m_steady.

    Uses m_steady (steady-state wCons) as the base demand — same as run_model
    does internally — so MPC step 0 demands exactly match run_model's setting.

    Length is a multiple of T_period_steps to ensure modular indexing in
    _load_demand_into_model wraps correctly.
    """
    T_period_steps = int(round(opts.T_period / opts.sample_time))  # e.g. 6
    # Need enough entries for all MPC steps; keep as multiple of T_period_steps.
    n_needed = simulation_steps + T_period_steps + 1
    n_periods = math.ceil(n_needed / T_period_steps) + 1
    demand_length = n_periods * T_period_steps

    return dynamic_demand_calculation(m_steady, time_length=demand_length)


def _load_demand_into_model(m, demand_data, step_index):
    """Fix sink wCons from pre-computed demand_data at global step step_index.

    Non-sink wCons (pipe-volume centres) must already be pinned to 0.
    IMPORTANT: Unfix before re-fixing to update values in MPC loop
    """
    times_list = list(m.Times)
    t_to_idx = {t: k for k, t in enumerate(times_list)}
    for idx in m.wCons:
        if not str(idx[0]).startswith("sink"):
            continue
        sink = idx[0]
        if sink not in demand_data:
            continue
        t_local = idx[-1]
        t_index = t_to_idx.get(t_local, 0)
        full = demand_data[sink]

        # UNFIX if already fixed (from previous MPC step)
        if m.wCons[idx].is_fixed():
            m.wCons[idx].unfix()

        # FIX with new demand value
        m.wCons[idx].fix(float(full[(step_index + t_index) % len(full)]))

# =============================================================================
# CSS reference refresh: interm_p, compressor_beta, interm_w, wSource
# =============================================================================
def _refresh_phase_references(m_controller, sim_t0):
    """Update CSS reference params for the current MPC step via direct interpolation.

    Phase map:
        t_real(tau) = T_finite + L * arctanh(tau)  (controller-local time)
        phase(tau)  = (sim_t0 + t_real(tau)) mod T_period
        tau = 1   → phase = sim_t0 mod T_period   (periodic endpoint)
    """
    info        = m_controller._phase_info
    T_period    = float(info['T_period'])
    T_finite    = float(info['T_finite'])
    L           = float(info['L'])
    ocss_path   = str(info['ocss_file_path'])
    pipes_fit   = info.get('pipes_with_fit', [])
    stns_fit    = info.get('stations_with_fit', [])
    stns_pc_fit = info.get('stations_p_comp_fit', [])
    time_pts    = info['time_points']

    ib = m_controller.inf_block
    tau_max = 1.0 - 1e-9

    # ---- load CSS workbook once per file path (module-level cache) ----------
    if ocss_path not in _css_wb_cache:
        def _load(sheet):
            try:
                return pd.read_excel(ocss_path, sheet_name=sheet, index_col=0)
            except Exception:
                return None
        _css_wb_cache[ocss_path] = {
            'interm_p':       _load('interm_p'),
            'compressor beta': _load('compressor beta'),
            'compressor power': _load('compressor power'),
        }
    wb = _css_wb_cache[ocss_path]
    df_p = wb['interm_p']
    df_b = wb['compressor beta']
    df_P = wb['compressor power']

    # ---- phase for each infinite-block tau ----------------------------------
    phase_at = {}
    for tau in time_pts:
        if abs(float(tau) - 1.0) <= 1e-12:
            phase_at[tau] = float(sim_t0) % T_period
        else:
            tau_clip = min(float(tau), tau_max)
            t_real = T_finite + L * math.atanh(tau_clip)
            phase_at[tau] = (float(sim_t0) + t_real) % T_period

    def _col(df, attr_name, keys):
        if df is None:
            return None
        col_by_norm = {_norm(str(c)): c for c in df.columns}
        if len(keys) == 1:
            target = _norm(f"{attr_name}[{keys[0]},:]")
        else:
            target = _norm(f"{attr_name}[{keys[0]},{keys[1]},:]")
        col = col_by_norm.get(target)
        if col is None:
            needed = [_norm(str(k)) for k in keys]
            col = next((c for c in df.columns if all(n in _norm(c) for n in needed)), None)
        return col

    def _val(df, col, phase):
        if df is None or col is None:
            return None
        closest = min(df.index, key=lambda x: abs(float(x) - phase))
        try:
            return float(df.loc[closest, col])
        except Exception:
            return None

    # ---- update infinite-block params ---------------------------------------
    if pipes_fit and hasattr(ib, 'interm_p_fit'):
        for (p, vol) in pipes_fit:
            c = _col(df_p, 'interm_p', (p, vol))
            for tau in time_pts:
                v = _val(df_p, c, phase_at[tau])
                if v is not None:
                    ib.interm_p_fit[p, vol, tau] = v

    if stns_fit and hasattr(ib, 'compressor_beta_fit'):
        for stn in stns_fit:
            c = _col(df_b, 'compressor_beta', (stn,))
            for tau in time_pts:
                v = _val(df_b, c, phase_at[tau])
                if v is not None:
                    ib.compressor_beta_fit[stn, tau] = v

    if stns_pc_fit and hasattr(ib, 'compressor_P_fit'):
        for stn in stns_pc_fit:
            c = _col(df_P, 'compressor_P', (stn,))
            for tau in time_pts:
                v = _val(df_P, c, phase_at[tau])
                if v is not None:
                    ib.compressor_P_fit[stn, tau] = v


# =============================================================================
# Rolling endpoint reference: roll the terminal CSS density target each MPC step
# =============================================================================
def _refresh_terminal_css_reference(m_controller, sim_t0):
    """Roll the endpoint (tau=1) CSS density reference with the receding horizon.

    The periodic endpoint reference is the CSS density at phase = sim_t0 mod
    T_period, so MPC step i (sim_t0 = i*sample_time) tracks a rolling target:
        step 0 (t=0 h) -> CSS(t0),  step 1 (t=1 h) -> CSS(t=1), ...
    Updates the mutable Param inf_block.terminal_css_target in place; returns the
    phase used (or None if there is no terminal constraint to refresh).
    """
    ib = getattr(m_controller, 'inf_block', None)
    if ib is None or not hasattr(ib, 'terminal_css_target'):
        return None

    info = m_controller._phase_info
    T_period = float(info['T_period'])
    ocss_path = str(info['ocss_file_path'])
    phase = float(sim_t0) % T_period

    cache = _css_wb_cache.setdefault(ocss_path, {})
    if 'interm_p' not in cache:
        try:
            cache['interm_p'] = pd.read_excel(
                ocss_path, sheet_name='interm_p', index_col=0)
        except Exception:
            cache['interm_p'] = None
    df = cache['interm_p']
    if df is None:
        return phase

    col_by_norm = {_norm(str(c)): c for c in df.columns}
    closest = min(df.index, key=lambda x: abs(float(x) - phase))
    n = 0
    for (p, vol) in ib.terminal_css_index:
        col = col_by_norm.get(_norm(f"interm_p[{p},{vol},:]"))
        if col is None:
            needed = [_norm(str(p)), _norm(str(vol))]
            col = next((c for c in df.columns
                        if all(k in _norm(c) for k in needed)), None)
        if col is None:
            continue
        ib.terminal_css_target[p, vol] = float(df.loc[closest, col])
        n += 1
    print(f"  [endpoint] rolled CSS reference to phase {phase:.2f} h "
          f"({n} interm_p targets)")
    return phase


def _refresh_tail_wcons_reference(m_controller, sim_t0):
    """Roll the infinite-tail CSS demand (wCons) with the receding horizon.

    The tail wCons are FIXED variables; this resets their fixed values to the
    CSS demand at the rolling phase for each collocation point tau:
        phase(tau)  = (sim_t0 + T_finite + L*atanh(tau)) mod T_period
        phase(tau=1) = sim_t0 mod T_period   (periodic endpoint)
    At sim_t0 = 0 this reproduces the build-time reference, then it advances
    every MPC step. Values are set in place; the variables stay fixed.
    """
    ib = getattr(m_controller, 'inf_block', None)
    if ib is None or not hasattr(ib, 'wCons'):
        return

    info = m_controller._phase_info
    T_period = float(info['T_period'])
    T_finite = float(info['T_finite'])
    L = float(info['L'])
    ocss_path = str(info['ocss_file_path'])
    tau_max = 1.0 - 1e-9

    cache = _css_wb_cache.setdefault(ocss_path, {})
    if 'wCons' not in cache:
        try:
            cache['wCons'] = pd.read_excel(
                ocss_path, sheet_name='wCons', index_col=0)
        except Exception:
            cache['wCons'] = None
    df = cache['wCons']
    if df is None:
        return

    col_by_norm = {_norm(str(c)): c for c in df.columns}
    col_cache = {}

    def _get_col(c0, c1):
        if (c0, c1) not in col_cache:
            col = col_by_norm.get(_norm(f"wCons[{c0},{c1},:]"))
            if col is None:
                col = next((c for c in df.columns
                            if _norm(str(c0)) in _norm(c)
                            and _norm(str(c1)) in _norm(c)), None)
            col_cache[(c0, c1)] = col
        return col_cache[(c0, c1)]

    def _phase(tau):
        if abs(float(tau) - 1.0) <= 1e-12:
            return float(sim_t0) % T_period
        tau_clip = min(float(tau), tau_max)
        t_real = T_finite + L * math.atanh(tau_clip)
        return (float(sim_t0) + t_real) % T_period

    n = 0
    for idx in ib.wCons:
        c0, c1, tau = idx
        col = _get_col(c0, c1)
        if col is None:
            val = 0.0
        else:
            phase = _phase(tau)
            closest = min(df.index, key=lambda x: abs(float(x) - phase))
            val = float(df.loc[closest, col])
        ib.wCons[idx].set_value(val)  # variable stays fixed
        n += 1
    print(f"  [tail] rolled wCons reference ({n} fixed points) "
          f"to sim_t0={sim_t0:.2f} h")


def _reinit_infinite_tail_from_css(m_controller, sim_t0):
    """Re-initialize the infinite-tail FREE variables to the current-phase CSS.

    The receding-horizon interface only shifts the finite block; the infinite
    block keeps its previous solution, which goes stale once the endpoint/tail
    references roll. This resets the inf-tail variable VALUES (warm start only,
    no structural change) to the CSS profile at the rolling phase per tau:
        phase(tau)   = (sim_t0 + T_finite + L*atanh(tau)) mod T_period
        phase(tau=1) = sim_t0 mod T_period
    Only free variables are touched (fixed pSource / wCons / eta are left alone).
    """
    ib = getattr(m_controller, 'inf_block', None)
    if ib is None:
        return

    info = m_controller._phase_info
    T_period = float(info['T_period'])
    T_finite = float(info['T_finite'])
    L = float(info['L'])
    ocss_path = str(info['ocss_file_path'])
    tau_max = 1.0 - 1e-9

    cache = _css_wb_cache.setdefault(ocss_path, {})

    def _df(sheet):
        if sheet not in cache:
            try:
                cache[sheet] = pd.read_excel(
                    ocss_path, sheet_name=sheet, index_col=0)
            except Exception:
                cache[sheet] = None
        return cache[sheet]

    def _phase(tau):
        if abs(float(tau) - 1.0) <= 1e-12:
            return float(sim_t0) % T_period
        tau_clip = min(float(tau), tau_max)
        return (float(sim_t0) + T_finite + L * math.atanh(tau_clip)) % T_period

    # inf.pipe_rho[t0] is pinned to the finite-block end by link_pipe_rho;
    # overwriting it with CSS would create a large link mismatch in the warm
    # start during the transient. Leave that one time point untouched.
    t0_inf = ib.time.first()

    # (inf_block var attribute, CSS sheet name, CSS column attribute name)
    specs = [
        ('pipe_rho',        'pipe_rho',         'pipe_rho'),
        ('interm_p',        'interm_p',         'interm_p'),
        ('compressor_beta', 'compressor beta',  'compressor_beta'),
        ('compressor_P',    'compressor power', 'compressor_P'),
        ('node_p',          'node pressure',    'node_p'),
        ('interm_w',        'interm_w',         'interm_w'),
    ]

    n_set = 0
    for var_attr, sheet, col_attr in specs:
        var = getattr(ib, var_attr, None)
        df = _df(sheet)
        if var is None or df is None:
            continue

        col_by_norm = {_norm(str(c)): c for c in df.columns}
        col_cache = {}
        row_cache = {}

        def _get_col(spatial):
            if spatial not in col_cache:
                inner = ",".join(str(s) for s in spatial)
                col = col_by_norm.get(_norm(f"{col_attr}[{inner},:]"))
                if col is None:
                    needed = [_norm(str(s)) for s in spatial]
                    col = next((c for c in df.columns
                                if all(k in _norm(c) for k in needed)), None)
                col_cache[spatial] = col
            return col_cache[spatial]

        def _closest_row(phase):
            key = round(phase, 6)
            if key not in row_cache:
                row_cache[key] = min(df.index, key=lambda x: abs(float(x) - phase))
            return row_cache[key]

        for idx in var:
            v = var[idx]
            if v.is_fixed():
                continue
            idx_t = idx if isinstance(idx, tuple) else (idx,)
            spatial, tau = idx_t[:-1], idx_t[-1]
            # Do not touch the link-pinned density at the infinite-block t0.
            if var_attr == 'pipe_rho' and tau == t0_inf:
                continue
            col = _get_col(spatial)
            if col is None:
                continue
            v.set_value(float(df.loc[_closest_row(_phase(tau)), col]))
            n_set += 1

    print(f"  [tail-init] re-initialized {n_set} inf-tail var values "
          f"to CSS at sim_t0={sim_t0:.2f} h")


# =============================================================================
# Lyapunov reference: roll the interm_p CSS targets (finite + infinite tail)
#   phase_fin(t)   = (sim_t0 + t) mod T_period
#   phase_inf(tau) = (sim_t0 + T_finite + L*atanh(tau)) mod T_period   [tau=1 -> sim_t0]
# =============================================================================
def _interm_p_css_df(ocss_path):
    cache = _css_wb_cache.setdefault(ocss_path, {})
    if 'interm_p' not in cache:
        try:
            cache['interm_p'] = pd.read_excel(
                ocss_path, sheet_name='interm_p', index_col=0)
        except Exception:
            cache['interm_p'] = None
    return cache['interm_p']


def _refresh_lyap_css_reference(m_controller, sim_t0):
    if not hasattr(m_controller, 'interm_p_css_fin'):
        return
    info = m_controller._phase_info
    T_period = float(info['T_period'])
    T_finite = float(info['T_finite'])
    L = float(info['L'])
    tau_max = 1.0 - 1e-9

    df = _interm_p_css_df(str(info['ocss_file_path']))
    if df is None:
        return
    col_by_norm = {_norm(str(c)): c for c in df.columns}
    col_cache = {}

    def _col(p, vol):
        if (p, vol) not in col_cache:
            col = col_by_norm.get(_norm(f"interm_p[{p},{vol},:]"))
            if col is None:
                col = next((c for c in df.columns
                            if _norm(str(p)) in _norm(c)
                            and _norm(str(vol)) in _norm(c)), None)
            col_cache[(p, vol)] = col
        return col_cache[(p, vol)]

    def _val(col, phase):
        if col is None:
            return None
        closest = min(df.index, key=lambda x: abs(float(x) - phase))
        return float(df.loc[closest, col])

    last_t = m_controller.Times.last()
    ib = getattr(m_controller, 'inf_block', None)
    for (p, vol) in m_controller.Pipes_VolExtrR_interm:
        col = _col(p, vol)
        for t in m_controller.Times:
            if t == last_t:
                continue
            v = _val(col, (float(sim_t0) + float(t)) % T_period)
            if v is not None:
                m_controller.interm_p_css_fin[p, vol, t] = v
        if ib is not None and hasattr(ib, 'interm_p_css'):
            for tau in sorted(ib.time):
                if abs(float(tau) - 1.0) <= 1e-12:
                    phase = float(sim_t0) % T_period
                else:
                    phase = (float(sim_t0) + T_finite
                             + L * math.atanh(min(float(tau), tau_max))) % T_period
                v = _val(col, phase)
                if v is not None:
                    ib.interm_p_css[p, vol, tau] = v
    print(f"  [lyapunov] rolled interm_p CSS reference to sim_t0={sim_t0:.2f} h")


def _plant_lyap_stage_cost(m_controller, m_plant, phase):
    """Realized plant interm_p stage cost at plant last time: sum (interm_p - CSS(phase))^2."""
    df = _interm_p_css_df(str(m_controller._phase_info['ocss_file_path']))
    if df is None:
        return 0.0
    col_by_norm = {_norm(str(c)): c for c in df.columns}
    closest = min(df.index, key=lambda x: abs(float(x) - phase))
    t_last_p = m_plant.Times.last()
    sse = 0.0
    for (p, vol) in m_controller.Pipes_VolExtrR_interm:
        col = col_by_norm.get(_norm(f"interm_p[{p},{vol},:]"))
        if col is None:
            col = next((c for c in df.columns
                        if _norm(str(p)) in _norm(c)
                        and _norm(str(vol)) in _norm(c)), None)
        if col is None:
            continue
        e = pyo.value(m_plant.interm_p[p, vol, t_last_p]) - float(df.loc[closest, col])
        sse += e * e
    return sse


def _report_endpoint_error(m_controller):
    """Print the terminal-constraint LHS error: sum (pipe_rho[tau=1]-CSS)^2."""
    ib = getattr(m_controller, 'inf_block', None)
    if ib is None or not hasattr(ib, 'terminal_css_target'):
        return
    t_last = ib.terminal_t_last
    sse = 0.0
    max_err = 0.0
    for (p, vol) in ib.terminal_css_index:
        e = (pyo.value(ib.interm_p[p, vol, t_last])
             - pyo.value(ib.terminal_css_target[p, vol]))
        sse += e * e
        max_err = max(max_err, abs(e))
    bound = pyo.value(ib.terminal_error_bound)
    # IPOPT only satisfies constraints to constr_viol_tol (~1e-4), so an
    # overshoot of a few e-8 means the constraint is simply ACTIVE at the
    # bound, not violated. Flag a real violation only beyond that tolerance.
    viol = sse - bound
    if viol <= 1e-4:
        status = "ACTIVE" if viol > 1e-7 else "OK"
    else:
        status = "VIOLATED"
    print(f"  [endpoint] LHS sum(interm_p[tau=1]-CSS)^2 = {sse:.6e} "
          f"(bound {bound:.2e}, {status}); max|err| = {max_err:.6e}")


# =============================================================================
# Plant model  (1-sample-horizon dynamic simulator)
# =============================================================================
def _build_plant(controller_t0_inputs, m_steady, opts):
    """Build and warm-start the plant model, exactly mirroring run_model_css_72.py.

    Two-solve initialisation:
      1. Fix pSource + interm_p[t0] + compressor_P[t0] from m_steady → first solve
         (compressor_P[t>0] free, inputData demands — matches css_72 first solve).
      2. Fix compressor_P at ALL times to controller values → returned model.
    CSS demands are loaded in run_nmpc at each MPC step.
    """
    networkData, inputData = import_data_from_excel(
        str(opts.network_data_path), str(opts.input_data_path))
    with open(str(opts.options_data_path), "r") as fh:
        Options = json.load(fh)
    Options["dynamic"] = True
    Options["T"] = 1   # plant horizon = 1 sampling interval

    m_plant = buildNonLinearModel(networkData, inputData, Options)
    m_plant = init_network_default(m_plant, p_default=55e5)
    t0 = m_plant.Times.first()
    t0_s = m_steady.Times.first()

    # Step 1: mirror run_model_css_72 first solve --------------------------
    # Fix pSource from steady-state
    m_plant.pSource["source_1", :].fix()

    # Fix interm_p[t0] from steady-state
    for p, vol in m_plant.Pipes_VolExtrR_interm.data():
        m_plant.interm_p[p, vol, t0] = pyo.value(m_steady.interm_p[p, vol, t0_s])
    m_plant.interm_p[:, :, t0].fix()

    # Fix compressor_P only at t0 (like css_72: m_dyn.compressor_P[:, t0].fix())
    for c in m_plant.Stations:
        m_plant.compressor_P[c, t0] = pyo.value(m_steady.compressor_P[c, t0_s])
    m_plant.compressor_P[:, t0].fix()

    # Pin non-sink wCons to 0
    for idx in m_plant.wCons:
        if not str(idx[0]).startswith("sink"):
            m_plant.wCons[idx].fix(0.0)

    m_plant.compressor_beta['compressorStation_1', :].setub(4)
    m_plant.ObjFun.deactivate()

    # First solve (inputData demands, compressor_P[t>0] free — same as css_72)
    ipopt_init = pyo.SolverFactory('ipopt')
    ipopt_init.options['linear_solver'] = opts.linear_solver
    res = ipopt_init.solve(m_plant, tee=False)
    if not pyo.check_optimal_termination(res):
        print("[_build_plant] WARNING: first solve did not converge — continuing.")

    # Step 2: fix compressor_P at ALL times to controller values -----------
    for c in m_plant.Stations:
        m_plant.compressor_P[c, t0].unfix()
        for t in m_plant.Times:
            m_plant.compressor_P[c, t].set_value(controller_t0_inputs[c])
    m_plant.compressor_P[:, :].fix()

    return m_plant

# =============================================================================
# Main NMPC loop
# =============================================================================
def run_nmpc(simulation_steps=None, sample_time=None, opts=None):
    """
    Closed-loop NMPC using the CSS-tracking infinite-horizon controller.
    """
    if opts is None:
        opts = _import_settings()
    if simulation_steps is None:
        simulation_steps = int(opts.plant_horizon)
    if sample_time is None:
        sample_time = float(opts.sample_time)

    _total_start = time.time()
    timer = HierarchicalTimer()
    timer.start("Initialization")

    # ----- controller (finite + CSS infinite block) ------------------------
    m_steady, m_controller = run_model(opts)

    # ----- Lyapunov stability constraint on interm_p (finite + inf tail) ----
    _stability_on = bool(getattr(opts, 'stability_constraint', False))
    T_period = float(opts.T_period)
    if _stability_on:
        apply_stability_constraint(
            m_controller, delta=float(getattr(opts, 'delta', 0.05)))

    # ----- demand data (pre-computed array, handles arbitrary sim length) --
    demand_data = _build_demand_data(m_steady, simulation_steps, opts)

    # ----- plant -----------------------------------------------------------
    t0_c = m_controller.Times.first()
    first_move_c = min(
        m_controller.Times,
        key=lambda t: abs(float(t) - sample_time),
    )
    controller_t0_inputs = {
        c: pyo.value(m_controller.compressor_P[c, t0_c])
        for c in m_controller.Stations
        }
    for c, val in controller_t0_inputs.items():
        m_controller.compressor_P[c, t0_c].set_value(val)
        m_controller.compressor_P[c, t0_c].fix()

    controller_initial_interm_p = {
        (p, vol): pyo.value(m_controller.interm_p[p, vol, t0_c])
        for p, vol in m_controller.Pipes_VolExtrR_interm.data()
    }

    # _build_plant already does the two-step css_72-style initialisation.
    m_plant = _build_plant(controller_t0_inputs, m_steady, opts)
    t0_p = m_plant.Times.first()

    # ----- interfaces ------------------------------------------------------
    controller_interface = mpc.DynamicModelInterface(
        m_controller, m_controller.Times)
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.Times)

    # Compressor power is the transferred control from controller -> plant.
    plant_fixed_variables = [
        m_controller.compressor_P[s, :] for s in m_controller.Stations
    ]
    non_initial_plant_time = list(m_plant.Times)[1:] # t=0 is fixed

    timer.stop("Initialization")


    # In-loop solver: same minimal settings as the finite CSS build
    # (run_model_css_72.py) — linear_solver + tol/mu_init/bound_push only.
    solver = pyo.SolverFactory("ipopt")
    solver.options["linear_solver"] = opts.linear_solver
    solver.options["tol"]        = 1e-5
    solver.options["mu_init"]    = 1e-6
    solver.options["bound_push"] = 1e-6
    tee = getattr(opts, 'tee_flag', False)

    sim_t0 = 0.0
    sim_data = plant_interface.get_data_at_time([sim_t0])
    iteration_times = {}
    controller_lyapunov_function = {}  # sim_t0 -> V_current each MPC step

    # =====================================================================
    # Receding-horizon loop
    # =====================================================================
    for i in range(simulation_steps):
        timer.start(f"iteration_{i}")
        sim_t0 = i * sample_time
        print(f"--------------- MPC step {i}  (t = {sim_t0:.1f} h) ---------------")

        # UNFIX only SINK wCons in CONTROLLER (demand points)
        # Non-sink wCons should stay fixed to 0 (pipe volume centers)
        if i == 0:
            # First iteration: unfix only SINK wCons that were fixed during initialization
            for idx in m_controller.wCons:
                if str(idx[0]).startswith("sink") and m_controller.wCons[idx].is_fixed():
                    m_controller.wCons[idx].unfix()

        # Staged delta and refreshed finite-horizon data for the current MPC step.
        _load_demand_into_model(m_controller, demand_data, step_index=i)
        _load_demand_into_model(m_plant, demand_data, step_index=i)
        _refresh_phase_references(m_controller, sim_t0=sim_t0)
        # Roll the endpoint CSS density reference to the current horizon phase.
        _refresh_terminal_css_reference(m_controller, sim_t0=sim_t0)
        # Roll the infinite-tail CSS demand (wCons) to the current phase.
        _refresh_tail_wcons_reference(m_controller, sim_t0=sim_t0)
        # Re-initialize the infinite-tail variables (warm start) to the rolled
        # CSS. Skip step 0: its tail is already at the freshly-solved optimum.
        # Gated by opts.reinit_inf_tail: when off, keep the previous solved tail
        # (feasible warm start) instead of re-sampling CSS.
        if i > 0 and getattr(opts, 'reinit_inf_tail', True):
            _reinit_infinite_tail_from_css(m_controller, sim_t0=sim_t0)

        # Step 0 has no previous value, so the descent is deactivated there
        # (like Finite_stage_demand/stability.py); active from step 1 on.
        if _stability_on:
            _refresh_lyap_css_reference(m_controller, sim_t0=sim_t0)
            if i == 0:
                m_controller.stability_constraint.deactivate()
            else:
                m_controller.stability_constraint.activate()

        print("  controller DOF:", degrees_of_freedom(m_controller))

        timer.start("Solve_controller")
        _solve_ok = True
        try:
            res = solver.solve(m_controller, tee=tee)
            pyo.assert_optimal_termination(res)
        except Exception:
            _solve_ok = False
            print("  WARNING: controller did not reach optimal — continuing.")
        timer.stop("Solve_controller")

        # Endpoint constraint LHS error at this MPC step (rolling CSS target).
        _report_endpoint_error(m_controller)

    # ------- transfer first-move input to plant ----------------------
        if _solve_ok:
            ts_data    = controller_interface.get_data_at_time(sample_time)
            input_data = ts_data.extract_variables(plant_fixed_variables)
            plant_interface.load_data(input_data, time_points=non_initial_plant_time)
        else:
            print("  [fallback] controller failed — plant keeps previous compressor_P.")

        assert degrees_of_freedom(m_plant) == 0
        timer.start("Solve_plant")
        try:
            res = solver.solve(m_plant, tee=tee)
            pyo.assert_optimal_termination(res)
        except Exception:
            print("  WARNING: plant simulation did not reach optimal — continuing.")
        timer.stop("Solve_plant")

    # ------- record plant trajectory on the sample interval ----------
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0 - m_plant.Times.first())
        sim_data.concatenate(m_data)

    # ------- Lyapunov: print V every step + update prev params -------
        # The lyapunov_function_definition is always active, so V_current has a
        # value at every step (incl. step 0 where the stability constraint is
        # inactive). Print it every loop, then refresh the prev params used by
        # the next step's constraint (V_k <= V_{k-1} - delta*plant_stage_cost).
        if _stability_on:
            V_cur = pyo.value(m_controller.lyapunov_function_current)
            controller_lyapunov_function[sim_t0] = V_cur
            # Split V = finite horizon part + infinite tail (phi_V[tau=1]).
            _ib = m_controller.inf_block
            V_inf = pyo.value(_ib.phi_V[sorted(_ib.time)[-1]])
            V_fin = V_cur - V_inf
            con_state = ("inactive" if not m_controller.stability_constraint.active
                         else "ACTIVE")
            print(f"  [lyapunov] step {i} (t={sim_t0:.1f}h): V_current={V_cur:.6e} "
                  f"(finite={V_fin:.6e}, inf phi_V={V_inf:.6e}) "
                  f"(stability constraint {con_state})")
            if _solve_ok:
                m_controller.lyapunov_function_prev = V_cur
                phase_plant = (sim_t0 + sample_time) % T_period
                ell_plant = _plant_lyap_stage_cost(m_controller, m_plant, phase_plant)
                m_controller.tracking_cost_plant_prev = ell_plant
                bound = V_cur - pyo.value(m_controller.delta) * ell_plant
                print(f"  [lyapunov] plant rho stage cost={ell_plant:.6e}; "
                      f"next-step requires V <= {bound:.6e}")

    # ------- reinit plant: tf -> t0 ----------------------------------
        tf_data = plant_interface.get_data_at_time(m_plant.Times.last())
        plant_interface.load_data(tf_data)

    # ------- reinit controller: shift + new t0 from plant tf ---------
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_c)

        timer.stop(f"iteration_{i}")
        iteration_times[i] = timer.get_total_time(f"iteration_{i}")
        print(f"  Iteration {i} time: {iteration_times[i]:.2f}s", flush=True)

    print(timer)
    print("Iteration times:", iteration_times)
    if _stability_on:
        print("Lyapunov function (V_current per step):")
        for t in sorted(controller_lyapunov_function):
            print(f"  t={t:.1f}h: {controller_lyapunov_function[t]:.6e}")
    total_runtime = time.time() - _total_start
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)
    with open("hierarchical_timer_mpc_inf.txt", "w") as f:
        f.write(str(timer))
    return (m_plant, m_controller, sim_data, iteration_times, total_runtime,
            controller_lyapunov_function)


def _append_runtime_sheet(xlsx_path, iteration_times, total_runtime):
    """Append a 'runtime' sheet with per-iteration and total wall-clock time."""
    df = pd.DataFrame({
        "iteration": list(iteration_times.keys()),
        "iteration_time_s": list(iteration_times.values()),
    })
    df.loc[len(df)] = ["TOTAL", total_runtime]
    with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a",
                        if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="runtime", index=False)

# =============================================================================
# Tee logger — mirrors stdout to a .txt file
# =============================================================================
class _Tee:
    """Writes every print/stdout line to both the terminal and a log file."""
    def __init__(self, log_path):
        import sys
        self._terminal = sys.stdout
        self._log = open(log_path, "w", buffering=1)
        sys.stdout = self

    def write(self, msg):
        self._terminal.write(msg)
        self._log.write(msg)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        import sys
        sys.stdout = self._terminal
        self._log.close()


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    opts = _import_settings()

    OUTPUT_PATH = str(Path(opts.output_path).with_name("Inf_enmpc_felement=1.xlsx"))
    LOG_PATH    = str(Path(opts.output_path).with_name("Inf_enmpc_felement=1.txt"))
    _tee = _Tee(LOG_PATH)

    (m_plant, m_controller, sim_data, iteration_times, total_runtime,
     controller_lyapunov_function) = run_nmpc(opts=opts)

    all_nodes_p = [m_plant.node_p[n, :]
                   for n in m_plant.Nodes if str(n).startswith("sink")]

    sheets_keys_dict = {
        "compressor power": [m_plant.compressor_P[s, :]    for s in m_plant.Stations],
        "compressor beta":  [m_plant.compressor_beta[s, :] for s in m_plant.Stations],
        "wCons":            [m_plant.wCons[s, 0, :]        for s in m_plant.Nodes
                             if str(s).startswith("sink")],
        "node pressure":    all_nodes_p,
        "interm_w":         [m_plant.interm_w[p, vol, :]   for p, vol in m_plant.Pipes_VolExtrC_interm],
        "interm_p":         [m_plant.interm_p[p, vol, :]   for p, vol in m_plant.Pipes_VolExtrR_interm],
        "wSource":          [m_plant.wSource[s, :]         for s in m_plant.NodesSources],
        "pSource":          [m_plant.pSource[s, :]         for s in m_plant.NodesSources],
        "pipe_rho":         [m_plant.pipe_rho[p, vol, :]   for p, vol in m_plant.Pipes_VolExtrR],
    }

    write_data_to_excel(
        sim_data, m_plant, sheets_keys_dict, OUTPUT_PATH,
        controller_1_lyapunov=(controller_lyapunov_function
                               if controller_lyapunov_function else None),
    )

    _append_runtime_sheet(OUTPUT_PATH, iteration_times, total_runtime)
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)

    def _plot_from_sim(sim_data, slices, title, ylabel):
        plt.figure()
        for sl in slices:
            series = sim_data.get_data_from_key(sl)
            try:
                times = sorted(series.keys())
                values = [series[t] for t in times]
            except AttributeError:
                times = list(range(len(series)))
                values = list(series)
            plt.plot(times, values, label=str(sl))
        plt.xlabel("Time (hrs)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()

    _plot_from_sim(
        sim_data,
        [m_plant.compressor_beta[s, :] for s in m_plant.Stations],
        title="Compressor beta (closed loop)",
        ylabel="beta = Pout/Pin",
    )
    _plot_from_sim(
        sim_data,
        [m_plant.compressor_P[s, :] for s in m_plant.Stations],
        title="Compressor power (closed loop)",
        ylabel="Compressor power (scaled)",
    )
    plt.show()

    print("Done.  Output:", OUTPUT_PATH)
    print(f"Log saved to: {LOG_PATH}")
    _tee.close()
