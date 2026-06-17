import math

import pandas as pd
import pyomo.environ as pyo
from pyomo.common.collections import ComponentMap
from pyomo.core.expr.visitor import identify_variables, replace_expressions
from pyomo.dae import ContinuousSet, DerivativeVar

from utils import _norm


def _is_unit_time(t):
    try:
        return float(t) == 1.0
    except (TypeError, ValueError):
        return False


def _fix_infinite_wcons_from_css(ib, ocss_file_path, T_finite, L, T_period, time_points):
    try:
        wcons_df = pd.read_excel(ocss_file_path, sheet_name="wCons", index_col=0)
    except Exception as err:
        ib.wCons.fix(0.0)
        print(f"[inf-block] WARNING: could not read CSS wCons sheet ({err}); "
              "wCons fixed to 0 on the infinite tail")
        return

    col_by_norm = {_norm(c): c for c in wcons_df.columns}
    n_fixed = 0
    n_css = 0
    n_zero = 0

    for idx in ib.WConsIndex:
        compo0, compo1 = idx
        target = _norm(f"wCons[{compo0},{compo1},:]")
        col = col_by_norm.get(target)
        if col is None:
            col = next((c for c in wcons_df.columns
                        if _norm(str(compo0)) in _norm(c)
                        and _norm(str(compo1)) in _norm(c)), None)

        for tau in time_points:
            if col is None:
                val = 0.0
                n_zero += 1
            else:
                tau_float = float(tau)
                if _is_unit_time(tau_float):
                    phase = 0.0
                else:
                    t_real = T_finite + L * math.atanh(tau_float)
                    phase = t_real % T_period
                closest = min(wcons_df.index, key=lambda x: abs(float(x) - phase))
                val = float(wcons_df.loc[closest, col])
                n_css += 1
            ib.wCons[compo0, compo1, tau].set_value(val)
            ib.wCons[compo0, compo1, tau].fix()
            n_fixed += 1

    print(f"[inf-block] wCons fixed on infinite tail at {n_fixed} point(s) "
          f"({n_css} from CSS, {n_zero} default zero)")


def _init_infinite_interm_p_css(ib, m_dyn, ocss_file_path,
                                T_finite, L, T_period, time_points):
    """Initialise ib.interm_p_css on the tail (sim_t0 = 0; rolled later in run_MPC):
        phase(tau)   = (T_finite + L*atanh(tau)) mod T_period
        phase(tau=1) = 0
    """
    try:
        df = pd.read_excel(ocss_file_path, sheet_name="interm_p", index_col=0)
    except Exception as err:
        print(f"[inf-block] WARNING: could not read CSS interm_p sheet ({err})")
        return

    col_by_norm = {_norm(c): c for c in df.columns}
    for (p, vol) in m_dyn.Pipes_VolExtrR_interm:
        col = col_by_norm.get(_norm(f"interm_p[{p},{vol},:]"))
        if col is None:
            col = next((c for c in df.columns
                        if _norm(str(p)) in _norm(c) and _norm(str(vol)) in _norm(c)),
                       None)
        if col is None:
            continue
        for tau in time_points:
            phase = 0.0 if _is_unit_time(float(tau)) else \
                (T_finite + L * math.atanh(float(tau))) % T_period
            closest = min(df.index, key=lambda x: abs(float(x) - phase))
            ib.interm_p_css[p, vol, tau] = float(df.loc[closest, col])


def _clone_time_indexed_constraint_to_infinite(m_dyn, con_name):
    ib = m_dyn.inf_block
    con = getattr(m_dyn, con_name)
    finite_times = set(m_dyn.Times)
    template_expr_by_prefix = {}

    for idx in con:
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        if not idx_tuple or idx_tuple[-1] not in finite_times:
            continue
        prefix = idx_tuple[:-1]
        if prefix in template_expr_by_prefix:
            continue

        expr = con[idx].expr
        if expr is None:
            continue

        template_time = idx_tuple[-1]
        used_times = set()
        for var in identify_variables(expr, include_fixed=True):
            var_idx = var.index()
            var_idx_tuple = var_idx if isinstance(var_idx, tuple) else (var_idx,)
            if var_idx_tuple and var_idx_tuple[-1] in finite_times:
                used_times.add(var_idx_tuple[-1])
        if len(used_times) > 1:
            raise ValueError(
                f"Cannot clone {con_name}{idx}: expression contains multiple "
                f"finite time points {sorted(used_times)}. Use collocation "
                "equations for the infinite-tail clone."
            )
        if used_times and template_time not in used_times:
            raise ValueError(
                f"Cannot clone {con_name}{idx}: template index time "
                f"{template_time} does not match expression time {used_times}."
            )

        template_expr_by_prefix[prefix] = expr

    if not template_expr_by_prefix:
        return 0

    prefixes = list(template_expr_by_prefix)
    prefix_dimen = len(prefixes[0])
    if prefix_dimen == 0:
        raise ValueError(
            f"Cannot clone scalar time-only constraint {con_name}; expected "
            "a spatial index plus time."
        )
    prefix_set = pyo.Set(dimen=prefix_dimen, initialize=prefixes)
    ib.add_component(f"ib_{con_name}_index", prefix_set)

    def _rule(b, *full_idx):
        prefix = full_idx[:-1]
        tau = full_idx[-1]
        expr = template_expr_by_prefix[prefix]
        replacement_map = ComponentMap()

        for var in identify_variables(expr, include_fixed=True):
            parent = var.parent_component()
            parent_name = parent.name.split('.')[-1]
            if not hasattr(b, parent_name):
                continue

            var_idx = var.index()
            var_idx_tuple = var_idx if isinstance(var_idx, tuple) else (var_idx,)
            if not var_idx_tuple or var_idx_tuple[-1] not in finite_times:
                continue

            inf_parent = getattr(b, parent_name)
            inf_idx = var_idx_tuple[:-1] + (tau,)
            replacement_map[var] = inf_parent[inf_idx]

        return replace_expressions(expr, substitution_map=replacement_map)

    clone = pyo.Constraint(prefix_set, ib.time, rule=_rule)
    ib.add_component(f"ib_{con_name}", clone)
    return len(prefixes) * len(ib.time)


def _add_infinite_pipe_mass_balance_collocation(m_dyn):
    ib = m_dyn.inf_block
    scale_w = 1.0

    def _w(p, vol, t):
        N_end = m_dyn.Nvol[p]
        if vol == 1:
            return ib.inlet_w[p, t]
        if vol == N_end:
            return ib.outlet_w[p, t]
        return ib.interm_w[p, vol, t]

    def _pipe_mass_rule(b, p, vol, t):
        if _is_unit_time(t):
            return pyo.Constraint.Skip

        N = m_dyn.Nvol[p]
        V_cell = m_dyn.Area[p] * m_dyn.Length[p] / (N - 1)
        w = _w(p, vol, t)
        w_next = _w(p, vol + 1, t)
        cons = b.wCons[p, vol, t]

        return (b.gamma * (1 - t**2) * b.drhodt[p, vol + 1, t] * V_cell
                == (w - w_next - cons) * scale_w)

    ib.ib_PIPE_mass_balance = pyo.Constraint(
        m_dyn.Pipes_VolCenterC, ib.time, rule=_pipe_mass_rule,
    )
    return len(m_dyn.Pipes_VolCenterC) * sum(
        1 for t in ib.time if not _is_unit_time(t)
    )


def _clone_finite_equations_to_infinite(m_dyn):
    ib = m_dyn.inf_block

    ib.node_p = pyo.Var(
        m_dyn.Nodes, ib.time, within=pyo.NonNegativeReals, initialize=55.0,
    )
    ib.inlet_w = pyo.Var(
        m_dyn.Arcs, ib.time, within=pyo.Reals, initialize=0.0,
    )
    ib.outlet_w = pyo.Var(
        m_dyn.Arcs, ib.time, within=pyo.Reals, initialize=0.0,
    )
    ib.interm_w = pyo.Var(
        m_dyn.Pipes_VolExtrC_interm, ib.time,
        within=pyo.Reals, initialize=0.0,
    )
    ib.u2 = pyo.Var(
        m_dyn.Pipes_VolExtrC, ib.time, within=pyo.Reals, initialize=0.001,
    )
    ib.u = pyo.Var(
        m_dyn.Pipes_VolExtrC, ib.time, within=pyo.Reals, initialize=0.1,
    )
    ib.pipe_rho = pyo.Var(
        m_dyn.Pipes_VolExtrR, ib.time,
        within=pyo.NonNegativeReals, initialize=40.0,
    )
    ib.drhodt = DerivativeVar(ib.pipe_rho, wrt=ib.time)
    ib.compressor_eta = pyo.Var(
        m_dyn.Stations, ib.time, bounds=(0, 1),
        within=pyo.NonNegativeReals, initialize=0.8,
    )
    ib.compressor_eta.fix(0.8)

    wcons_pairs = sorted({idx[:-1] for idx in m_dyn.wCons})
    ib.WConsIndex = pyo.Set(dimen=2, initialize=wcons_pairs)
    ib.wCons = pyo.Var(
        ib.WConsIndex, ib.time, within=pyo.Reals, initialize=0.0,
    )

    ib.pSource = pyo.Var(
        m_dyn.NodesSources, ib.time,
        within=pyo.NonNegativeReals, initialize=55.0,
    )
    ib.wSource = pyo.Var(
        m_dyn.NodesSources, ib.time,
        within=pyo.NonNegativeReals, initialize=158.0,
    )

    clone_counts = {
        "PIPE_mass_balance": _add_infinite_pipe_mass_balance_collocation(m_dyn)
    }
    clone_names = [
        "NODE_mass_balance",
        "NODE_SOURCE_pressure",
        "NODE_SOURCE_mass_balance",
        "STATION_mass_balance",
        "STATION_pressure_ratio",
        "STATION_power_balance",
        "VALVE_mass_balance",
        "VALVE_pressure_flow",
        "PIPE_momentum_balance",
        "PIPE_nonlinear_speed",
        "PIPE_nonlinear_rho",
        "PIPE_nonlinear_friction_smooth",
    ]
    clone_counts.update({
        name: _clone_time_indexed_constraint_to_infinite(m_dyn, name)
        for name in clone_names
        if hasattr(m_dyn, name)
    })

    print(f"[inf-block] cloned finite-model algebraic constraints and added "
          f"collocation pipe mass balance onto infinite block "
          f"({sum(clone_counts.values())} entries across {len(clone_counts)} components).")


def _prune_inf_block_to_collocation_points(m_dyn):
    ib = m_dyn.inf_block

    keep_times = set()
    for con in ib.component_objects(pyo.Constraint, active=True):
        if '_disc_eq' in con.name:
            for idx in con:
                t = idx[-1] if isinstance(idx, tuple) else idx
                keep_times.add(t)

    if not keep_times:
        print("[inf-block] WARNING: no disc_eq found; skipping collocation pruning")
        return m_dyn

    all_times = set(ib.time)
    prune_times = all_times - keep_times
    keep_names = {'pipe_rho', 'interm_p', 'phi_P', 'phi_V'}

    n_var = n_con = 0
    for comp in ib.component_objects(pyo.Var, active=True):
        if comp.name.split('.')[-1] in keep_names:
            continue
        for idx in list(comp):
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t in prune_times:
                del comp[idx]
                n_var += 1

    for con in ib.component_objects(pyo.Constraint, active=True):
        if '_disc_eq' in con.name or '_time_cont_eq' in con.name:
            continue
        for idx in list(con):
            t = idx[-1] if isinstance(idx, tuple) else idx
            if t in all_times and t in prune_times:
                del con[idx]
                n_con += 1

    print(f"[inf-block] collocation pruning (CSTR-style): -{n_var} var entries, "
          f"-{n_con} constraint entries (pipe_rho, interm_p & phi_P kept at all times; "
          f"{len(prune_times)} boundary points pruned, {len(keep_times)} Gauss points kept)")
    return m_dyn


def _load_css_zero(ocss_file_path, sheet, attr_name, index_set):
    try:
        df = pd.read_excel(ocss_file_path, sheet_name=sheet, index_col=0)
    except Exception as err:
        print(f"[inf-block] WARNING: could not read CSS {sheet} sheet ({err}); "
              "terminal CSS constraint skipped")
        return {}

    closest = min(df.index, key=lambda x: abs(float(x) - 0.0))
    col_by_norm = {_norm(c): c for c in df.columns}
    values = {}
    for idx in index_set:
        idx_tuple = idx if isinstance(idx, tuple) else (idx,)
        if len(idx_tuple) == 1:
            target = _norm(f"{attr_name}[{idx_tuple[0]},:]")
        else:
            target = _norm(f"{attr_name}[{idx_tuple[0]},{idx_tuple[1]},:]")
        col = col_by_norm.get(target)
        if col is None:
            needed = [_norm(str(k)) for k in idx_tuple]
            col = next((c for c in df.columns
                        if all(n in _norm(c) for n in needed)), None)
        if col is not None:
            values[idx_tuple] = float(df.loc[closest, col])
    return values


def _add_infinite_block(m_dyn, ocss_file_path, T_period=24.0,
                        nfe_inf=10, ncp_inf=3, L=10.0,
                        collocation_scheme='LAGRANGE-LEGENDRE',
                        remove_collocation=True,
                        terminal_css_constraint=True,
                        add_lyapunov_phi_V=False):
    terminal_css_constraint = bool(terminal_css_constraint)

    T_finite = float(m_dyn.Times.last())
    gamma = 1.0 / L

    m_dyn.inf_block = pyo.Block()
    ib = m_dyn.inf_block
    ib.time = ContinuousSet(bounds=(0, 1))

    print(f"[inf-block] Creating infinite block with full dynamics (gamma={gamma:.6f})")

    ib.interm_p = pyo.Var(
        m_dyn.Pipes_VolExtrR_interm, ib.time,
        within=pyo.NonNegativeReals, initialize=55.0,
    )
    ib.compressor_beta = pyo.Var(
        m_dyn.Stations, ib.time, bounds=(1.05, 2),
        within=pyo.NonNegativeReals, initialize=1.5,
    )
    ib.compressor_P = pyo.Var(
        m_dyn.Stations, ib.time,
        within=pyo.NonNegativeReals, initialize=100.0,
    )
    ib.phi_P = pyo.Var(ib.time, initialize=0.0, within=pyo.NonNegativeReals)
    ib.dphi_P_dt = DerivativeVar(ib.phi_P, wrt=ib.time)
    ib.gamma = pyo.Param(initialize=gamma, mutable=True)

    # Lyapunov integral state (rho tracking) — built only when stability is on,
    # before discretization so collocation builds its disc/cont eqs like phi_P.
    if add_lyapunov_phi_V:
        ib.phi_V = pyo.Var(ib.time, initialize=0.0, within=pyo.NonNegativeReals)
        ib.dphi_V_dt = DerivativeVar(ib.phi_V, wrt=ib.time)

    print("[inf-block] Cloning dynamics equations from finite block...")
    _clone_finite_equations_to_infinite(m_dyn)

    discretizer = pyo.TransformationFactory('dae.collocation')
    discretizer.apply_to(
        m_dyn, wrt=ib.time, nfe=nfe_inf, ncp=ncp_inf,
        scheme=collocation_scheme,
    )

    time_points = sorted(ib.time)
    print(f"[inf-block] Discretized: nfe={nfe_inf}, ncp={ncp_inf} "
          f"=> {len(time_points)} collocation points using {collocation_scheme}")

    if 'compressorStation_1' in [str(s) for s in m_dyn.Stations]:
        ib.compressor_beta['compressorStation_1', :].setub(4)

    tf_for_bounds = m_dyn.Times.last()
    for n in m_dyn.Nodes:
        lb = m_dyn.node_p[n, tf_for_bounds].lb
        ub = m_dyn.node_p[n, tf_for_bounds].ub
        for t in time_points:
            if lb is not None:
                ib.node_p[n, t].setlb(lb)
            if ub is not None:
                ib.node_p[n, t].setub(ub)
    for idx in m_dyn.Pipes_VolExtrR_interm:
        lb = m_dyn.interm_p[idx[0], idx[1], tf_for_bounds].lb
        ub = m_dyn.interm_p[idx[0], idx[1], tf_for_bounds].ub
        for t in time_points:
            if lb is not None:
                ib.interm_p[idx[0], idx[1], t].setlb(lb)
            if ub is not None:
                ib.interm_p[idx[0], idx[1], t].setub(ub)
    for idx in m_dyn.Pipes_VolExtrR:
        lb = m_dyn.pipe_rho[idx[0], idx[1], tf_for_bounds].lb
        ub = m_dyn.pipe_rho[idx[0], idx[1], tf_for_bounds].ub
        for t in time_points:
            if lb is not None:
                ib.pipe_rho[idx[0], idx[1], t].setlb(lb)
            if ub is not None:
                ib.pipe_rho[idx[0], idx[1], t].setub(ub)

    _fix_infinite_wcons_from_css(
        ib, ocss_file_path, T_finite, L, T_period, time_points,
    )

    def _phi_P_ode_rule(b, t):
        if t == b.time.first() or _is_unit_time(t):
            return pyo.Constraint.Skip
        power_sum = sum(b.compressor_P[c, t] for c in m_dyn.Stations) / 1000.0
        return b.gamma * (1 - t**2) * b.dphi_P_dt[t] == power_sum

    ib.phi_P_ode = pyo.Constraint(ib.time, rule=_phi_P_ode_rule)
    ib.phi_P[0].fix(0.0)

    # Lyapunov integral state phi_V: gamma*(1-tau^2) carries the tau->real-time
    # transform, so phi_V[tau=1] = integral over real time of the interm_p stage cost.
    if add_lyapunov_phi_V:
        ib.interm_p_css = pyo.Param(
            m_dyn.Pipes_VolExtrR_interm, ib.time, mutable=True, initialize=55.0,
        )
        _init_infinite_interm_p_css(
            ib, m_dyn, ocss_file_path, T_finite, L, T_period, time_points,
        )

        def _phi_V_ode_rule(b, t):
            if t == b.time.first() or _is_unit_time(t):
                return pyo.Constraint.Skip
            stage = sum(
                (b.interm_p[p, vol, t] - b.interm_p_css[p, vol, t]) ** 2
                for (p, vol) in m_dyn.Pipes_VolExtrR_interm
            )
            return b.gamma * (1 - t**2) * b.dphi_V_dt[t] == stage

        ib.phi_V_ode = pyo.Constraint(ib.time, rule=_phi_V_ode_rule)
        ib.phi_V[0].fix(0.0)
        print("[inf-block] added Lyapunov integral state phi_V (interm_p tracking)")

    tf = m_dyn.Times.last()
    t0 = ib.time.first()
    n_ps_fixed = 0
    for s in m_dyn.NodesSources:
        val = pyo.value(m_dyn.pSource[s, tf])
        for t in time_points:
            ib.pSource[s, t].set_value(val)
            ib.pSource[s, t].fix()
            n_ps_fixed += 1
    print(f"[inf-block] pSource fixed on infinite tail at {n_ps_fixed} point(s).")

    def _link_pipe_rho(m, p, vol):
        return m.pipe_rho[p, vol, tf] == m.inf_block.pipe_rho[p, vol, t0]

    m_dyn.link_pipe_rho = pyo.Constraint(
        m_dyn.Pipes_VolExtrR_interm, rule=_link_pipe_rho)
    print("[inf-block] pipe_rho (differential state) linked at finite-infinite boundary")

    css_rho0 = _load_css_zero(
        ocss_file_path, "interm_p", "interm_p", m_dyn.Pipes_VolExtrR_interm,
    )
    print(f"[inf-block] Terminal CSS hard constraint at tau=1: "
          f"{len(css_rho0) if terminal_css_constraint else 0} interm_p entries compared to CSS t=0")

    m_dyn._phase_info = {
        'T_period': T_period,
        'T_finite': T_finite,
        'L': L,
        'gamma': gamma,
        'ocss_file_path': ocss_file_path,
        'terminal_css_constraint': terminal_css_constraint,
        'terminal_css_rho0': css_rho0 if terminal_css_constraint else {},
        'time_points': list(time_points),
    }

    if remove_collocation:
        m_dyn = _prune_inf_block_to_collocation_points(m_dyn)

    return m_dyn


def _add_hard_terminal_constraint(m_dyn, terminal_error_bound=1.0e-6):
    ib = m_dyn.inf_block
    phase_info = getattr(m_dyn, '_phase_info', {})
    time_points = sorted(ib.time)
    t_last = time_points[-1]
    css_rho0 = phase_info.get('terminal_css_rho0', {})

    if not css_rho0:
        print("[constraint] WARNING: no terminal_css_rho0 found, skipping hard constraint")
        return m_dyn

    # The CSS reference is stored as a MUTABLE Param so the receding-horizon
    # loop (run_MPC) can roll the endpoint reference phase each MPC step.
    ib.terminal_css_index = pyo.Set(dimen=2, initialize=list(css_rho0.keys()))
    ib.terminal_css_target = pyo.Param(
        ib.terminal_css_index, mutable=True,
        initialize={k: css_rho0[k] for k in css_rho0},
    )
    ib.terminal_error_bound = pyo.Param(
        mutable=True, initialize=terminal_error_bound)
    ib.terminal_t_last = t_last

    terminal_error = sum(
        (ib.interm_p[p, vol, t_last] - ib.terminal_css_target[p, vol]) ** 2
        for (p, vol) in ib.terminal_css_index
    )
    ib.terminal_error_constraint = pyo.Constraint(
        expr=terminal_error <= ib.terminal_error_bound)
    print(f"[constraint] Added aggregated terminal constraint: "
          f"sum_(p,vol) (interm_p[:,:,tau=1] - CSS)^2 <= {terminal_error_bound:g} "
          f"over {len(css_rho0)} interm_p entries")

    return m_dyn
