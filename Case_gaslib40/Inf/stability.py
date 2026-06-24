"""Lyapunov stability constraint on interm_p/pSource3 for the infinite-horizon NMPC.

V = finite tracking error / delta_finite + inf_block.phi_V[tau=1] / lyapunov_inf_scale
Before step_change:
    V_k <= (1 - delta_percent) * V_{k-1}
After step_change:
    V_k <= V_{k-1} - delta * plant_first_stage_cost
"""
import pyomo.environ as pyo

TRACKED_PSOURCE_NAMES = {'source_3'}


def apply_stability_constraint(
        m_controller, delta=0.05, lyapunov_inf_scale=1.0,
        delta_percent=0.0, delta_finite=1.0):
    ib = m_controller.inf_block
    tau_last = sorted(ib.time)[-1]
    lyapunov_inf_scale = float(lyapunov_inf_scale)
    if lyapunov_inf_scale <= 0:
        raise ValueError("lyapunov_inf_scale must be positive")
    delta_finite = float(delta_finite)
    if delta_finite <= 0:
        raise ValueError("delta_finite must be positive")
    delta_percent = float(delta_percent)
    if not 0 <= delta_percent < 1:
        raise ValueError("delta_percent must be in [0, 1)")
    tracked_psources = [
        s for s in m_controller.NodesSources
        if str(s) in TRACKED_PSOURCE_NAMES
    ]

    m_controller.interm_p_css_fin = pyo.Param(
        m_controller.Pipes_VolExtrR_interm, m_controller.Times,
        mutable=True, initialize=55.0,
    )
    m_controller.lyapunov_pSource_index = pyo.Set(
        initialize=tracked_psources,
    )
    m_controller.pSource_css_fin = pyo.Param(
        m_controller.lyapunov_pSource_index, m_controller.Times,
        mutable=True, initialize=55.0,
    )
    m_controller.lyapunov_function_current = pyo.Var(initialize=1.0)
    m_controller.lyapunov_function_prev = pyo.Param(initialize=1.0, mutable=True)
    m_controller.tracking_cost_plant_prev = pyo.Param(initialize=1.0, mutable=True)
    m_controller.delta = pyo.Param(initialize=delta, mutable=True)
    m_controller.delta_percent = pyo.Param(
        initialize=delta_percent, mutable=True)
    m_controller.use_first_stage_cost_descent = pyo.Param(
        initialize=0.0, mutable=True)
    m_controller.lyapunov_inf_scale = pyo.Param(
        initialize=lyapunov_inf_scale, mutable=True)
    m_controller.delta_finite = pyo.Param(
        initialize=delta_finite, mutable=True)

    def _definition(m):
        finite = sum(
            (m.interm_p[p, vol, t] - m.interm_p_css_fin[p, vol, t]) ** 2
            for (p, vol) in m.Pipes_VolExtrR_interm
            for t in m.Times if t != m.Times.last()
        )
        finite += sum(
            (m.pSource[s, t] - m.pSource_css_fin[s, t]) ** 2
            for s in m.lyapunov_pSource_index
            for t in m.Times if t != m.Times.last()
        )
        return (m.lyapunov_function_current
                == finite / m.delta_finite
                + m.inf_block.phi_V[tau_last] / m.lyapunov_inf_scale)
    m_controller.lyapunov_function_definition = pyo.Constraint(rule=_definition)

    def _stability(m):
        percent_bound = (1.0 - m.delta_percent) * m.lyapunov_function_prev
        first_stage_bound = (
            m.lyapunov_function_prev
            - m.delta * m.tracking_cost_plant_prev
        )
        switch = m.use_first_stage_cost_descent
        return (m.lyapunov_function_current
                <= (1.0 - switch) * percent_bound
                + switch * first_stage_bound)
    m_controller.stability_constraint = pyo.Constraint(rule=_stability)

    print(f"[stability] Lyapunov-on-interm_p+pSource3 added "
          f"(finite/{delta_finite:g} + phi_V/{lyapunov_inf_scale:g} tail), "
          f"delta_percent={delta_percent:g}, plant_delta={delta:g}")
