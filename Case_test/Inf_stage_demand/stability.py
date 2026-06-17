"""Lyapunov stability constraint on interm_p for the infinite-horizon NMPC.

V = sum_{t != last} sum_(p,vol) (interm_p - interm_p_css_fin)^2 + inf_block.phi_V[tau=1]
V_k <= V_{k-1} - delta * plant_stage_cost_{k-1}
"""
import pyomo.environ as pyo


def apply_stability_constraint(m_controller, delta=0.05):
    ib = m_controller.inf_block
    tau_last = sorted(ib.time)[-1]

    m_controller.interm_p_css_fin = pyo.Param(
        m_controller.Pipes_VolExtrR_interm, m_controller.Times,
        mutable=True, initialize=55.0,
    )
    m_controller.lyapunov_function_current = pyo.Var(initialize=1.0)
    m_controller.lyapunov_function_prev = pyo.Param(initialize=1.0, mutable=True)
    m_controller.tracking_cost_plant_prev = pyo.Param(initialize=1.0, mutable=True)
    m_controller.delta = pyo.Param(initialize=delta, mutable=True)

    def _definition(m):
        finite = sum(
            (m.interm_p[p, vol, t] - m.interm_p_css_fin[p, vol, t]) ** 2
            for (p, vol) in m.Pipes_VolExtrR_interm
            for t in m.Times if t != m.Times.last()
        )
        return m.lyapunov_function_current == finite + m.inf_block.phi_V[tau_last]
    m_controller.lyapunov_function_definition = pyo.Constraint(rule=_definition)

    def _stability(m):
        return (m.lyapunov_function_current
                <= m.lyapunov_function_prev - m.delta * m.tracking_cost_plant_prev)
    m_controller.stability_constraint = pyo.Constraint(rule=_stability)

    print(f"[stability] Lyapunov-on-interm_p added (finite + phi_V tail), delta={delta}")
