import pyomo.environ as pyo
import pandas as pd

# only for tf = 72
def terminal_constraints(m, w_terminal, p_terminal):
    def terminal_flow(m, p, vol):
        tf = m.Times.last()
        return m.interm_w[p, vol, tf] >= w_terminal
    m.terminal_flow = pyo.Constraint(m.Pipes_VolExtrC)

    def terminal_pressure(m, p, vol):
        tf = m.Times.last()
        return m.interm_p[p, vol, tf] >= p_terminal
    m.terminal_pressure = pyo.Constraint(m.Pipes_VolExtrC)

    return m

def load_css(m_controller, ocss_file_path):
    needed_sheets = ['interm_p', 'compressor beta', 'compressor power', 'wSource', 'pSource']
    df = pd.read_excel(ocss_file_path, sheet_name=needed_sheets, index_col='Unnamed: 0')

    css_period = len(df['interm_p'].index) - 1 # # of \interm_p column -1, eg: 25-1
    m_controller._css_period = css_period
    m_controller._css_file_path = ocss_file_path
    m_controller._css_data = df

    m_controller.interm_p_ocss = pyo.Var(m_controller.Pipes_VolExtrR_interm, m_controller.Times)
    for p, vol in m_controller.Pipes_VolExtrR_interm:
        column_name = "interm_p['" + str(p) + "', " + str(vol) +", :]"
        for t in m_controller.Times:
            m_controller.interm_p_ocss[p, vol, t] = df['interm_p'][column_name][t % css_period]

    m_controller.compressor_beta_ocss = pyo.Var(m_controller.Stations, m_controller.Times)
    for s in m_controller.Stations:
        column_name = "compressor_beta['" + str(s) + "', :]"
        for t in m_controller.Times:
            m_controller.compressor_beta_ocss[s, t] = df['compressor beta'][column_name][t % css_period]

    m_controller.compressor_P_ocss = pyo.Var(m_controller.Stations, m_controller.Times)
    for s in m_controller.Stations:
        column_name = "compressor_P['" + str(s) + "', :]"
        for t in m_controller.Times:
            m_controller.compressor_P_ocss[s, t] = df['compressor power'][column_name][t % css_period]

    m_controller.wSource_ocss = pyo.Var(m_controller.NodesSources, m_controller.Times)
    for s in m_controller.NodesSources:
        column_name = "wSource['" + str(s) + "', :]"
        for t in m_controller.Times:
            m_controller.wSource_ocss[s, t] = df['wSource'][column_name][t % css_period]

    m_controller.pSource_ocss = pyo.Var(m_controller.NodesSources, m_controller.Times)
    for s in m_controller.NodesSources:
        column_name = "pSource['" + str(s) + "', :]"
        for t in m_controller.Times:
            m_controller.pSource_ocss[s, t] = df['pSource'][column_name][t % css_period]

    m_controller.compressor_beta_ocss.fix()
    m_controller.compressor_P_ocss.fix()
    m_controller.interm_p_ocss.fix()
    m_controller.wSource_ocss.fix()
    m_controller.pSource_ocss.fix()


def _css_column(attr_name, *idx):
    if len(idx) == 1:
        return f"{attr_name}['{idx[0]}', :]"
    if len(idx) == 2:
        return f"{attr_name}['{idx[0]}', {idx[1]}, :]"
    raise ValueError(f"Unsupported CSS index for {attr_name}: {idx}")


def _css_value(m, sheet, column, phase):
    df = m._css_data[sheet]
    period = float(m._css_period)
    phase = float(phase) % period
    closest = min(df.index, key=lambda x: abs(float(x) - phase))
    return float(df.loc[closest, column])


def refresh_css_references_for_horizon(m, sim_t0):
    for t in m.Times:
        phase = float(sim_t0) + float(t)

        for p, vol in m.Pipes_VolExtrR_interm:
            col = _css_column("interm_p", p, vol)
            m.interm_p_ocss[p, vol, t].set_value(
                _css_value(m, "interm_p", col, phase))

        for c in m.Stations:
            col = _css_column("compressor_beta", c)
            m.compressor_beta_ocss[c, t].set_value(
                _css_value(m, "compressor beta", col, phase))

            col = _css_column("compressor_P", c)
            m.compressor_P_ocss[c, t].set_value(
                _css_value(m, "compressor power", col, phase))

        for s in m.NodesSources:
            col = _css_column("wSource", s)
            m.wSource_ocss[s, t].set_value(
                _css_value(m, "wSource", col, phase))

            col = _css_column("pSource", s)
            m.pSource_ocss[s, t].set_value(
                _css_value(m, "pSource", col, phase))

    if hasattr(m, "terminal_interm_p_target"):
        update_css_terminal_targets(m, phase=m.Times.last())

    return m

# for t0 = 0, and tf = 24
def css_terminal_constraints(m, num_time_periods=3, horizon=72, ocss_file_path=None):
    load_css(m, ocss_file_path)

    N = num_time_periods # N = 1
    K = int(horizon / N) # K = 24 / 1 = 24

    def _terminal_controls(m, c):
        tf = N * K
        t0 = (N-1) * K # last point of last 2 iterations the same
        return m.compressor_P[c, tf] == m.compressor_P[c, t0]
    m.terminal_controls_constraint = pyo.Constraint(m.Stations, rule=_terminal_controls)

    def _terminal_pressure(m, p, vol):
        tf = N * K
        t0 = (N-1) * K
        return m.interm_p[p, vol, tf] == m.interm_p[p, vol, t0]
    m.terminal_pressure = pyo.Constraint(m.Pipes_VolExtrR_interm, rule=_terminal_pressure)

    def _terminal_source_pressure(m, s):
        tf = N * K
        t0 = (N-1) * K
        return m.pSource['source_3', tf] == m.pSource['source_3', t0]
    m.terminal_source_pressure = pyo.Constraint(rule=_terminal_source_pressure)

    return m


def css_terminal_constraints_each_point(m, num_time_periods=3, horizon=72, ocss_file_path=None):
    load_css(m, ocss_file_path)

    N = num_time_periods
    K = int(horizon/N)
    m.last_period = pyo.Set(initialize=[t for t in m.Times if t >= (N-1)*K and t<= N*K])

    # for the start of the last period t = 48
    def _terminal_pressure(m, p, vol):
        return m.interm_p[p, vol, (N-1)*K] == m.interm_p_ocss[p, vol, 0]
    m.terminal_pressure = pyo.Constraint(m.Pipes_VolExtrR_interm, rule=_terminal_pressure)

    # for t = 48 - 72
    def _terminal_controls(m, c, t):
        return m.compressor_P_ocss[c, t] == m.compressor_P[c, t]
    m.terminal_controls_constraint = pyo.Constraint(m.Stations, m.last_period, rule=_terminal_controls)

    return m


def css_terminal_constraints_final_to_ocss_phase0(m, ocss_file_path=None):
    load_css(m, ocss_file_path)

    tf = m.Times.last()
    m.terminal_interm_p_target = pyo.Param(
        m.Pipes_VolExtrR_interm, initialize=0.0, mutable=True)
    m.terminal_compressor_P_target = pyo.Param(
        m.Stations, initialize=0.0, mutable=True)
    update_css_terminal_targets(m, phase=0)

    def _terminal_pressure(m, p, vol):
        return m.interm_p[p, vol, tf] == m.terminal_interm_p_target[p, vol]
    m.terminal_pressure_phase0 = pyo.Constraint(
        m.Pipes_VolExtrR_interm, rule=_terminal_pressure)

    def _terminal_controls(m, c):
        return m.compressor_P[c, tf] == m.terminal_compressor_P_target[c]
    m.terminal_controls_phase0 = pyo.Constraint(
        m.Stations, rule=_terminal_controls)

    return m


def _ocss_time_point(m, phase):
    css_period = getattr(m, '_css_period', None)
    if css_period is not None:
        phase = float(phase) % float(css_period)

    times = list(m.Times)
    for t in times:
        if abs(float(t) - float(phase)) <= 1e-9:
            return t
    return min(times, key=lambda t: abs(float(t) - float(phase)))


def update_css_terminal_targets(m, phase):
    t_ref = _ocss_time_point(m, phase)

    for p, vol in m.Pipes_VolExtrR_interm:
        m.terminal_interm_p_target[p, vol].set_value(
            pyo.value(m.interm_p_ocss[p, vol, t_ref]))

    for c in m.Stations:
        m.terminal_compressor_P_target[c].set_value(
            pyo.value(m.compressor_P_ocss[c, t_ref]))

    return t_ref
