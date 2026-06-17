import matplotlib.pyplot as plt
import pandas as pd
import pyomo.contrib.mpc as mpc
import pyomo.environ as pyo

from utils import plot_compressor_beta, plot_compressor_power, write_data_to_excel


def _report_solution(m_dyn):
    ib = m_dyn.inf_block

    obj_name = next((n for n in ('ObjFun', 'obj', 'objective')
                     if hasattr(m_dyn, n)), None)
    if obj_name is not None:
        print(f"objective = {pyo.value(getattr(m_dyn, obj_name)):.6e}")

    if hasattr(ib, 'terminal_error_constraint'):
        phase_info = getattr(m_dyn, '_phase_info', {})
        css_rho0 = phase_info.get('terminal_css_rho0', {})
        t_last = sorted(ib.time)[-1]
        max_err = max(
            abs(pyo.value(ib.pipe_rho[p, vol, t_last]) - target)
            for (p, vol), target in css_rho0.items()
        )
        print(f"endpoint max |pipe_rho - CSS| = {max_err:.6e}")


def _inf_df(var, idx_iter, time_list, label_fmt):
    data = {}
    for idx in idx_iter:
        col = label_fmt(idx)
        data[col] = [pyo.value(var[(*idx, t)]) for t in time_list]
    return pd.DataFrame(data, index=pd.Index(time_list, name="time"))


def _write_controller_output(m_dyn, output_path):
    ib = m_dyn.inf_block
    m_dyn_interface = mpc.DynamicModelInterface(m_dyn, m_dyn.Times)
    sim_data = m_dyn_interface.get_data_at_time(list(m_dyn.Times))

    all_nodes_p = []
    for n in m_dyn.Nodes:
        if str(n).startswith('sink'):
            all_nodes_p.append(m_dyn.node_p[n, :])

    sheets_keys_dict = {
        "compressor power": [m_dyn.compressor_P[s, :] for s in m_dyn.Stations],
        "compressor beta": [m_dyn.compressor_beta[s, :] for s in m_dyn.Stations],
        "wCons": [m_dyn.wCons[s, 0, :] for s in m_dyn.Nodes if str(s).startswith('sink')],
        "node pressure": all_nodes_p,
        "interm_w": [m_dyn.interm_w[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC_interm],
        "interm_p": [m_dyn.interm_p[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR_interm],
        "wSource": [m_dyn.wSource[s, :] for s in m_dyn.NodesSources],
        "pSource": [m_dyn.pSource[s, :] for s in m_dyn.NodesSources],
        "pipe_rho": [m_dyn.pipe_rho[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR],
    }
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, output_path)

    time_list = sorted(ib.time)
    pipes_interm = list(m_dyn.Pipes_VolExtrR_interm)
    stations = list(m_dyn.Stations)

    interm_p_inf_df = _inf_df(
        ib.interm_p, pipes_interm, time_list,
        lambda idx: f"interm_p[{idx[0]},{idx[1]},:]",
    )
    beta_inf_df = _inf_df(
        ib.compressor_beta, [(c,) for c in stations], time_list,
        lambda idx: f"compressor_beta[{idx[0]},:]",
    )
    phi_P_df = pd.DataFrame(
        {"phi_P[:]": [pyo.value(ib.phi_P[t]) for t in time_list]},
        index=pd.Index(time_list, name="time"),
    )

    with pd.ExcelWriter(output_path, engine="openpyxl",
                        mode="a", if_sheet_exists="replace") as w:
        interm_p_inf_df.to_excel(w, sheet_name="inf_interm_p")
        beta_inf_df.to_excel(w, sheet_name="inf_compressor_beta")
        phi_P_df.to_excel(w, sheet_name="inf_phi_P")


def _plot_results(m_dyn):
    plot_compressor_beta(m_dyn)
    plot_compressor_power(m_dyn)

    speed = {}
    for p in m_dyn.Pipes:
        speed[p] = []
        for t in m_dyn.Times:
            sum_u_at_t = sum(
                abs(pyo.value(m_dyn.u[p, v[1], t]))
                for v in m_dyn.Pipes_VolExtrC
            )
            speed[p].append(sum_u_at_t / len(m_dyn.Pipes_VolExtrC))

    plt.figure()
    for key in speed.keys():
        plt.plot(speed[key], label=key)
    plt.legend()
    plt.show()


def save_and_plot_results(m_dyn, output_path):
    _report_solution(m_dyn)
    _write_controller_output(m_dyn, str(output_path))
    _plot_results(m_dyn)
