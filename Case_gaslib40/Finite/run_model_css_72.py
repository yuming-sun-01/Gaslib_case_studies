from utils import import_data_from_excel
from utils import debug_gas_model
from utils import dynamic_demand_calculation, uncertain_demand_calculation
from utils import write_data_to_excel
from model import buildNonLinearModel
from model import init_network_default
from terminal import css_terminal_constraints_each_point
import json
import os
import time
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from idaes.core.util.model_statistics import degrees_of_freedom

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def write_init_values_to_excel(m_dyn, init_output_path):
    """Write the current variable .value entries, before the final solve."""
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
        "inlet_w": [m_dyn.inlet_w[a, :] for a in m_dyn.Arcs],
        "outlet_w": [m_dyn.outlet_w[a, :] for a in m_dyn.Arcs],
        "interm_w": [m_dyn.interm_w[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC_interm],
        "interm_p": [m_dyn.interm_p[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR_interm],
        "u2": [m_dyn.u2[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC],
        "u": [m_dyn.u[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC],
        "wSource": [m_dyn.wSource[s, :] for s in m_dyn.NodesSources],
        "pSource": [m_dyn.pSource[s, :] for s in m_dyn.NodesSources],
        "pipe_rho": [m_dyn.pipe_rho[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR],
    }
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, init_output_path)
    print(f"[init] wrote solver initial values to '{init_output_path}'")


def run_model(horizon = 24, num_time_periods= 1, network_data_path = None, input_data_path = None, options_data_path = None, ocss_file_path = None,
              periodic_constraints = False, calculating_css = False, uncertainty = None,
              init_output_path=None, stop_after_init_output=False):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "data", "data_files", "Gaslib_40", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(_BASE_DIR, "data", "data_files", "Gaslib_40", "inputData_240hrs_gaslib40.xlsx")
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "data", "Options.json")
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "optimal_css_24hrs_gaslib40.xlsx")

    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)

    with open(options_data_path, 'r') as file:
        Options = json.load(file)
    
    Options['dynamic'] = False
    Options['T'] = Options['T0'] + Options['dt']/3600
    m_steady = buildNonLinearModel(
        networkData, inputData, Options)
    
    m_steady = init_network_default(m_steady, p_default=55e5)

    m_steady.pSource['source_1', :].fix()
    m_steady.pSource['source_2', :].fix()
    m_steady.wSource['source_3', :].fix()

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    _t0_steady = time.time()
    res_steady = ipopt.solve(m_steady, tee=True)
    _steady_cpu = time.time() - _t0_steady
    print(f"[timing] steady-state solve: {_steady_cpu:.2f}s")
    try:
        pyo.assert_optimal_termination(res_steady)
    except:
        debug_gas_model(m_steady)

    Options['dynamic'] = True
    Options['T'] = horizon

    m_dyn = buildNonLinearModel(
        networkData, inputData, Options)
    
    m_dyn = init_network_default(m_dyn, p_default=55e5)

    m_dyn.pSource["source_1", :].fix()
    m_dyn.pSource["source_2", :].fix()
    m_dyn.wSource["source_3", :].fix()

    t0 = m_dyn.Times.first()
    for p, vol in m_dyn.Pipes_VolExtrR_interm.data():
        m_dyn.interm_p[p, vol, t0] = m_steady.interm_p[p, vol, t0]
    m_dyn.interm_p[:, :, t0].fix()

    for c in m_dyn.Stations:
        m_dyn.compressor_P[c, t0] = m_steady.compressor_P[c, t0]
    m_dyn.compressor_P[:, t0].fix()
    
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    res_dyn = ipopt.solve(m_dyn, tee=True)
    try:
        pyo.assert_optimal_termination(res_dyn)
    except:
        debug_gas_model(m_dyn)

    demand_profiles = dynamic_demand_calculation(m_dyn, num_time_periods=num_time_periods)
    if uncertainty is not None:
        demand_profiles = uncertain_demand_calculation(m_dyn, demand_profiles, uncertainty)
        
    for s in m_dyn.wCons:
        if s[0].startswith("sink"):
            counter = 0
            for t in m_dyn.Times:
                m_dyn.wCons[s[0], 0, t].fix(demand_profiles[s[0]][counter])
                counter += 1

    m_dyn.compressor_beta['compressorStation_1', :].setub(4)

    if calculating_css:
        m_dyn.interm_p[:, :, t0].unfix()
        m_dyn.compressor_P[:, t0].unfix()
        def _terminal_controls(m, c):
            tf = m.Times.last()
            t0 = m.Times.first()
            return m.compressor_P[c, tf] == m.compressor_P[c, t0] 
        m_dyn.terminal_controls_constraint = pyo.Constraint(m_dyn.Stations, rule = _terminal_controls)
     
        def _terminal_pressure(m, p , vol):
            tf = m.Times.last()
            t0 = m.Times.first()
            return m.interm_p[p, vol, tf] ==m.interm_p[p, vol, t0]
        m_dyn.terminal_pressure = pyo.Constraint(m_dyn.Pipes_VolExtrR_interm, rule = _terminal_pressure)

    if periodic_constraints:
        m_dyn = css_terminal_constraints_each_point(m_dyn, num_time_periods= num_time_periods, horizon=horizon, ocss_file_path = ocss_file_path)
    print("\n=== Variable counts per component ===")
    rows_v = []
    for var in m_dyn.component_objects(pyo.Var, active=True, descend_into=False):
        n_free  = sum(1 for v in var.values() if not v.is_fixed())
        n_fixed = sum(1 for v in var.values() if v.is_fixed())
        rows_v.append((var.name.split('.')[-1], n_free, n_fixed))
    rows_v.sort(key=lambda x: -(x[1] + x[2]))
    tot_free = tot_fixed = 0
    for var_name, nf, nx in rows_v:
        print(f"  [gasnetwork] {var_name}: free={nf}, fixed={nx}, total={nf+nx}")
        tot_free += nf; tot_fixed += nx
    print(f"  --- TOTAL free={tot_free}, fixed={tot_fixed}, all={tot_free+tot_fixed} ---")
    print(f"DOF: {degrees_of_freedom(m_dyn)}")
    print("\n=== Constraint counts per component ===")
    rows_c = []
    for con in m_dyn.component_objects(pyo.Constraint, active=True, descend_into=False):
        n = sum(1 for _ in con)
        rows_c.append((con.name.split('.')[-1], n))
    rows_c.sort(key=lambda x: -x[1])
    total_c = 0
    for con_name, n in rows_c:
        print(f"  [gasnetwork] {con_name}: {n}")
        total_c += n
    print(f"  --- TOTAL active equality+inequality constraints: {total_c} ---\n")

    if init_output_path is not None:
        write_init_values_to_excel(m_dyn, init_output_path)
        if stop_after_init_output:
            timing = {
                'steady_cpu_s': _steady_cpu,
                'dynamic_cpu_s': None,
                'total_cpu_s':   _steady_cpu,
            }
            return m_steady, m_dyn, timing

    ipopt.options["tol"] = 1e-5
    ipopt.options["mu_init"] = 1e-6
    ipopt.options["bound_push"] = 1e-6
    _t0_dyn = time.time()
    res_dyn = ipopt.solve(m_dyn, tee=True)
    _dyn_cpu = time.time() - _t0_dyn
    print(f"[timing] dynamic solve: {_dyn_cpu:.2f}s")
    print(f"[timing] total (steady + dynamic): {_steady_cpu + _dyn_cpu:.2f}s")
    pyo.assert_optimal_termination(res_dyn)
    timing = {
        'steady_cpu_s': _steady_cpu,
        'dynamic_cpu_s': _dyn_cpu,
        'total_cpu_s':   _steady_cpu + _dyn_cpu,
    }
    return m_steady, m_dyn, timing


if __name__ == "__main__":
    init_output_path = os.path.join(_BASE_DIR, "init.xlsx")
    result_output_path = os.path.join(_BASE_DIR, "optimal_css_72hrs_gaslib40.xlsx")

    m_steady, m_dyn, timing = run_model(
        horizon=72,
        num_time_periods=3,
        periodic_constraints=True,
        calculating_css=False,
        init_output_path=init_output_path,
        stop_after_init_output=False,
    )

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
        "inlet_w": [m_dyn.inlet_w[a, :] for a in m_dyn.Arcs],
        "outlet_w": [m_dyn.outlet_w[a, :] for a in m_dyn.Arcs],
        "interm_w": [m_dyn.interm_w[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC_interm],
        "interm_p": [m_dyn.interm_p[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR_interm],
        "u2": [m_dyn.u2[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC],
        "u": [m_dyn.u[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrC],
        "wSource": [m_dyn.wSource[s, :] for s in m_dyn.NodesSources],
        "pSource": [m_dyn.pSource[s, :] for s in m_dyn.NodesSources],
        "pipe_rho": [m_dyn.pipe_rho[p, vol, :] for p, vol in m_dyn.Pipes_VolExtrR],
    }
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, result_output_path)

    obj_val = pyo.value(m_dyn.ObjFun)
    summary_df = pd.DataFrame([{
        'steady_cpu_s':  timing['steady_cpu_s'],
        'dynamic_cpu_s': timing['dynamic_cpu_s'],
        'total_cpu_s':   timing['total_cpu_s'],
        'objective':     obj_val,
    }])
    with pd.ExcelWriter(result_output_path, engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        summary_df.to_excel(writer, sheet_name='summary', index=False)
    print(f"[result] wrote solved trajectory to '{result_output_path}'")
