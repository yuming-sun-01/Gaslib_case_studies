# perturbation = 1.02
from utils import import_data_from_excel
from utils import debug_gas_model
from utils import dynamic_demand_calculation, uncertain_demand_calculation
from utils import plot_compressor_beta, plot_compressor_power
from utils import write_data_to_excel
from model import buildNonLinearModel
from model import init_network_default
from terminal import css_terminal_constraints, css_terminal_constraints_each_point
import json
import math
import os
import pyomo.environ as pyo
import pyomo.contrib.mpc as mpc
from idaes.core.util.model_statistics import report_statistics
import numpy as np
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_model(horizon = 24, num_time_periods= 1, network_data_path = None, input_data_path = None, options_data_path = None, ocss_file_path = None,
              periodic_constraints = False, calculating_css = False, uncertainty = None):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "test", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(_BASE_DIR, "test", "inputData_10periods.xlsx")
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "test", "Options.json")
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "optimal_css_24hrs_test.xlsx")
    # network and input data
    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)

    #Load options file
    with open(options_data_path, 'r') as file:
        Options = json.load(file)
    
    ''' Steady state model '''
    Options['dynamic'] = False # no time derivative, only one time point: T0+dt
    Options['T'] = Options['T0'] + Options['dt']/3600
    m_steady = buildNonLinearModel(
        networkData, inputData, Options)
    
    # Initialization to default
    m_steady = init_network_default(m_steady, p_default=55e5)

    m_steady.pSource['source_1', :].fix(40)
    


    # Fix source variable
    m_steady.compressor_beta['compressorStation_1', 0].fix(1.05)
    m_steady.compressor_beta['compressorStation_2', 0].fix(1.05)
    m_steady.compressor_beta['compressorStation_3', 0].fix(1.05)


    # Solve
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    res_steady = ipopt.solve(m_steady, tee=True)
    try:
        pyo.assert_optimal_termination(res_steady)
    except:
        debug_gas_model(m_steady)
    print("\n=== Steady pSource ===")
    for s in m_steady.NodesSources:
        for t in m_steady.Times:
            print(f"m_steady.pSource[{s}, {t}] = {pyo.value(m_steady.pSource[s, t])}")

    print("\n=== Steady wSource ===")
    for s in m_steady.NodesSources:
        for t in m_steady.Times:
            print(f"m_steady.wSource[{s}, {t}] = {pyo.value(m_steady.wSource[s, t])}")

    ''' Dynamic model '''
    Options['dynamic'] = True
    Options['T'] = horizon

    m_dyn = buildNonLinearModel(
        networkData, inputData, Options)
    
    # Initialize to default
    m_dyn = init_network_default(m_dyn, p_default=55e5)

    # Fix pressure source
    m_dyn.pSource["source_1", :].fix()


    # Fix initial state --> to steady state ##########perturbation##############
    t0 = m_dyn.Times.first()
    perturbation = 1.0
    for p, vol in m_dyn.Pipes_VolExtrR_interm.data():
        m_dyn.interm_p[p, vol, t0] = m_steady.interm_p[p, vol, t0] * perturbation
    m_dyn.interm_p[:, :, t0].fix()

    for c in m_dyn.Stations:
        m_dyn.compressor_P[c, t0] = m_steady.compressor_P[c, t0] * perturbation
    m_dyn.compressor_P[:, t0].fix()
    
    # Solve
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    res_dyn = ipopt.solve(m_dyn, tee=True)
    try:
        pyo.assert_optimal_termination(res_dyn)
    except:
        debug_gas_model(m_dyn)

    # Load dynamic demand profile
    demand_profiles = dynamic_demand_calculation(m_dyn, num_time_periods=num_time_periods)
    if uncertainty is not None:
        demand_profiles = uncertain_demand_calculation(m_dyn, demand_profiles, uncertainty)
        
    for s in m_dyn.wCons:
        if s[0].startswith("sink"): # only deal with sink
            counter = 0
            for t in m_dyn.Times:
                m_dyn.wCons[s[0], 0, t].fix(demand_profiles[s[0]][counter])
                counter += 1

    # Increase UB on compressor beta
    m_dyn.compressor_beta['compressorStation_1', :].setub(4)

    # Add periodic terminal constraint

    #If we are not calculating css then the initial state of the plant should not be unfixed
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

    # Let IDAES report the model statistics (variables, constraints, DOF)
    # straight from the model instead of counting components by hand.
    report_statistics(m_dyn)
    ipopt.options["tol"] = 1e-5
    ipopt.options["mu_init"] = 1e-6
    ipopt.options["bound_push"] = 1e-6
    res_dyn = ipopt.solve(m_dyn, tee=True)
    pyo.assert_optimal_termination(res_dyn)
    return m_steady, m_dyn


if __name__ == "__main__":
    # ---- Run CSS calculation ------------------------------------------ #
    ocss_output_path = os.path.join(_BASE_DIR, "test_enmpc_6hrs.xlsx")

    m_steady, m_dyn = run_model(
        horizon=60,
        num_time_periods=10,
        periodic_constraints=True,
        calculating_css=False,
    )

    # ---- Write output -------------------------------------------------- #
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
    write_data_to_excel(sim_data, m_dyn, sheets_keys_dict, ocss_output_path)

    # ---- Plot ---------------------------------------------------------- #
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

'''
#Check if states are the same
import math 
for p, v in m_dyn.Pipes_VolExtrC_interm:
    try:
        assert math.isclose(pyo.value(m_dyn.interm_w[p, v, 0]), pyo.value(m_dyn.interm_w[p, v, cycle_length]))
    except:
        print(pyo.value(m_dyn.interm_w[p, v, 0]), pyo.value(m_dyn.interm_w[p, v, cycle_length]))
for s in m_dyn.Nodes:
    assert math.isclose(pyo.value(m_dyn.node_p[s, 0]), pyo.value(m_dyn.node_p[s, cycle_length]))
    
for s in m_dyn.NodesSources:
    assert math.isclose(pyo.value(m_dyn.wSource[s, 0]), pyo.value(m_dyn.wSource[s, cycle_length]))
'''
