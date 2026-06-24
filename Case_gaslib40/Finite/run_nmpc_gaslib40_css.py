import os
import pyomo.contrib.mpc as mpc
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom
from model import buildNonLinearModel, init_network_default
from run_model_css_72 import run_model
from utils import dynamic_demand_calculation, import_data_from_excel
import json
from stability import apply_stability_constraint
from pyomo.common.timing import HierarchicalTimer
import matplotlib.pyplot as plt
import time

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_to_build_plant_model(network_data_path=None, input_data_path=None, options_data_path=None):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "data", "data_files", "Gaslib_40", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(_BASE_DIR, "data", "data_files", "Gaslib_40", "inputData.xlsx")
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "data", "Options.json")

    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)

    with open(options_data_path, 'r') as file:
        Options = json.load(file)

    Options['dynamic'] = True
    Options['T'] = 1

    return networkData, inputData, Options

def make_plant_and_controller_model(ocss_file_path, horizon=24, num_time_periods=1):
    m_steady, m_dyn, timing = run_model(horizon=horizon, num_time_periods=num_time_periods,
                                        ocss_file_path=ocss_file_path, periodic_constraints=True)
    m_controller = m_dyn

    networkData, inputData, Options = get_data_to_build_plant_model()
    m_plant= buildNonLinearModel(networkData, inputData, Options)

    m_plant = init_network_default(m_plant, p_default=55e5)

    m_plant.pSource['source_1', :].fix()
    m_plant.pSource['source_2', :].fix()
    m_plant.wSource['source_3', :].fix()

    t0 = m_plant.Times.first()
    for p, vol in m_plant.Pipes_VolExtrR_interm.data():
        m_plant.interm_p[p, vol, t0] = m_steady.interm_p[p, vol, t0]
    m_plant.interm_p[:, :, t0].fix()

    for c in m_plant.Stations:
        for t in m_plant.Times:
            m_plant.compressor_P[c, t] = m_dyn.compressor_P[c, t0]
    m_plant.compressor_P[:, :].fix()
    m_plant.ObjFun.deactivate()
    m_plant.compressor_beta['compressorStation_1', :].setub(4)

    assert degrees_of_freedom(m_plant) == 0

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    ipopt.solve(m_plant, tee=True)

    return m_controller, m_plant        

def load_demand_data(m, demand_data, start, stop, soft_constraint=False):
    for s in m.sink_node_set:
        actual_demand = demand_data[s][start:stop]
        for t_index, t in enumerate(m.Times, start=0):
            m.actual_demand[s, t] = actual_demand[t_index]
            if not soft_constraint:
                m.wCons[s, 0, t].fix(m.actual_demand[s, t])

def run_nmpc(
        simulation_steps = 24,
        sample_time = 1, 
        controller_horizon = 24,
        plant_horizon = 1,
        num_time_periods = 1,
        ocss_file_path = None):
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "optimal_css_24hrs_gaslib40_extended.xlsx")

    timer = HierarchicalTimer()
    timer.start('Initialization')
    m_controller, m_plant = make_plant_and_controller_model(ocss_file_path, horizon=controller_horizon, num_time_periods=num_time_periods)
    timer.stop('Initialization')

    apply_stability_constraint(m_controller)

    sink_node_set = [s for s in m_controller.Nodes if s.startswith('sink')]

    m_controller.sink_node_set = pyo.Set(initialize = sink_node_set)
    m_plant.sink_node_set = pyo.Set(initialize=sink_node_set)

    m_controller.actual_demand = pyo.Param(m_controller.sink_node_set, m_controller.Times, initialize=1, mutable=True)
    m_plant.actual_demand = pyo.Param(m_plant.sink_node_set, m_plant.Times, initialize=1, mutable=True)

    demand_data = dynamic_demand_calculation(m_controller, num_time_periods = num_time_periods, extended_profile=True)
    plt.plot(demand_data['sink_12'], label= 'controller demand')
    plt.plot(demand_data['sink_12'], label = 'plant demand')
    plt.title('Demand data')
    plt.xlabel('time (hrs)')
    plt.ylabel('Demand (kg/s)')
    plt.legend()

    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.Times)
    t0_controller = m_controller.Times.first()

    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.Times)

    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['linear_solver'] = 'ma57'
    tee = True

    plant_fixed_variables = [m_controller.compressor_P['compressorStation_1', :],
                         m_controller.compressor_P['compressorStation_2', :],
                         m_controller.compressor_P['compressorStation_3', :],
                         m_controller.compressor_P['compressorStation_4', :],
                         m_controller.compressor_P['compressorStation_5', :],
                         m_controller.compressor_P['compressorStation_6', :]]

    non_initial_plant_time = list(m_plant.Times)[1:]
    t0_controller = m_controller.Times.first()
    sim_t0 = 0.0

    sim_data = plant_interface.get_data_at_time([sim_t0])
    controller_lyapunov_function = {}
    iteration_times = {}

    for i in range(simulation_steps):
        print('--------------Running controller %d th time-----------------'%i)
        timer.start(f'iteration_{i}')

        sim_t0 = i * sample_time

        start = sim_t0
        stop = start + controller_horizon + 1
        load_demand_data(m_controller, demand_data, start, stop)
        stop = start + plant_horizon + 1  
        load_demand_data(m_plant, demand_data, start, stop)

        if sim_t0 == 0.0:
            m_controller.stability_constraint.deactivate()
        else:
            m_controller.stability_constraint.activate()

        print(degrees_of_freedom(m_controller))
        timer.start('Solve_controller_model')
        res = solver.solve(m_controller, tee=tee)
        timer.stop('Solve_controller_model')
        try:
            pyo.assert_optimal_termination(res)
        except:
            import pdb;pdb.set_trace()

        ts_data = controller_interface.get_data_at_time(sample_time)
        input_data = ts_data.extract_variables(plant_fixed_variables)

        plant_interface.load_data(input_data, time_points=non_initial_plant_time)

        print(degrees_of_freedom(m_plant))
        assert degrees_of_freedom(m_plant) == 0
        timer.start('Solve_plant_model')
        res = solver.solve(m_plant, tee=tee)
        timer.stop('Solve_plant_model')
        pyo.assert_optimal_termination(res)

        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0-m_plant.Times.first())
        sim_data.concatenate(m_data)

        tf_data = plant_interface.get_data_at_time(m_plant.Times.last())
        plant_interface.load_data(tf_data)

        m_controller.lyapunov_function_prev = pyo.value(m_controller.lyapunov_function_current)
        m_controller.tracking_cost_plant_prev = pyo.value(sum((m_plant.compressor_P[s, 1.0] - m_controller.compressor_P_ocss[s, 1.0])**2 
                                                              for s in m_controller.Stations) + 
                                                          sum((m_plant.interm_p[p, vol, 1.0] - m_controller.interm_p_ocss[p, vol, 1.0])**2 
                                                              for p, vol  in m_controller.Pipes_VolExtrR_interm) 
                                                          + (m_plant.pSource['source_3', 1.0] - m_controller.pSource_ocss['source_3', 1.0])**2)
        controller_lyapunov_function[sim_t0] = pyo.value(m_controller.lyapunov_function_current)

        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)

        N = num_time_periods
        K = int(controller_horizon/num_time_periods)
        [m_controller.compressor_P_ocss[s, N*K].fix(m_controller.compressor_P_ocss[s, (N-1)*K]) for s in m_controller.Stations]
        [m_controller.interm_p_ocss[p, vol, N*K].fix(m_controller.interm_p_ocss[p, vol, (N-1)*K]) for p, vol in m_controller.Pipes_VolExtrR_interm]
        [m_controller.pSource_ocss['source_3', N*K].fix(m_controller.pSource_ocss['source_3', (N-1)*K])]

        timer.stop(f'iteration_{i}')
        iteration_times[i] = timer.get_total_time(f'iteration_{i}')
        print(f"Iteration {i} time: {iteration_times[i]:.2f}s", flush=True)

        print("Lyapunov function:")
        print(controller_lyapunov_function)
    print(timer)
    print("Iteration times:", iteration_times)
    with open(os.path.join(_BASE_DIR, 'hierarchical_timer_std_nmpc_gaslib40.txt'), 'w') as f:
        f.write(str(timer))
    return m_plant, m_controller, sim_data, controller_lyapunov_function, iteration_times

if __name__ =="__main__":
    _total_start = time.time()
    m_plant, m_controller, sim_data, controller_lyapunov_function, iteration_times = run_nmpc(simulation_steps = 144,
                                               sample_time = 1,
                                               controller_horizon = 240,
                                               plant_horizon = 1,
                                               num_time_periods=10)
    _total_runtime = time.time() - _total_start
    print(f"Total runtime: {_total_runtime:.2f}s", flush=True)
    
    from pyomo.contrib.mpc.examples.cstr.model import _plot_time_indexed_variables
    _plot_time_indexed_variables(sim_data, [m_plant.compressor_P["compressorStation_1",:], 
                                            m_plant.compressor_P["compressorStation_2",:], 
                                            m_plant.compressor_P["compressorStation_3",:], 
                                            m_plant.compressor_P["compressorStation_4",:], 
                                            m_plant.compressor_P["compressorStation_5",:], 
                                            m_plant.compressor_P["compressorStation_6",:]], show=True)
    
    all_nodes_p = []
    for n in m_plant.Nodes:
        if str(n).startswith('sink'):
            all_nodes_p.append(m_plant.node_p[n, :])
    _plot_time_indexed_variables(sim_data, all_nodes_p, show = True)
    _plot_time_indexed_variables(sim_data, [m_plant.interm_p['pipe_8', 2, :], m_plant.interm_p['pipe_8', 3, :], m_plant.interm_p['pipe_8', 4, :], m_plant.node_p['sink_9', :], m_plant.node_p['sink_18', :]], show = True)
    
    from utils import write_data_to_excel
    sheets_keys_dict = {"compressor power": [m_plant.compressor_P[s, :] for s in m_plant.Stations],
                        "compressor beta": [m_plant.compressor_beta[s, :] for s in m_plant.Stations],
                        "wCons": [m_plant.wCons[s, 0, :] for s in m_plant.Nodes if str(s).startswith('sink')],
                        "node pressure": all_nodes_p,
                        "interm_w": [m_plant.interm_w[p, vol, :] for p, vol in m_plant.Pipes_VolExtrC_interm],
                        "interm_p": [m_plant.interm_p[p, vol, :] for p, vol in m_plant.Pipes_VolExtrR_interm],
                        "wSource": [m_plant.wSource[s, :] for s in m_plant.NodesSources],
                        "pSource": [m_plant.pSource[s, :] for s in m_plant.NodesSources]
                        }
    write_data_to_excel(sim_data, m_plant, sheets_keys_dict, "final_paper_enmpc_explicit_terminal_each_point_stability_10_cycles_72.xlsx",
                        controller_1_lyapunov=controller_lyapunov_function,
                        iteration_times=iteration_times,
                        total_runtime=_total_runtime)
