# perturbation = 1.0
import os
import pyomo.contrib.mpc as mpc
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom
from model import buildNonLinearModel, init_network_default
from run_model_css_72 import run_model
from utils import dynamic_demand_calculation, import_data_from_excel, debug_gas_model, plot_compressor_beta, plot_compressor_power
import json
from stability import apply_stability_constraint
from pyomo.common.timing import HierarchicalTimer
import matplotlib.pyplot as plt
import time
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data_to_build_plant_model(network_data_path=None, input_data_path=None, options_data_path=None):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "test", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(_BASE_DIR, "test", "inputData_10periods.xlsx")
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "test", "Options.json")

    # Load network and input data
    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)

    # load options file
    with open(options_data_path, 'r') as file:
        Options = json.load(file)

    Options['dynamic'] = True
    Options['T'] = 1

    return networkData, inputData, Options

def make_plant_and_controller_model(ocss_file_path, horizon=24, num_time_periods=1): # this value will be covered
    # Get initialized dynamic model to build controller
    # The dynamic model is initialized using a sinusoidal demand profile
    # centered around the steady state demand with a horizon of 24 hours
    m_steady, m_dyn = run_model(horizon=horizon, num_time_periods=num_time_periods,
                                ocss_file_path=ocss_file_path, periodic_constraints=True)
    m_controller = m_dyn

    # Make plant model
    networkData, inputData, Options = get_data_to_build_plant_model()
    m_plant= buildNonLinearModel(networkData, inputData, Options)

    # Initialize the plant model (mirrors run_model_css_72)
    m_plant = init_network_default(m_plant, p_default=55e5)

    t0 = m_plant.Times.first()

    #fix pressure and flow source in the plant
    m_plant.pSource['source_1', :].fix()

    # IC from steady state, no per-pipe factor (mirrors run_model_css_72)
    for p, vol in m_plant.Pipes_VolExtrR_interm.data():
        m_plant.interm_p[p, vol, t0] = pyo.value(m_steady.interm_p[p, vol, t0])
    m_plant.interm_p[:, :, t0].fix()

    for c in m_plant.Stations:
        m_plant.compressor_P[c, t0] = pyo.value(m_steady.compressor_P[c, t0])
        for t in m_plant.Times:
            m_plant.compressor_P[c, t] = pyo.value(m_steady.compressor_P[c, t0])
    m_plant.compressor_P[:, :].fix()
    m_plant.ObjFun.deactivate()
    m_plant.compressor_beta['compressorStation_1', :].setub(4)

    assert degrees_of_freedom(m_plant) == 0

    # Initialize plant model by simulation
    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = 'ma57'
    ipopt.solve(m_plant, tee=False)

    return m_controller, m_plant        

def load_demand_data(m, demand_data, start, stop, soft_constraint=False):
    #This function just updates the mutable parameter actual demand 
    #and if we are not writing soft constraints on demand then fixes the demand to the actual demand value
    for s in m.sink_node_set:
        for t_index, t in enumerate(m.Times, start=0):
            m.actual_demand[s, t] = demand_data[s][(start + t_index) % 6]
            if not soft_constraint:
                m.wCons[s, 0, t].fix(m.actual_demand[s, t])

# Run nmpc
def run_nmpc(
        simulation_steps = 24,
        sample_time = 1, 
        controller_horizon = 6,
        plant_horizon = 1,
        num_time_periods = 1,
        ocss_file_path = None):
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "optimal_css_24hrs_test_extended.xlsx")

    timer = HierarchicalTimer()
    timer.start('Initialization')
    # Get initialized controller and plant models
    m_controller, m_plant = make_plant_and_controller_model(ocss_file_path, horizon=controller_horizon, num_time_periods=num_time_periods)
    timer.stop('Initialization')

    apply_stability_constraint(m_controller)

    # Create a set for sink nodes to easily load demand profiles
    sink_node_set = [s for s in m_controller.Nodes if s.startswith('sink')]

    m_controller.sink_node_set = pyo.Set(initialize = sink_node_set)
    m_plant.sink_node_set = pyo.Set(initialize=sink_node_set)

    # If we need to write demand as soft constraint introduce slack variables
    # It makes sense to make actual demand a mutable parameter if we are going to write 
    # soft constraints since evry time in the controller we can just update the mutable parameter 
    m_controller.actual_demand = pyo.Param(m_controller.sink_node_set, m_controller.Times, initialize=1, mutable=True)
    m_plant.actual_demand = pyo.Param(m_plant.sink_node_set, m_plant.Times, initialize=1, mutable=True)

    # Get extended demand data
    # Here num time period is N which is how many times the demand profile repeats 
    demand_data = dynamic_demand_calculation(m_controller, num_time_periods = num_time_periods, extended_profile=True)
    plt.plot(demand_data['sink_1'], label= 'controller demand')
    plt.plot(demand_data['sink_1'], label = 'plant demand')
    plt.title('Demand data')
    plt.xlabel('time (hrs)')
    plt.ylabel('Demand (kg/s)')
    plt.legend()

    # Create dynamic model interface for controller
    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.Times)
    t0_controller = m_controller.Times.first()

    # Create dynamic model interface for plant
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.Times)

    # Define solver
    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-4
    solver.options['linear_solver'] = 'ma57'
    tee = False

    # variables that will be fixed in the plant simulation
    plant_fixed_variables = [m_controller.compressor_P['compressorStation_1', :],
                         m_controller.compressor_P['compressorStation_2', :],
                         m_controller.compressor_P['compressorStation_3', :]]

    non_initial_plant_time = list(m_plant.Times)[1:] # t = 0 is fixed
    t0_controller = m_controller.Times.first()
    sim_t0 = 0.0

    # Initialize data structure to hold results of "rolling horizon" simulation
    sim_data = plant_interface.get_data_at_time([sim_t0]) # global time
    controller_lyapunov_function = {}
    iteration_times = {}

    # m_controller = tracking_objective(m_controller)
    for i in range(simulation_steps):
        print('--------------Running controller %d th time-----------------'%i)
        timer.start(f'iteration_{i}')

        sim_t0 = i * sample_time

        # Load demand data into the controller and plant
        start = sim_t0
        stop = start + controller_horizon + 1
        load_demand_data(m_controller, demand_data, start, stop)
        stop = start + plant_horizon + 1  
        load_demand_data(m_plant, demand_data, start, stop)

        # Remove stability constraint if it is t = 0
        if sim_t0 == 0.0:
            m_controller.stability_constraint.deactivate()
        else:
            m_controller.stability_constraint.activate()

        # Solve controller model to get inputs
        print(degrees_of_freedom(m_controller))
        # assert degrees_of_of_freedom(m_controller) == 47 * 6
        timer.start('Solve_controller_model')
        res = solver.solve(m_controller, tee=False)
        timer.stop('Solve_controller_model')
        try:
            pyo.assert_optimal_termination(res)
        except:
            import pdb;pdb.set_trace()

        ts_data = controller_interface.get_data_at_time(sample_time)
        input_data = ts_data.extract_variables(plant_fixed_variables)

        plant_interface.load_data(input_data, time_points=non_initial_plant_time)

        # Solve plant model to simulate
        print(degrees_of_freedom(m_plant))
        assert degrees_of_freedom(m_plant) == 0
        timer.start('Solve_plant_model')
        res = solver.solve(m_plant, tee=False)
        timer.stop('Solve_plant_model')
        pyo.assert_optimal_termination(res)

        # Extract data from simulated model
        m_data = plant_interface.get_data_at_time(non_initial_plant_time)
        m_data.shift_time_points(sim_t0-m_plant.Times.first())
        sim_data.concatenate(m_data)

        # Reinitialize plant model
        tf_data = plant_interface.get_data_at_time(m_plant.Times.last())
        plant_interface.load_data(tf_data)

        # Update the stability constraint parameter
        m_controller.lyapunov_function_prev = pyo.value(m_controller.lyapunov_function_current)
        m_controller.tracking_cost_plant_prev = pyo.value(sum((m_plant.compressor_P[s, 1.0] - m_controller.compressor_P_ocss[s, 1.0])**2
                                                              for s in m_controller.Stations) +
                                                          sum((m_plant.interm_p[p, vol, 1.0] - m_controller.interm_p_ocss[p, vol, 1.0])**2
                                                              for p, vol  in m_controller.Pipes_VolExtrR_interm) )
        controller_lyapunov_function[sim_t0] = pyo.value(m_controller.lyapunov_function_current)

        '''
        # Plot interm_p and controls or debugging
        if sim_t0 % 10 == 0:
            plt.figure()
            for s in m_controller.Stations:
                plt.plot(pyo.value(m_controller.compressor_beta[s, :]))
                plt.plot(pyo.value(m_controller.compressor_beta_ocss[s, :]), ':k')
                plt.title("At time k = %i" %int(sim_t0))
                
            plt.figure()
            for s in m_controller.NodesSources:
                plt.plot(pyo.value(m_controller.wSource[s, :]))
                plt.plot(pyo.value(m_controller.wSource_ocss[s, :]), ':k')
                plt.title("At time k = %i" %int(sim_t0))
        '''

        # Reinitialize controller model
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(tf_data, time_points=t0_controller)

        # Update ocss at the last point to equal to the first point
        N = num_time_periods
        K = int(controller_horizon/num_time_periods)
        [m_controller.compressor_P_ocss[s, N*K].fix(m_controller.compressor_P_ocss[s, (N-1)*K]) for s in m_controller.Stations]
        [m_controller.interm_p_ocss[p, vol, N*K].fix(m_controller.interm_p_ocss[p, vol, (N-1)*K]) for p, vol in m_controller.Pipes_VolExtrR_interm]

        timer.stop(f'iteration_{i}')
        iteration_times[i] = timer.get_total_time(f'iteration_{i}')
        print(f"Iteration {i} time: {iteration_times[i]:.2f}s", flush=True)

        print("Lyapunov function:")
        print(controller_lyapunov_function)
    print(timer)
    print("Iteration times:", iteration_times)
    with open(os.path.join(_BASE_DIR, 'hierarchical_timer_std_nmpc_test.txt'), 'w') as f:
        f.write(str(timer))
    return m_plant, m_controller, sim_data, controller_lyapunov_function, iteration_times

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


if __name__ =="__main__":
    _total_start = time.time()
    m_plant, m_controller, sim_data, controller_lyapunov_function, iteration_times = run_nmpc(simulation_steps = 24,
                                               sample_time = 1,
                                               controller_horizon = 6,
                                               plant_horizon = 1,
                                               num_time_periods= 1)
    _total_runtime = time.time() - _total_start
    print(f"Total runtime: {_total_runtime:.2f}s", flush=True)
    
    #Plot compressor power in the plant (Note: it is scaled by 1e5)
    from pyomo.contrib.mpc.examples.cstr.model import _plot_time_indexed_variables
    _plot_time_indexed_variables(sim_data, [m_plant.compressor_P["compressorStation_1",:], 
                                            m_plant.compressor_P["compressorStation_2",:], 
                                            m_plant.compressor_P["compressorStation_3",:]], show=True)
    
    #All pressure variables at demand nodes
    all_nodes_p = []
    for n in m_plant.Nodes:
        if str(n).startswith('sink'):
            all_nodes_p.append(m_plant.node_p[n, :])
    _plot_time_indexed_variables(sim_data, all_nodes_p, show = True)
    # _plot_time_indexed_variables(sim_data, [m_plant.interm_p['pipe_8', 2, :], m_plant.interm_p['pipe_8', 3, :], m_plant.interm_p['pipe_8', 4, :], m_plant.node_p['sink_9', :], m_plant.node_p['sink_18', :]], show = True)
    
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
    # _output_path = os.path.join(_BASE_DIR, "test_enmpc_explicit_terminal_each_point_stability_72hrs_10_cycles.xlsx")
    _output_path = os.path.join(_BASE_DIR, "f_enmpc_cycle_1.xlsx")
    write_data_to_excel(sim_data, m_plant, sheets_keys_dict, _output_path,
                        controller_1_lyapunov=controller_lyapunov_function)
    _append_runtime_sheet(_output_path, iteration_times, _total_runtime)


'''
def write_soft_constraints(m, terminal_constraints=False):
    m.slack = pyo.Var(m.sink_node_set, m.Times, domain=pyo.Reals)

    def _soft_constraint_on_demands(m, s, t):
        m.wCons[s, 0, t].unfix()
        return m.wCons[s, 0, t] == m.actual_demand[s, t] + m.slack[s, t]
    m.demand_constraint_soft = pyo.Constraint(m.sink_node_set, m.Times, rule=_soft_constraint_on_demands)
    
    m.ObjFun.deactivate()

    #This differentiation is necessary because plant model doesn't have terminal constraints
    if terminal_constraints:
        m.obj = pyo.Objective(expr = m.ObjFun
                              + 1e5 * sum(m.slack[s, t]**2 for s in m.sin_node_set for t in m.Times)
                              + 1e5 * sum(m.terminal_flow_slacks[p, vol]**2 for p, vol in m.Pipes_VolExtrC_interm)
                              + 1e5 * sum(m.terminal_pressure_slacks[p, vol]**2 for p, vol in m.Pipes_VolExtrR_interm)
                              )
    else:
        m.obj = pyo.Objective(expr = m.ObjFun
                              + 1e5 * sum(m.terminal_pressure_slacks[p, vol]**2 for p, vol in m.Pipes_VolExtrR_interm))

def tracking_objective(m):
    m.objFunc.deactivate()
    m.obj = pyo.Objective(expr = sum((m.interm_p[p, vol, t]-m.interm_p_ocss[p, vol, t])**2 
                                         for p, vol in m.Pipes_VolExtrR_interm for t in m.Times if t != m.Times.last())
                                        + sum((m.compressor_P[s, t] - m.compressor_P_ocss[s, t])**2
                                            for s in m.Stations for t in m.Times if t != m.Times.last())
                                        + sum((m.pSource['source_3', t] - m.pSource_ocss['source_3', t])**2
                                            for t in m.Times if t != m.Times.last())
                                        + 0 * m.ObjFun
                                            )
    return m
'''
