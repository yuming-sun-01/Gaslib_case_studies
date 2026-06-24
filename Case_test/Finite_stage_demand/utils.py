from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import (DiagnosticsToolbox)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
import pyomo.environ as pyo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
import os

#==============================================================================
""" debug_model """
#==============================================================================
'''
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.model_diagnostics import (DiagnosticsToolbox)
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
import pyomo.environ as pyo
'''
def debug_gas_model(model):
    
    #Fix degrees of freedom to make the problem square
    for s in model.Stations:
        for t in model.Times:
            model.compressor_P[s, t].fix()
    print("degrees_of_freedom = ", degrees_of_freedom(model))
    #assert degrees_of_freedom(model) == 0
    dt = DiagnosticsToolbox(model)
    dt.report_structural_issues()

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.solve(model, tee=True)
    
    dt = DiagnosticsToolbox(model)
    dt.report_numerical_issues()
    dt.display_variables_at_or_outside_bounds()
    dt.display_underconstrained_set()
    dt.display_overconstrained_set()
    dt.display_constraints_with_large_residuals()
   
def analyze_violations(model):
    # Analyze constraint violations
    violated_constraints = []
    for constr in model.component_data_objects(pyo.Constraint, active=True):
        constr_value = pyo.value(constr)
        if constr_value > 10:  # Constraint is violated
            violated_constraints.append((constr.name, constr_value))
    
    # Sort the violated constraints by the degree of violation
    violated_constraints.sort(key=lambda x: x[1], reverse=True)
    
    # Print the most violated constraints
    print("Most violated constraints:")
    for name, value in violated_constraints[:]:  # Display top 5 violated constraints
        print(f"{name}: {value}")
        
    # Analyze variable bound violations
    violated_variable_bounds = []
    for var in model.component_data_objects(pyo.Var):
        if var.has_lb():
            if pyo.value(var) < var.lb:
                violation_amount = var.lb - pyo.value(var)
                violated_variable_bounds.append((var.name, "Lower bound", violation_amount))
        if var.has_ub():
            if pyo.value(var) > var.ub:
                violation_amount = pyo.value(var) - var.ub
                violated_variable_bounds.append((var.name, "Upper bound", violation_amount))
    
    # Sort the violated variable bounds by the degree of violation
    violated_variable_bounds.sort(key=lambda x: x[2], reverse=True)
    
    # Print the most violated variable bounds
    print("Most violated variable bounds:")
    for name, bound_type, violation_amount in violated_variable_bounds[:5]:  # Display top 5 violated variable bounds
        print(f"{name}: {bound_type} violated by {violation_amount}")

#==============================================================================
""" import data """
#==============================================================================
'''
import pandas as pd
'''
def DataFrame_2levels_to_dict(df):
    """
    Rearrange pipe dataframe with indices: (pipe, vol) and columns: time
    into a dictionary of type {pipe: vol: {t0: .., t1: }}
    """

    dct_interm = pd.DataFrame.to_dict(df, orient = "index")
    dct = {keys[0]: {} for keys in dct_interm.keys()}
    for keys in dct_interm.keys():
        dct[keys[0]][keys[1]] = dct_interm[keys] 
    return dct

#==============================================================================
""" NETWORK DATA """

def import_network_data_from_excel(data_path):
    # ARCS
    df_arcs = pd.read_excel(data_path, sheet_name = "Arcs", index_col = 0)
    Arcs = pd.DataFrame.to_dict(df_arcs, orient = "index") 
    # PIPES
    df_pipes = pd.read_excel(data_path, sheet_name = "Pipes", index_col = 0)
    Pipes = pd.DataFrame.to_dict(df_pipes, orient = "index")      
    # NODES
    df_nodes = pd.read_excel(data_path, sheet_name = "Nodes", index_col = 0)
    Nodes = pd.DataFrame.to_dict(df_nodes, orient = "index")
    
    # STATIONS
    df_stations = pd.read_excel( data_path, sheet_name = "Stations", index_col = 0)
    Stations = pd.DataFrame.to_dict(df_stations, orient = "index")    

    # VALVES
    Valves = {}
    try:
        df_v = pd.read_excel(
            data_path, sheet_name = "Valves", index_col = 0)
        Valves = pd.DataFrame.to_dict(df_v, orient = "index")   
    except:
        print('!!! Valves sheet missing in networkData')
           
    # DATA
    Data = {"Arcs": Arcs, "Pipes": Pipes, "Nodes": Nodes, 
            "Valves" : Valves, 
            "Stations": Stations}   
    return Data

#==============================================================================
"""  TIME VARYING DATA """

## ADJUST DATA
def rearrange_setpoint_data(df):
    """ Rearrange setpoints dataframe with indices: (source, setpoint_type) and columns: time
    into 2 dictionaries (p setpoint and w setpoint) of type {source: {t0: .., t1: }}"""
    pCV = {}
    wCV = {}
    for elem, var in df.index:
        if var == "p":
            pCV[elem] = {t: df.loc[elem,"p"][t] for t in df.columns} 
        elif var == "w":
            wCV[elem] = {t: df.loc[elem,"w"][t] for t in df.columns} 
    return pCV, wCV

def set_pipe_cons_to_default(wcons, Pipes, value = 0):
    """ 
    Set wcons = 0 in all Pipes volumes that are not specified in wcons dictionary 
    """
    if len(list(Pipes.keys())) >= 1:
        ref_pipe = list(Pipes.keys())[0]
        times = wcons[ref_pipe][list(wcons[ref_pipe].keys())[0]].keys()
              
        for pipe in Pipes.keys(): # add wcons = 0 in other pipes volumes
            if pipe not in wcons.keys():
                wcons[pipe] = {}
            for vol in range(1, int(Pipes[pipe]["Nvol"])):
                if vol not in wcons[pipe].keys():
                    wcons[pipe][vol] = {}
                    for counter, t in enumerate(times):
                        wcons[pipe][vol][t] = value
    return wcons

## IMPORT DATA
def import_time_varying_data_from_excel(data_path):
    # SOURCES
    df_sources = pd.read_excel(data_path, sheet_name = "SourcesSP",index_col = [0,1], header = 0)
    pSource, wSource = rearrange_setpoint_data(df_sources)                 
    # WCONS (pipe/nodes)
    df_wcons = pd.read_excel(data_path, sheet_name = "wcons",index_col = [0,1], header = 0)
    wcons = DataFrame_2levels_to_dict(df_wcons)   
    #Input gas parameters
    df_params = pd.read_excel(data_path, sheet_name = "GasParams", header = 0)
    params = pd.DataFrame.to_dict(df_params)   
    Data = {
        "pSource": pSource,"wSource": wSource, "wCons":wcons, "GasParams":params}
    return Data


#==============================================================================
""" ALL """

def import_data_from_excel(network_data_path, input_data_path):
    # topology
    networkData = import_network_data_from_excel(network_data_path)
    # profiles
    inputData = import_time_varying_data_from_excel(input_data_path)
    # add missing consumption in pipes finite volumes
    inputData['wCons'] = set_pipe_cons_to_default(inputData['wCons'], networkData['Pipes']) 
    # try:
    #     inputData['wCons'] = set_pipe_cons_to_default(inputData['wCons'], networkData['Pipes'])
    # except:
    #     print('!!! ERROR importing inputData. No matching with networkData')
    return networkData, inputData

#==============================================================================
""" make_demand_dynamic """
#==============================================================================
'''
import numpy as np
import matplotlib.pyplot as plt
import pyomo.environ as pyo
'''
def dynamic_demand_profile(ss_demand, num_time_periods = 1, epsilon = 0):
    amplitude = 0.175
    time_length = len(ss_demand)
    ss_demand = np.asarray(ss_demand)

    phase = np.mod(np.arange(time_length), 6)
    profile_time = np.array([0, 1, 2, 3, 4, 5, 6])
    profile_shape = np.array([0, 1, 1, 0, -1, -1, 0])

    cyclic_shape = np.interp(phase, profile_time, profile_shape)
    dynamic_demand = ss_demand + (ss_demand + epsilon)*cyclic_shape*amplitude

    return dynamic_demand

def dynamic_demand_profile_extended(dynamic_demand):
    dynamic_demand1 = dynamic_demand
    dynamic_demand2 = dynamic_demand[1:]
    
    dynamic_demand_extended = np.concatenate((dynamic_demand1, dynamic_demand2))
    return dynamic_demand_extended
    
    
def dynamic_demand_calculation(m, num_time_periods = 1, time_length = None, extended_profile = False, epsilon = 0):
    if time_length is None:
        time_length = len(m.Times)
        
    dynamic_demand = {}
    for s in m.wCons:
        if s[0].startswith('sink'):
            ss_demand = [pyo.value(m.wCons[s])]*time_length
            dynamic_demand[s[0]] = dynamic_demand_profile(ss_demand,  num_time_periods, epsilon = epsilon)
            if extended_profile:
                dynamic_demand[s[0]] = dynamic_demand_profile_extended(dynamic_demand[s[0]])
    return dynamic_demand

def uncertain_demand_calculation(m, dynamic_demand, uncertainty={(0,13): -0.1, (13, 25): -0.1}):
    uncertain_demand = {}
    for s in m.wCons:
        if s[0].startswith('sink'):
            uncertain_demand_list = []
            for keys in uncertainty.keys():
                dyn_demand = dynamic_demand[s[0]][keys[0]:keys[1]]
                ss_demand = dynamic_demand[s[0]][0]
                uncertain_demand_profile =  dyn_demand + (dyn_demand - ss_demand)*uncertainty[keys]
                uncertain_demand_list.append(uncertain_demand_profile)
            uncertain_demand[s[0]] = np.concatenate(uncertain_demand_list)
    return uncertain_demand
                
'''         
if __name__ == "__main__":
    ss_demand = [16.35417]*72
    dynamic_demand = dynamic_demand_profile(ss_demand,num_time_periods = 3, epsilon = 0)
    print(dynamic_demand)
    plt.plot(dynamic_demand)
    # dynamic_demand = dynamic_demand_profile(ss_demand, num_time_periods=3, epsilon = 1)
    # plt.plot(dynamic_demand)
    # plt.plot(ss_demand)
    
    #Extended demand profile
    plt.figure()
    dynamic_demand_extended = dynamic_demand_profile_extended(dynamic_demand)
    plt.plot(dynamic_demand_extended)
    plt.xlabel('Time(hrs)')
    plt.ylabel('Gas demand at each node (kg/s)')
    #plt.plot(ss_demand)
'''
#==============================================================================
""" network_graph """
#==============================================================================
'''
import networkx as nx
import matplotlib.pyplot as pl
from matplotlib.lines import Line2D
'''
#==============================================================================
""" CREATE GRAPH """

TRANSPARENT = True
FONTSIZE = 10
AXESOFF = True
LINEWIDTH = 4.0

plt.rcParams["font.size"] = FONTSIZE
plt.rcParams["text.usetex"] = True
def set_elem_attr(G, elem, attr, isedge = True):
    """ Set attributes to an elemet of a graph """
    for a in attr.keys():
        if isedge:
            G.edges[elem][a] = attr[a] 
        else:
            G.nodes[elem][a] = attr[a]  
    return G

def graph_construction(networkData):
    
    """ 
    Build (and plot) graph from network dictionary.
    
    """
    
    #Declaration of the network graph
    G = nx.Graph()
        
    Arcs = networkData["Arcs"]
    Valves = networkData["Valves"]
    Pipes = networkData["Pipes"]
    Nodes = networkData["Nodes"]    
    Stations = networkData["Stations"]    

    # create dictionary of nodes elements
    n_elems = {}
    for elem, node in enumerate(sorted(Nodes.keys())):
        n_elems[node]=elem
            
    # build graph
    for node in Nodes.keys():
        elem = n_elems[node]
        G.add_node(elem, name=node)
        G = set_elem_attr(G, elem, Nodes[node], isedge = False)   
    
    for arc in Arcs.keys():       
        row = Arcs[arc]
        n_in = n_elems[row["nodeIN"]]
        n_out = n_elems[row["nodeOUT"]]
        if arc in Pipes.keys():
            category = "pipe"
        elif arc in Valves.keys():
            category = "valve"
        elif arc in Stations.keys():
            category = "compressor station"
        G.add_edge(n_in, n_out, array_origin = n_in, category=category)
        
        # set attributes
        elem = (n_in, n_out)
        G.edges[elem]["name"] = arc
        G = set_elem_attr(G, elem, Arcs[arc])

    return G


#==============================================================================
""" PLOT GRAPH """

def offset_labels(G, pos, offset_x = 0.1,offset_y=0):
    pos_labels = {}
    labels = {}
    labels_to_plot = {}
    keys = pos.keys()
    for key in keys:
        x, y = pos[key]
        pos_labels[key] = (x+offset_x, y+offset_y)
        labels[key] = G.nodes[key]['name']
        if G.nodes[key]['name'].startswith('source'):
            labels_to_plot[key] = 'S' + G.nodes[key]['name'].split('_', 1)[1]
        if G.nodes[key]['name'].startswith('sink'):
            labels_to_plot[key] = 'D' + G.nodes[key]['name'].split('_', 1)[1]
        if G.nodes[key]['name'].startswith('innode'):
            labels_to_plot[key] = 'N' + G.nodes[key]['name'].split('_', 1)[1]
         
    return pos_labels, labels, labels_to_plot

def graph_plot(G, node_labels=False, edge_labels = False):
    
    ## set positions
    ref_node =list(G.nodes)[0]
    pos = {n : [G.nodes[n]["long"], G.nodes[n]["lat"]] for n in G.nodes}

    ## set figure
    fig = plt.figure()        

    ## draw
    # edge_colors
    edge_colors = []
    widths = []
    for a in G.edges:
        # set color
        color = 'tab:green'
        if 'category' in G.edges[a].keys():
            if G.edges[a]["category"] == "pipe": 
                color = "black"
                width = 0.5
            elif "valve" in G.edges[a]["category"]: 
                color = "pink"
                width = 4
            elif G.edges[a]["category"] == "compressor station": 
                color = "red"  
                width = 4
            else:
                color = "black"
                width = 0.5                  
        edge_colors.append(color)
        widths.append(width)
    
    # node colors
    node_colors = ["tab:blue" for n in G.nodes]
    node_sizes = [5 for n in G.nodes]    
    nx.draw_networkx(G, pos=pos, 
            edge_color = edge_colors, 
            node_color = node_colors,
            linewidths = 4,
            with_labels = False,
            node_size = node_sizes)

    # nodes labels
    if node_labels:
        pos_labels, labels, labels_to_plot = offset_labels(G, pos, offset_x = 0, offset_y = 0.05)
        nx.draw_networkx_labels(G, pos=pos_labels, labels=labels_to_plot)

    return 

def plot_graph_with_layout(G, node_labels=False, edge_labels = False):
    ## draw
    # edge_colors
    show_edge_labels = edge_labels
    edge_colors = []
    widths = []
    edge_label_text = {}
    for a in G.edges:
        # set color
        color = 'tab:green'
        if 'category' in G.edges[a].keys():
            if G.edges[a]["category"] == "pipe": 
                color = "black"
                width = 0.5
                edge_label_text[a] = ''
            elif "valve" in G.edges[a]["category"]: 
                color = "pink"
                width = 4
                edge_label_text[a] = ''
            elif G.edges[a]["category"] == "compressor station": 
                color = "red"  
                width = 4
                edge_label_text[a] = ''
            else:
                color = "black"
                width = 0.5  
                edge_label_text[a] = ''
        edge_colors.append(color)
        widths.append(width)

    compressor_labels = {
        (3, 39): 'C4',
        (1, 18): 'C3',
        (5, 25): 'C1',
        (30, 7): 'C6',
        (38, 6): 'C5',
        (0, 10): 'C2',
    }
    for edge, label in compressor_labels.items():
        if G.has_edge(*edge):
            edge_label_text[edge] = label
    
    
    # node colors
    node_colors = ["tab:blue" for n in G.nodes]
    node_sizes = [5 for n in G.nodes]
    layout = nx.kamada_kawai_layout(G)
    
    pos_labels, labels, labels_to_plot = offset_labels(G, pos=layout)
    node_colors, node_types = colorcode_nodes(labels)
    
    plt.figure()
    nx.draw_networkx(G, pos=layout, 
            edge_color = edge_colors, 
            linewidths = 4,
            with_labels = False,
            node_size = node_sizes,
            node_color=[node_colors[node_types[node]] for node in G.nodes()])
    
    # Adjust label positions by adding the offset to the x-coordinate
    label_positions = get_label_positions(layout, labels)

    if node_labels:
        nx.draw_networkx_labels(G, pos=label_positions, labels={node: f'{label}\n' for node, label in labels_to_plot.items()},
                            font_size=7, font_color='black')
    
    if show_edge_labels:
        edge_label_positions = {node: (x + 0.05, y-0.03) for node, (x, y) in layout.items()}

        custom_label_positions = {
            3: (-0.07, 0.15),    # C4
            1: (-0.07, 0.15),    # C3
            5: (0.06, 0.1),      # C1
            0: (-0.17, 0.05),    # C2
            38: (-0.18, 0.05),   # C5
            30: (-0.08, 0.17),   # C6
        }
        for node, (dx, dy) in custom_label_positions.items():
            if node in layout:
                edge_label_positions[node] = (layout[node][0] + dx, layout[node][1] + dy)

        nx.draw_networkx_edge_labels(G, pos = edge_label_positions, edge_labels=edge_label_text, font_color='red', font_size=7, rotate=False)
   
    node_legend_items = [Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=7, label='$\mathrm{Sources (S)}$'),
                         Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=7, label='$\mathrm{Sinks (D)}$'),
                         Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=7, label='$\mathrm{Connecting \: nodes (N)}$'),
                         Line2D([0], [0], color='r', markersize=7, label='$\mathrm{Compressor \: lines (C)}$'),
                         Line2D([0], [0], color='k', markersize=7, label='$\mathrm{Pipes}$')]
    
    # Add legend to the plot
    plt.legend(handles=node_legend_items, loc='lower left')
    plt.axis("off")

    plt.savefig("gaslib40_schematic.pdf")
    plt.savefig("gaslib40_schematic.png")
    return 

def get_label_positions(layout, labels):
    label_positions = {}
    for node, (x,y) in layout.items():
        #Default label position
        label_positions[node] = (x + 0.05, y-0.03)
        node_name = labels[node]
        
        #Change label position if its not good
        if node_name.startswith('source'):
            label_positions[node] = (x - 0.05, y-0.06)
        if node_name.startswith('sink') or node_name =='innode_2' or node_name == 'innode_6':
            label_positions[node] = (x + 0.03, y+0.03)
        if node_name == 'sink_2' or node_name == 'sink_3':
            label_positions[node] = (x - 0.04, y+0.02)
        if node_name == 'innode_5':
            label_positions[node] = (x - 0.03, y+0.03)
        if node_name == 'innode_4' or node_name == 'innode_3':
            label_positions[node] = (x + 0.01, y-0.1)
        if node_name == 'sink_11' or node_name == 'innode_1' or node_name == 'innode_7' or node_name =='sink_15' or node_name == 'sink_4':
            label_positions[node] = (x + 0.07, y - 0.03)
        if node_name =='sink_10' or node_name =='innode_8' or node_name == 'sink_26' or node_name == 'sink_6':
            label_positions[node] = (x - 0.01, y - 0.1)
        if node_name=='sink_23' or node_name =='sink_29':
            label_positions[node] = (x - 0.08, y-0.05)
        if node_name =='sink_25':
            label_positions[node] = (x - 0.08, y)
        if node_name == 'sink_19':
            label_positions[node] = (x, y - 0.1)
        if node_name == 'sink_20':
            label_positions[node] = (x, y + 0.05)
    return label_positions

def colorcode_nodes(labels):
    # Define node colors based on node types
    node_colors = {'source': 'blue', 'sink': 'green', 'bypass': 'yellow', 'other':'lightpink'}

    # Extract node types from labels and assign colors accordingly
    node_types = {}
    for node, label in labels.items():
        if label.startswith('sink') or label.startswith('exit'):
            node_types[node] = 'sink'
        elif label.startswith('source') or label.startswith('entry'):
            node_types[node] = 'source'
        elif label.startswith('innode'):
            node_types[node] = 'bypass'
        else:
            node_types[node] = 'other'
    return node_colors, node_types
'''
if __name__ == "__main__":
    network_data_path = r'D:\\MPC\\pipeline\\gas_net\\data\\data_files\\Gaslib_40\\networkData.xlsx'
    input_data_path = r'D:\\MPC\\pipeline\\gas_net\\data\\data_files\\Gaslib_40\\inputData.xlsx'

    #Load network and input data
    networkData, inputData = import_data_from_excel(network_data_path, input_data_path)
    G = graph_construction(networkData)
    plot_graph_with_layout(G, node_labels=True, edge_labels = True)
'''
#==============================================================================
""" write_data_to_excel """
#==============================================================================
'''
import pandas as pd
import os
'''
path = r"D:\MPC\pipeline\pipe_rewrite"
def write_data_to_excel(sim_data, m_plant, sheets_keys_dict, file_name, terminal_pressure_slack_dict = None, terminal_flow_slack_dict = None, 
                        controller_1_lyapunov = None,
                        controller_2_lyapunov = None,
                        controller_3_lyapunov = None,
                        controller_multistage_lyapunov = None,
                        iteration_times = None,
                        total_runtime = None):
    writer = pd.ExcelWriter(os.path.join(path, file_name))
    for sheet_name, keys in sheets_keys_dict.items():
        data_dict = {}
        for key in keys:
            data_dict[str(key)] = sim_data.get_data_from_key(key)
        df = pd.DataFrame(data_dict)
        df.to_excel(writer, sheet_name=sheet_name)
            
    df = pd.DataFrame()
    if terminal_pressure_slack_dict is not None:
        df['terminal_pressure_slacks'] = terminal_pressure_slack_dict.values()
    
    if terminal_flow_slack_dict is not None:
        df['terminal_flow_slacks'] = terminal_flow_slack_dict.values()
    df.to_excel(writer, sheet_name="terminal_slacks")
    
    df = pd.DataFrame()
    if controller_1_lyapunov is not None:
        df['controller_1_lyapunov'] = controller_1_lyapunov.values()
    if controller_2_lyapunov is not None:
        df['controller_2_lyapunov'] = controller_2_lyapunov.values()
    if controller_3_lyapunov is not None:
        df['controller_3_lyapunov'] = controller_3_lyapunov.values()
    if controller_multistage_lyapunov is not None:
        df['controller_multistage_lyapunov'] = controller_multistage_lyapunov.values()
    
    df.to_excel(writer, sheet_name="controller_lyapunov")

    if iteration_times is not None or total_runtime is not None:
        df = pd.DataFrame()
        if iteration_times is not None:
            df = pd.DataFrame({
                "iteration": list(iteration_times.keys()),
                "iteration_time_s": list(iteration_times.values()),
            })
        if total_runtime is not None:
            df.loc[len(df)] = ["TOTAL", total_runtime]
        df.to_excel(writer, sheet_name="runtime", index=False)

    writer.close()

#==============================================================================
""" plot_dynamic_profiles """
#==============================================================================
'''
import matplotlib.pyplot as plt
import pyomo.environ as pyo
'''
def plot_compressor_beta(m):
    plt.figure()
    compressor_beta = {}
    for c in m.Stations:
        beta = []
        for t in m.Times:
            beta.append(pyo.value(m.compressor_beta[c, t]))
        compressor_beta[c] = beta
        plt.plot(beta, label = c)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Compressor beta (Pout/Pin)")
    plt.legend()
    
def plot_compressor_power(m):
    plt.figure()
    compressor_power = {}
    for c in m.Stations:
        power = []
        for t in m.Times:
            power.append(pyo.value(m.compressor_P[c, t]))
        compressor_power[c] = power
        plt.plot(power, label = c)
    plt.xlabel("Time (hrs)")
    plt.ylabel("Compressor Power (kWh)")
    plt.legend()
