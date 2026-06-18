import json
import os
import pandas as pd
import numpy as np 
from pathlib import Path
from gas_net.data.compressor_xml_file_reader import get_compressor_coefficients

class DataParserGaslib:
    def __init__(self, folder ="data_files\GasLib_11", 
                 file_names = ["params.json", "network.json", "nominations.json", "slack_nodes.json"]):
        self.folder = folder
        self.file_names = file_names
        self.current_path = os.getcwd()
        self.folder_path = os.path.join(self.current_path, self.folder)
        self.excel_file_path_network = os.path.join(self.folder_path, "networkData.xlsx")
        self.excel_file_path_input = os.path.join(self.folder_path, "inputData.xlsx")
        self.all_arcs = {}
        
    def _write_to_excel(self, excel_file_path, new_sheet_name, column_names, new_data):
        # Check if the file exists
        if Path(excel_file_path).is_file():
            # Load existing Excel file
            xls = pd.ExcelFile(excel_file_path)
        
            # Create a DataFrame from the new data
            if column_names == []:
                new_df = pd.DataFrame(new_data)
            else:
                new_df = pd.DataFrame(new_data, columns = column_names)
        
        
            # Append the new DataFrame to the existing Excel file
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode = 'a') as writer:
                new_df.to_excel(writer, sheet_name=new_sheet_name, index=False, header=True)
        
            print(f'Data appended to {excel_file_path}, sheet: {new_sheet_name}')
        
        else:
            # Create a new Excel file if it doesn't exist
            if column_names == []:
                new_df = pd.DataFrame(new_data)
            else:
                new_df = pd.DataFrame(new_data, columns = column_names)
        
            # Write the new DataFrame to the new Excel file
            new_df.to_excel(excel_file_path, sheet_name = new_sheet_name, index=False, header=True)
        
            print(f'New Excel file created at {excel_file_path}')
     
    def _read_node_data(self, data):
        new_sheet_name = "Nodes"
        column_names =['', 'source', 'pMin', 'pMax', 'height', 'lat', 'long']
        dt = []
        self.source_nodes = []
        self.sink_nodes = []
        self.node_map = {}
        for key in data.keys():
            dt.append([data[key]['name'], 
                       data[key]['name'] if data[key]['name'].startswith('entry') else np.NaN, 
                       data[key]['min_pressure'],
                       data[key]['max_pressure'],
                       data[key]['elevation'],
                       data[key]['x_coord'],
                       data[key]['y_coord']])
            if data[key]['name'].startswith('entry') or data[key]['name'].startswith('source'):
                self.source_nodes.append(data[key]['name'])
            if data[key]['name'].startswith('exit') or data[key]['name'].startswith('sink'):
                self.sink_nodes.append(data[key]['name'])
            self.node_map[int(key)] = data[key]['name']
        self._write_to_excel(self.excel_file_path_network, new_sheet_name, column_names, dt)
    #Nvol	Direction	uMax	MMgas	Tgas	xCH4	Z

    def _read_pipe_data(self, data):
        new_sheet_name = "Pipes"
        column_names = ['', 'length', 'A', 'omega', 'diameter', 'roughness', 'Nvol', 
                        'MMgas', 'Tgas', 'xCH4', 'Z']
        dt = []
        for key in data.keys():
            diameter = data[key]['diameter']
            area = np.pi*diameter**2/4
            perimeter = np.pi*diameter
            dt.append([data[key]['name'],
                      data[key]['length'],
                      area,
                      perimeter,
                      diameter,
                      data[key]['roughness'],
                      int(2),
                      18.5674,
                      self.gas_temperature,
                      0.9,
                      0.86
                ])
            
            self.all_arcs[data[key]['name']] = [self.node_map[data[key]['fr_node']], self.node_map[data[key]['to_node']]] 
        self._write_to_excel(self.excel_file_path_network, new_sheet_name, column_names, dt)
        
    def _read_arc_data(self):
        new_sheet_name = "Arcs"
        column_names = ['', 'nodeIN', 'nodeOUT']
        dt = []
        for key in self.all_arcs.keys():
            node_in = self.all_arcs[key][0]
            node_out = self.all_arcs[key][1]
            dt.append([key,
                       node_in,
                       node_out
                ])
            
        self._write_to_excel(self.excel_file_path_network, new_sheet_name, column_names, dt)
    
    def _read_compressor_data(self, data):
        new_sheet_name = "Stations"
        column_names = ['', 'pInMax', 'pInMin', 'pOutMax', 'max_power']
        dt = []
        for key in data.keys():
            dt.append([data[key]['name'],
                       np.NaN,
                       data[key]['min_inlet_pressure'],
                       data[key]['max_outlet_pressure'],
                       data[key]['max_power']
                ])
            self.all_arcs[data[key]['name']] = [self.node_map[data[key]['fr_node']], self.node_map[data[key]['to_node']]] 
        self._write_to_excel(self.excel_file_path_network, new_sheet_name, column_names, dt) 
            
    def _read_sources_data(self, data):
        new_sheet_name = "SourcesSP_steady_state"
        column_names = ['', 'max_injection', 'min_injection', 'cost']
        dt = []
        for key in data.keys():
            if key.startswith("entry_nominations"):
                for key2 in data[key]:
                   for source in self.source_nodes:
                       if source.endswith(key2):
                           dt.append([source,
                                     data[key][key2]['max_injection'],
                                     data[key][key2]['min_injection'],
                                     data[key][key2]['cost']])
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt)
        
    def _read_sink_data(self, data):
        new_sheet_name = "Sinks_steady_state"
        column_names = ['', 'max_withdrawal', 'min_withdrawal', 'cost']
        dt = []
        for key in data.keys():
            if key.startswith("exit_nominations"):
                for key2 in data[key]:
                   for sink in self.sink_nodes:
                       if sink.endswith(key2):
                           dt.append([sink,
                                     data[key][key2]['max_withdrawal'],
                                     data[key][key2]['min_withdrawal'],
                                     data[key][key2]['cost']])
        
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt)
      
    def _read_gas_parameters(self, data):
        new_sheet_name = "GasParams"
        column_names =["gamma", "Temperature", "Specific_gravity"]
        dt = []
        self.gas_temperature =  data["Temperature (K):"]
        dt.append([data["Specific heat capacity ratio"],
                  data["Temperature (K):"],
                  data["Gas specific gravity (G):"]])
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt)
       
    def _read_compressor_maps_data(self):
        file_path = os.path.join(self.folder_path, "compressor_coefs.cs")
        dt_isoline, dt_eta, dt_surge, dt_choke = get_compressor_coefficients(file_path)
        column_names = []
        new_sheet_name = "station_isoline_coef"
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt_isoline)
        
        new_sheet_name = "station_eta_coef"
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt_eta)
        
        new_sheet_name = "station_surge_coefs"
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt_surge)
        
        new_sheet_name = "station_choke_coefs"
        self._write_to_excel(self.excel_file_path_input, new_sheet_name, column_names, dt_choke)
        
    def read_data(self):
        for name in self.file_names:
            file_path = os.path.join(self.folder_path, name)
            
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
    
            if name == "network.json":
                for key in data.keys():
                    if key == 'nodes':
                       self._read_node_data(data[key])
                    elif key == 'pipes':
                        self._read_pipe_data(data[key])
                    elif key == 'compressors':
                        self._read_compressor_data(data[key])
                   
                self._read_arc_data()
            #Not reading this data yet
            #Need to generate input data since gaslib has steady state consumption
            if name == "nominations.json":
                for key in data.keys():
                    self._read_sources_data(data[key])
                    self._read_sink_data(data[key])
            if name == 'params.json':
                for key in data.keys():
                    self._read_gas_parameters(data[key])
                    
            if name == "slack_nodes.json":
                print(data)
        self._read_compressor_maps_data()

   
 

if __name__ =="__main__":
    dataparser = DataParserGaslib()
    dataparser.read_data()