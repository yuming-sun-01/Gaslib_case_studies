# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 15:44:41 2024

@author: ssnaik
"""

import os
import pytest
from gas_net.util.import_data import import_data_from_excel
from gas_net.modelling_library.SETS import (NODE_sets, 
                                            ARC_sets, 
                                            STATION_set,
                                            PIPE_sets
                                            )
import pyomo.environ as pyo

_DATA_FILES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "data_files"
)


class TestGaslibParser:
    def test_sets_build_gaslib11(self):

        self.network_data_path = os.path.join(_DATA_FILES_DIR, "GasLib_11", "networkData.xlsx")
        self.input_data_path = os.path.join(_DATA_FILES_DIR, "GasLib_11", "inputData.xlsx")
        
        # import data
        self.networkData, self.inputData = import_data_from_excel(self.network_data_path, self.input_data_path)
        
        #Make model
        m = pyo.ConcreteModel()
        
        #Make node sets
        NODE_sets(m, self.networkData)
        assert len(m.Nodes) == 11
        
        #Make arc sets
        ARC_sets(m, self.networkData)
        
        #Make station set
        STATION_set(m, self.networkData)
        assert len(m.Stations) == 2
        
        #Make pipe set
        PIPE_sets(m, self.networkData)
        assert len(m.Pipes) == 8
        
        assert len(m.Arcs) == len(m.Pipes) + len(m.Stations)
        
     
    def test_sets_build_gaslib40(self):
        self.network_data_path = os.path.join(_DATA_FILES_DIR, "Gaslib_40", "networkData.xlsx")
        self.input_data_path = os.path.join(_DATA_FILES_DIR, "Gaslib_40", "inputData.xlsx")
        
        # import data
        self.networkData, self.inputData = import_data_from_excel(self.network_data_path, self.input_data_path)
        
        #Make model
        m = pyo.ConcreteModel()
        
        #Make node sets
        NODE_sets(m, self.networkData)
        assert len(m.Nodes) == 40
        
        #Make arc sets
        ARC_sets(m, self.networkData)
        print(len(m.Arcs))
        
        #Make station set
        STATION_set(m, self.networkData)
        assert len(m.Stations) == 6
        
        #Make pipe set
        PIPE_sets(m, self.networkData)
        assert len(m.Pipes) == 39
        
        assert len(m.Arcs) == len(m.Pipes) + len(m.Stations)
    
    
        
if __name__ == "__main__":
    pytest.main()
