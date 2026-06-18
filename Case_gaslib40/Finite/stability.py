# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:51:58 2024

@author: ssnaik
"""

import pyomo.environ as pyo
import pandas as pd


def apply_stability_constraint(m_controller):
    #Define tracking cost 
    m_controller.lyapunov_function_current = pyo.Var(initialize = 1)
    m_controller.lyapunov_function_prev = pyo.Param(initialize = 1, mutable = True)
    m_controller.tracking_cost_plant_prev = pyo.Param(initialize= 1, mutable = True)
    m_controller.delta = pyo.Param(initialize = 0.1)
    def _lyapunov_function_definition(m):
        return m.lyapunov_function_current == (sum((m.interm_p[p, vol, t] - m.interm_p_ocss[p, vol, t])**2 
                                                   for p, vol in m.Pipes_VolExtrR_interm for t in m.Times if t != m.Times.last()) 
                                               + sum((m.compressor_P[s, t] - m.compressor_P_ocss[s, t])**2
                                                     for s in m.Stations for t in m.Times if t != m.Times.last())
                                               )
    m_controller.lyapunov_function_definition = pyo.Constraint(rule = _lyapunov_function_definition)
    
    def _stability_constraint(m):
        return m.lyapunov_function_current <= m.lyapunov_function_prev - m.delta*m.tracking_cost_plant_prev
    m_controller.stability_constraint = pyo.Constraint(rule = _stability_constraint)
    
    
if __name__ == "__main__":
    from gas_net.examples.run_nlp_gaslib40 import run_model
    import pyomo.contrib.mpc as mpc
    _, m_controller = run_model()
    
    

    controller_interface = mpc.DynamicModelInterface(m_controller, m_controller.Times)
    
    #See if this shifts the steady state by 1
    #It does!
    controller_interface.shift_values_by_time(1)
    
    #Rewrite the controller 0 values at the last time point
    
    [m_controller.compressor_beta_ocss[s, 24.0].fix(m_controller.compressor_beta_ocss[s, 0.0]) for s in m_controller.Stations]
    [m_controller.interm_p_ocss[p, vol, 24.0].fix(m_controller.interm_p_ocss[p, vol, 0.0]) for p, vol in m_controller.Pipes_VolExtrR_interm]
    
    