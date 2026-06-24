from pathlib import Path

BASE_PATH = Path(__file__).parent

class Options:
    network_data_path = BASE_PATH / "data" / "data_files" / "Gaslib_40" / "networkData.xlsx"
    input_data_path   = BASE_PATH / "data" / "data_files" / "Gaslib_40" / "inputData_240hrs_gaslib40.xlsx"
    options_data_path = BASE_PATH / "data" / "Options.json"
    ocss_file_path    = BASE_PATH / "optimal_css_24hrs_gaslib40_extended.xlsx"

    horizon = 24
    num_time_periods = 1

    L = 10#28#44#24#32
    scale_obj = 1

    nfe_inf = 5
    ncp_inf = 3
    collocation_scheme = 'LAGRANGE-LEGENDRE'
    remove_collocation = True

    terminal_css_constraint = True
    terminal_error_bound = 0
    T_period = 24.0

    # Lyapunov stability constraint on interm_p. delta: descent margin.
    stability_constraint = True
    delta = 0.1
    # 48-step geometric descent to 10%: 1 - 0.1**(1/48).
    delta_percent = 0.1
    # Use percent descent before this MPC step, then first-stage-cost descent.
    step_change = 0
    delta_finite = 1
    lyapunov_inf_scale = 1

    use_tau_one_for_terminal_cost = True

    linear_solver    = 'ma57'
    tol              = 1e-4
    constr_viol_tol  = 1e-4
    mu_strategy      = "adaptive"
    max_iter         = 10000

    ipopt_mu_init    = 1e-6
    ipopt_bound_push = 1e-6
    ipopt_bound_frac = 1e-8

    plant_horizon = 72
    sample_time   = 1.0

    uncertainty = None

    output_path = BASE_PATH / "Controller_output_gaslib40.xlsx"
    tee_flag    = True


def _import_settings():
    return Options()


if __name__ == "__main__":
    opts = _import_settings()
    print("Configuration loaded:")
    for attr in dir(opts):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = {getattr(opts, attr)}")
