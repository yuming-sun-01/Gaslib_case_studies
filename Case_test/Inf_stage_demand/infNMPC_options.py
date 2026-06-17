from pathlib import Path

BASE_PATH = Path(__file__).parent

class Options:
    network_data_path = BASE_PATH / "test" / "networkData.xlsx"
    input_data_path   = BASE_PATH.parent / "Finite_stage_demand" / "test" / "inputData_10periods.xlsx"
    options_data_path = BASE_PATH / "test" / "Options.json"
    ocss_file_path    = BASE_PATH / "optimal_css_24hrs_test_extended.xlsx"

    horizon = 6
    num_time_periods = 1

    L = 32
    scale_obj = 1

    nfe_inf = 5
    ncp_inf = 3
    collocation_scheme = 'LAGRANGE-LEGENDRE'
    remove_collocation = True

    terminal_css_constraint = True
    terminal_error_bound = 0
    T_period = 6.0

    # Lyapunov stability constraint on interm_p. delta: descent margin.
    stability_constraint = False
    delta = 0.1

    use_tau_one_for_terminal_cost = True

    linear_solver    = 'ma57'
    tol              = 1e-41
    constr_viol_tol  = 1e-4
    mu_strategy      = "adaptive"
    max_iter         = 10000

    ipopt_mu_init    = 1e-6
    ipopt_bound_push = 1e-6
    ipopt_bound_frac = 1e-8

    plant_horizon = 24
    sample_time   = 1.0

    uncertainty = None

    output_path = BASE_PATH / "Controller_output.xlsx"
    tee_flag    = False


def _import_settings():
    return Options()


if __name__ == "__main__":
    opts = _import_settings()
    print("Configuration loaded:")
    for attr in dir(opts):
        if not attr.startswith('_'):
            print(f"  {attr:30s} = {getattr(opts, attr)}")
