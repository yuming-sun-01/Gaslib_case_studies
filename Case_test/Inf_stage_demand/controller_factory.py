import json

import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom

from infNMPC_options import _import_settings
from data_save_and_plot import save_and_plot_results
from infinite_block import _add_hard_terminal_constraint, _add_infinite_block
from initialization_tools import _build_and_warm_start_finite_model
from objective import _setup_combined_objective
from utils import import_data_from_excel


def _load_config(opts):
    networkData, inputData = import_data_from_excel(
        str(opts.network_data_path),
        str(opts.input_data_path),
    )
    with open(str(opts.options_data_path), 'r') as f:
        Options = json.load(f)
    return networkData, inputData, Options


def _print_model_size(m_dyn):
    print("\n=== Variable counts per component ===")
    rows_v = []
    for block, label in [(m_dyn, 'gasnetwork'), (m_dyn.inf_block, 'inf_block')]:
        for var in block.component_objects(pyo.Var, active=True, descend_into=False):
            n_free = sum(1 for v in var.values() if not v.is_fixed())
            n_fixed = sum(1 for v in var.values() if v.is_fixed())
            rows_v.append((label, var.name.split('.')[-1], n_free, n_fixed))
    rows_v.sort(key=lambda x: -(x[2] + x[3]))
    tot_free = tot_fixed = 0
    for label, var_name, nf, nx in rows_v:
        print(f"  [{label}] {var_name}: free={nf}, fixed={nx}, total={nf+nx}")
        tot_free += nf
        tot_fixed += nx
    print(f"  --- TOTAL free={tot_free}, fixed={tot_fixed}, all={tot_free + tot_fixed} ---")
    print(f"DOF: {degrees_of_freedom(m_dyn)}")

    print("\n=== Constraint counts per component ===")
    rows_c = []
    for block, label in [(m_dyn, 'gasnetwork'), (m_dyn.inf_block, 'inf_block')]:
        for con in block.component_objects(pyo.Constraint, active=True, descend_into=False):
            n = sum(1 for _ in con)
            rows_c.append((label, con.name.split('.')[-1], n))
    rows_c.sort(key=lambda x: -x[2])
    total_c = 0
    for label, con_name, n in rows_c:
        print(f"  [{label}] {con_name}: {n}")
        total_c += n
    print(f"  --- TOTAL active equality+inequality constraints: {total_c} ---\n")


def run_model(opts=None, **kwargs):
    if opts is None:
        opts = _import_settings()

    for key, value in kwargs.items():
        if hasattr(opts, key):
            setattr(opts, key, value)

    networkData, inputData, Options = _load_config(opts)

    ipopt = pyo.SolverFactory('ipopt')
    ipopt.options['linear_solver'] = opts.linear_solver

    m_steady, m_dyn = _build_and_warm_start_finite_model(
        networkData, inputData, Options, opts, ipopt,
    )

    m_dyn = _add_infinite_block(
        m_dyn,
        str(opts.ocss_file_path),
        T_period=opts.T_period,
        nfe_inf=opts.nfe_inf,
        ncp_inf=opts.ncp_inf,
        L=opts.L,
        collocation_scheme=opts.collocation_scheme,
        remove_collocation=bool(getattr(opts, 'remove_collocation', True)),
        terminal_css_constraint=getattr(opts, 'terminal_css_constraint', True),
        add_lyapunov_phi_V=bool(getattr(opts, 'stability_constraint', False)),
    )

    scale_obj = getattr(opts, 'scale_obj', 1.0)
    m_dyn = _setup_combined_objective(m_dyn, scale_obj=scale_obj, options=opts)

    if getattr(opts, 'terminal_css_constraint', True):
        m_dyn = _add_hard_terminal_constraint(
            m_dyn,
            terminal_error_bound=getattr(opts, 'terminal_error_bound', 1.0e-6),
        )

    _print_model_size(m_dyn)

    # Same minimal settings as the finite CSS build (run_model_css_72.py) and
    # the in-loop MPC solver: tol / mu_init / bound_push only.
    ipopt.options["tol"]        = 1e-5
    ipopt.options["mu_init"]    = 1e-6
    ipopt.options["bound_push"] = 1e-6
    ipopt.solve(m_dyn, tee=True)
    return m_steady, m_dyn


def main():
    opts = _import_settings()
    m_steady, m_dyn = run_model(opts)
    save_and_plot_results(m_dyn, opts.output_path)


if __name__ == "__main__":
    main()
