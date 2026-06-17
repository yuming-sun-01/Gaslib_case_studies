import pyomo.environ as pyo

from model import buildNonLinearModel, init_network_default
from utils import debug_gas_model, dynamic_demand_calculation


def _solve_steady_model(networkData, inputData, Options, ipopt):
    Options['dynamic'] = False
    Options['T'] = Options['T0'] + Options['dt'] / 3600
    m_steady = init_network_default(
        buildNonLinearModel(networkData, inputData, Options),
        p_default=55e5,
    )

    m_steady.pSource['source_1', :].fix(40)
    m_steady.compressor_beta['compressorStation_1', 0].fix(1.05)
    m_steady.compressor_beta['compressorStation_2', 0].fix(1.05)
    m_steady.compressor_beta['compressorStation_3', 0].fix(1.05)

    res_steady = ipopt.solve(m_steady, tee=True)
    try:
        pyo.assert_optimal_termination(res_steady)
    except Exception:
        debug_gas_model(m_steady)

    print("\n=== Steady pSource ===")
    for s in m_steady.NodesSources:
        for t in m_steady.Times:
            print(f"m_steady.pSource[{s}, {t}] = {pyo.value(m_steady.pSource[s, t])}")

    print("\n=== Steady wSource ===")
    for s in m_steady.NodesSources:
        for t in m_steady.Times:
            print(f"m_steady.wSource[{s}, {t}] = {pyo.value(m_steady.wSource[s, t])}")

    return m_steady


def _build_and_warm_start_finite_model(networkData, inputData, Options, opts, ipopt):
    m_steady = _solve_steady_model(networkData, inputData, Options, ipopt)
    Options['dynamic'] = True
    Options['T'] = opts.horizon
    m_dyn = init_network_default(
        buildNonLinearModel(networkData, inputData, Options),
        p_default=55e5,
    )
    m_dyn.pSource["source_1", :].fix()

    t0 = m_dyn.Times.first()
    for p, vol in m_dyn.Pipes_VolExtrR_interm.data():
        m_dyn.interm_p[p, vol, t0] = m_steady.interm_p[p, vol, t0]
    m_dyn.interm_p[:, :, t0].fix()
    for c in m_dyn.Stations:
        m_dyn.compressor_P[c, t0] = m_steady.compressor_P[c, t0]
    m_dyn.compressor_P[:, t0].fix()

    print("[init] First solve with inputData demands ...")
    res_dyn = ipopt.solve(m_dyn, tee=True)
    try:
        pyo.assert_optimal_termination(res_dyn)
    except Exception:
        debug_gas_model(m_dyn)

    num_time_periods = int(getattr(opts, 'num_time_periods', 1))
    demand_profiles = dynamic_demand_calculation(
        m_dyn, num_time_periods=num_time_periods,
    )
    if getattr(opts, 'uncertainty', None) is not None:
        from utils import uncertain_demand_calculation
        demand_profiles = uncertain_demand_calculation(
            m_dyn, demand_profiles, opts.uncertainty,
        )
    for s in m_dyn.wCons:
        if s[0].startswith("sink"):
            counter = 0
            for t in m_dyn.Times:
                m_dyn.wCons[s[0], 0, t].fix(demand_profiles[s[0]][counter])
                counter += 1

    m_dyn.compressor_beta['compressorStation_1', :].setub(4)

    print("[init] Solving finite block only for warm-start ...")
    m_dyn.pSource["source_1", :].fix()
    ipopt.options['tol'] = 1e-5
    ipopt.options['mu_init'] = 1e-6
    ipopt.options['bound_push'] = 1e-6
    res_finite = ipopt.solve(m_dyn, tee=True)
    if not pyo.check_optimal_termination(res_finite):
        print("[init] WARNING: finite-only solve did not reach optimum; continuing.")

    return m_steady, m_dyn
