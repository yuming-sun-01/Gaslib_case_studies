def _setup_combined_objective(m_dyn, scale_obj=1.0, options=None):
    ib = m_dyn.inf_block
    time_points = sorted(ib.time)

    use_tau_one = True
    if options is not None:
        use_tau_one = getattr(options, 'use_tau_one_for_terminal_cost', True)

    if use_tau_one:
        t_cost = time_points[-1]
    else:
        t_cost = time_points[-2] if len(time_points) >= 2 else time_points[-1]

    obj_name = next(
        (name for name in ('ObjFun', 'obj', 'objective')
         if hasattr(m_dyn, name)),
        None,
    )
    if obj_name is not None:
        obj_comp = getattr(m_dyn, obj_name)
        obj_expr = obj_comp.expr + scale_obj * ib.phi_P[t_cost]
        obj_comp.set_value(obj_expr)
        print(f"[objective] finite power + {scale_obj}*phi_P[tau={t_cost:.4f}] '{obj_name}'")
    else:
        print("[objective] WARNING: No objective found in model")

    return m_dyn
