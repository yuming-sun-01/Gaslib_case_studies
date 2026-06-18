# Gaslib_case_studies

## Initial condition

In `initialization_tools`:

```python
m_steady.pSource['source_1', :].fix(40)
m_steady.compressor_beta['compressorStation_1', 0].fix(1.05)
m_steady.compressor_beta['compressorStation_2', 0].fix(1.05)
m_steady.compressor_beta['compressorStation_3', 0].fix(1.05)
```

## Model

In `model.py`:

Add collocation for `pipe_mass_balance`: choose `rho` as differentiate variables.

## Time transformation and discretization settings

In `infNMPC_options.py`:

```python
L = 32
nfe_inf = 5
ncp_inf = 3
collocation_scheme = 'LAGRANGE-LEGENDRE'
remove_collocation = True
```

## Terminal constraints

In `infinite_block.py`:

```python
terminal_error = sum(
    (ib.interm_p[p, vol, t_last] - ib.terminal_css_target[p, vol]) ** 2
    for (p, vol) in ib.terminal_css_index
)

ib.terminal_error_constraint = pyo.Constraint(
    expr=terminal_error <= ib.terminal_error_bound
)
```

```text
terminal_error_bound = 0 !!!
```

## Stability constraints

In `stability.py`:

Track rho in both finite and inf horizons.

```text
# from the print info: inf V(k) is much larger (no scaling)
# It computes fast if terminal_css_constraint = False
# Much slower if terminal_css_constraint = True, even as loose as delta=0
```
