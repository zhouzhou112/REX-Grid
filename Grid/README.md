# Grid: PSTE

This is the cleaned linear programming implementation of the sequential
power-system transition and expansion model. It has continuous variables only.
Thermal commitment is aggregated and relaxed; planning adequacy and spinning
reserve requirements are included.

`main_multiyear.py` runs the seven planning stages, `dispatch_model.py` defines
the model, `utils_yearly.py` loads portfolios and `utils_intra.py` plans the
intra-provincial network. The representative-day procedure uses k-medoids;
extreme-output candidates enter the selection.

Run from the repository root:
`python -m Grid.main_multiyear --mode RL --carbon CN2050`.
This entry retains the M demand pathway. Paths are in `config/paths.json`.
Hourly weather files must be provided before evaluating a case.

See the [main README](../README.md) for units and assumptions.
