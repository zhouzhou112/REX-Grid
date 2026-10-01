# Model verification and known issues

Release v1.1.0 provides a cleaned implementation supporting the revised manuscript.
The paper and Supplementary Information define the model and assumptions.

## Verification

The model was compared with the reference implementation used for the reported runs.
Both versions received identical stored M–CN2050/LCO 2020 time series and initial
capacities. A second, artificial 2050 construction retained those 2020 stocks to
exercise the later-stage constraints. No optimization or training was performed.
This verifies model construction on shared inputs; it does not regenerate weather
or assert identical solved outcomes.

| Construction | Reference variables | Released variables | Reference constraints | Released constraints |
|---|---:|---:|---:|---:|
| 2020 | 596,875 | 596,874 | 1,014,138 | 973,818 |
| 2050 | 596,915 | 596,914 | 1,014,198 | 973,878 |

The only omitted model objects are one unused constant variable and 40,320
redundant SOC nonnegativity rows (SOC variables already have lower bound zero).
After subtracting them, variable/constraint/nonzero counts, objective and RHS sums,
absolute matrix-coefficient sums, finite-bound sums and constraint groups agree to
relative tolerance `1e-9`. Both models have zero integer variables.
Path configuration, imports, comments and reporting are adapted for distribution;
the capacity inheritance and optimization expressions are retained.

The reference model SHA-256 is
`2f25beeda9afa75b851916d4023503b843401229edf8a81802c739ba40e473fd`.
The source correspondence is direct: `AnnualDispatchOptimizer`,
`initialize_annual_inputs`, `run_case`, the portfolio loader and the intra-provincial
network planner retain their respective roles.

## Known issues retained for consistency with reported results

1. **Terminal fixed O&M.** Substation and converter-station fixed O&M on
interprovincial corridors is evaluated in thousand USD while the other cost terms
are in USD, understating this term by a factor of 1,000 relative to the SI rate.
Ex-post evaluation of all 27 stored main-result cases gives an omitted amount of
0.243287–0.325352% of full path cost, including wind/PV investment. The CMP–LCO
difference changes by at most 0.005680 percentage points; CMP–GRD changes by at
most 0.044724 percentage points. Existing corridors contribute a common 1.479781
billion USD. In the separate CMP-R/LCO check (PSTE-only costs), omissions are
2.282699/2.300459 billion USD and the relative difference changes from 3.215154%
to 3.192481%. Under the same inherited state, a stored solution remains feasible:
each stage's corrected optimum lies between its reported cost and that cost plus
the stage omission. This does not establish an unchanged ranking or a corrected
sequentially reoptimized path.

2. **Base-year corridors.** Only records with status "Operating" in
`data/exist_trasmission.csv` enter base-year capacity. The two Kunliulong records
under construction in 2020 are excluded; additions on these province pairs are
optimized endogenously. This assumption is common to all strategies.

## Input integrity

All 248 distributed model-input files were checked against the reference inputs.
Local saved-folder metadata was removed from 22 Excel distribution copies.
Every worksheet, cell, formula and other workbook component is unchanged; only
the saved-folder metadata in `xl/workbook.xml` differs. Published file hashes are
listed in [input_hashes.csv](input_hashes.csv). Detailed verification records are
retained by the authors.
