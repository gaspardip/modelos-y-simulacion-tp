# Fix time-weighted order parameter averaging
**Status:** InProgress
**Agent PID:** 129641

## Original Todo
SCIENTIFIC: El promedio del parámetro de orden en utils.py línea 104 no es ponderado por tiempo, puede dar resultados sesgados en transiciones lentas

## Description
Fix the order parameter averaging in utils.py to use proper time-weighted integration instead of simple arithmetic mean. The current implementation samples r every 10 integration steps and takes a simple average, which can introduce bias during slow transitions near the critical coupling. We'll implement trapezoidal rule integration to properly account for the continuous evolution of the order parameter.

## Implementation Plan
- [x] Add `time_weighted_average()` utility function using trapezoidal rule integration (src/utils.py)
- [x] Replace simple averaging with time-weighted averaging in `run_simulation_complete_graph()` (src/utils.py:149)
- [x] Replace simple averaging with time-weighted averaging in `run_simulation()` (src/utils.py:180)
- [x] Update comments to document the physical motivation for time-weighted averaging
- [ ] Test with simple simulation to verify order parameter values are reasonable
- [ ] User test: Run script 1 and verify synchronization curves look smoother near transitions

## Notes
[Implementation notes]