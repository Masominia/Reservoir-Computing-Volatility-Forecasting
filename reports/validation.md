# Validation checks

| area | check | value | threshold | status |
|---|---|---|---|---|
| Physics | Mackey-Glass, 84 steps ahead: ESN error relative to the linear model | 0.1535 | < 0.5 | PASS |
| Physics | memory capacity never exceeds the number of nodes (Jaeger bound) | 70.5383 | <= 100 | PASS |
| Physics | memory falls beyond the edge of chaos, both reservoirs (MC at the largest gain / peak MC) | 0.6214 | < 0.8 | PASS |
| Physics | ring reservoir: memory collapses once the feedback gain exceeds 1 | 9.44e-03 | < 0.1 | PASS |
| Forecast | ESN beats HAR at one day (MSE ratio) | 0.8683 | < 1 | PASS |
| Forecast | HAR beats the random walk at every horizon (sanity of the benchmark) | 1.3202 | > 1 | PASS |
| Forecast | pool of HAR-X and ESN is never worse than HAR-X by more than 3% (MSE) | 1.0188 | < 1.03 | PASS |
| Robustness | largest spread of MSE ratio across random reservoirs | 0.0809 | < 0.10 | PASS |

8 PASS, 0 WARN, 0 FAIL.
