# ReservoirVol report

Seed 20260101.

## 1. Do the reservoirs work as physics says they should?

Mackey-Glass chaotic series (tau = 17), 300 nodes, readout trained on 3,000 points; normalised RMSE on the following points:

|  | linear AR(50) | echo state network | photonic ring reservoir |
|---|---|---|---|
| 1 step ahead | 1.57e-03 | 7.38e-04 | 6.19e-04 |
| 20 steps ahead | 0.4589 | 1.77e-03 | 6.24e-03 |
| 84 steps ahead | 0.5411 | 0.0831 | 0.1643 |

Linear memory capacity (100 nodes, theoretical maximum 100):

| spectral radius / feedback | echo state network | photonic ring reservoir |
|---|---|---|
| 0.2 | 5.5 | 6.1 |
| 0.4 | 9.4 | 10.8 |
| 0.6 | 14.5 | 18.3 |
| 0.8 | 21.2 | 30.9 |
| 0.9 | 25.2 | 44.2 |
| 1.0 | 26.9 | 70.5 |
| 1.1 | 22.3 | 1.8 |
| 1.3 | 16.7 | 0.7 |

## 2. Data and target

S&P 500, 1999-01-05 to 2018-12-31 (5,030 days). Target: log of the average Garman-Klass variance over the next h days. Average Garman-Klass volatility 14.8% (open-to-close) against 19.1% close-to-close. Autocorrelation of daily log variance at lags 1, 5, 22, 66: 0.65, 0.57, 0.47, 0.36 (long memory).

## 3. Hyperparameters, chosen before the test period

Grid search with readouts fitted on data before 2003 and scored on 2003-2004; nothing from 2005 onward is used. 200 nodes.

| reservoir | horizon | selected | ridge | validation MSE |
|---|---|---|---|---|
| echo state network | 1 | spectral radius 0.5, leak 0.5, input scale 0.3 | 0.001 | 0.539 |
| echo state network | 5 | spectral radius 0.95, leak 1, input scale 0.1 | 0.001 | 0.159 |
| echo state network | 22 | spectral radius 0.5, leak 0.5, input scale 1 | 0.001 | 0.275 |
| photonic ring reservoir | 1 | feedback 0.8, input scale 0.05, phase 0 | 0.1 | 0.544 |
| photonic ring reservoir | 5 | feedback 0.8, input scale 0.05, phase 0.4 | 0.001 | 0.161 |
| photonic ring reservoir | 22 | feedback 0.95, input scale 0.2, phase 0.4 | 0.01 | 0.272 |

## 4. Out-of-sample forecasts, 2005-2018

3,522 forecast days; all readouts re-estimated each January on an expanding window; reservoir forecasts average 10 random reservoirs.

### MSE of log variance relative to HAR (below 1 = better)

| model | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| Random walk | 1.452 | 1.320 | 1.512 |
| EWMA (RiskMetrics) | 1.143 | 1.171 | 1.088 |
| GARCH(1,1) | 1.137 | 1.171 | 1.110 |
| HAR | 1.000 | 1.000 | 1.000 |
| HAR-X (VIX, leverage) | 0.901 | 0.865 | 0.905 |
| ESN + HAR-X features | 0.866 | 0.874 | 0.979 |
| Echo state network | 0.868 | 0.868 | 0.996 |
| Photonic ring reservoir | 0.870 | 0.881 | 1.022 |
| Pool: HAR-X + ESN | 0.869 | 0.848 | 0.923 |

### QLIKE relative to HAR

| model | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| Random walk | 1.398 | 1.391 | 1.810 |
| EWMA (RiskMetrics) | 1.103 | 1.126 | 1.043 |
| GARCH(1,1) | 1.073 | 1.083 | 1.009 |
| HAR | 1.000 | 1.000 | 1.000 |
| HAR-X (VIX, leverage) | 0.881 | 0.849 | 0.909 |
| ESN + HAR-X features | 0.846 | 0.847 | 0.990 |
| Echo state network | 0.844 | 0.835 | 0.999 |
| Photonic ring reservoir | 0.850 | 0.839 | 0.986 |
| Pool: HAR-X + ESN | 0.847 | 0.826 | 0.924 |

### Out-of-sample R2

| model | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| Random walk | 0.348 | 0.557 | 0.391 |
| EWMA (RiskMetrics) | 0.486 | 0.607 | 0.562 |
| GARCH(1,1) | 0.489 | 0.607 | 0.553 |
| HAR | 0.551 | 0.665 | 0.597 |
| HAR-X (VIX, leverage) | 0.595 | 0.710 | 0.636 |
| ESN + HAR-X features | 0.611 | 0.707 | 0.606 |
| Echo state network | 0.610 | 0.709 | 0.599 |
| Photonic ring reservoir | 0.609 | 0.705 | 0.589 |
| Pool: HAR-X + ESN | 0.610 | 0.716 | 0.629 |

### Diebold-Mariano tests (negative = the model in the first column is better)

| model | vs | horizon | DM (MSE) | p (MSE) | DM (QLIKE) | p (QLIKE) |
|---|---|---|---|---|---|---|
| Echo state network | HAR | 1 | -9.755 | 3.34e-22 | -6.278 | 3.86e-10 |
| Echo state network | HAR-X (VIX, leverage) | 1 | -4.042 | 5.42e-05 | -3.373 | 7.53e-04 |
| Photonic ring reservoir | HAR | 1 | -10.456 | 3.22e-25 | -6.186 | 6.86e-10 |
| Photonic ring reservoir | HAR-X (VIX, leverage) | 1 | -5.270 | 1.45e-07 | -3.415 | 6.46e-04 |
| ESN + HAR-X features | HAR | 1 | -10.113 | 1.02e-23 | -6.262 | 4.26e-10 |
| ESN + HAR-X features | HAR-X (VIX, leverage) | 1 | -4.050 | 5.23e-05 | -3.006 | 2.67e-03 |
| Pool: HAR-X + ESN | HAR | 1 | -11.216 | 1.04e-28 | -7.143 | 1.11e-12 |
| Pool: HAR-X + ESN | HAR-X (VIX, leverage) | 1 | -7.547 | 5.64e-14 | -5.888 | 4.28e-09 |
| Echo state network | HAR | 5 | -5.199 | 2.12e-07 | -4.886 | 1.07e-06 |
| Echo state network | HAR-X (VIX, leverage) | 5 | 0.211 | 0.833 | -0.893 | 0.372 |
| Photonic ring reservoir | HAR | 5 | -4.422 | 1.01e-05 | -4.250 | 2.19e-05 |
| Photonic ring reservoir | HAR-X (VIX, leverage) | 5 | 1.317 | 0.188 | -0.727 | 0.467 |
| ESN + HAR-X features | HAR | 5 | -5.312 | 1.15e-07 | -4.820 | 1.50e-06 |
| ESN + HAR-X features | HAR-X (VIX, leverage) | 5 | 0.690 | 0.490 | -0.126 | 0.900 |
| Pool: HAR-X + ESN | HAR | 5 | -7.129 | 1.22e-12 | -5.709 | 1.23e-08 |
| Pool: HAR-X + ESN | HAR-X (VIX, leverage) | 5 | -2.471 | 0.014 | -2.999 | 2.72e-03 |
| Echo state network | HAR | 22 | -0.080 | 0.936 | -0.023 | 0.982 |
| Echo state network | HAR-X (VIX, leverage) | 22 | 2.235 | 0.025 | 1.522 | 0.128 |
| Photonic ring reservoir | HAR | 22 | 0.391 | 0.696 | -0.231 | 0.817 |
| Photonic ring reservoir | HAR-X (VIX, leverage) | 22 | 2.747 | 6.05e-03 | 1.623 | 0.105 |
| ESN + HAR-X features | HAR | 22 | -0.585 | 0.559 | -0.215 | 0.830 |
| ESN + HAR-X features | HAR-X (VIX, leverage) | 22 | 2.418 | 0.016 | 1.858 | 0.063 |
| Pool: HAR-X + ESN | HAR | 22 | -2.554 | 0.011 | -2.349 | 0.019 |
| Pool: HAR-X + ESN | HAR-X (VIX, leverage) | 22 | 1.021 | 0.307 | 0.732 | 0.464 |

### ESN relative to HAR-X by period (MSE ratio)

|  | h = 1 | h = 5 | h = 22 |
|---|---|---|---|
| 2005-2007 | 0.950 | 0.940 | 0.970 |
| 2008-2009 (crisis) | 0.940 | 0.970 | 1.271 |
| 2010-2018 | 0.973 | 1.025 | 1.083 |

## 5. Does the random reservoir matter?

MSE relative to HAR for each of the random reservoirs on its own, and for their average:

| reservoir | horizon | single reservoirs: mean | min | max | ensemble |
|---|---|---|---|---|---|
| esn | 1 | 0.871 | 0.868 | 0.876 | 0.868 |
| ring | 1 | 0.871 | 0.869 | 0.873 | 0.870 |
| esn | 5 | 0.888 | 0.878 | 0.903 | 0.868 |
| ring | 5 | 0.883 | 0.878 | 0.892 | 0.881 |
| esn | 22 | 1.011 | 1.000 | 1.029 | 0.996 |
| ring | 22 | 1.057 | 1.019 | 1.100 | 1.022 |

Reservoir size (h = 1, average of 3 reservoirs, same hyperparameters):

| nodes | echo state network | photonic ring reservoir |
|---|---|---|
| 25 | 0.870 | 0.888 |
| 50 | 0.866 | 0.878 |
| 100 | 0.866 | 0.873 |
| 200 | 0.871 | 0.871 |
| 400 | 0.885 | 0.871 |
