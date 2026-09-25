# Data

The pipeline reads only this folder.

| File | Content | Coverage | Source |
|---|---|---|---|
| `sp500_ohlc_daily.csv` | S&P 500 index, daily open, high, low, close | 1999-01-04 to 2018-12-31 (5,031 days) | Yahoo Finance, as bundled with the Python package `arch` |
| `vix_daily.csv` | CBOE VIX, daily close | 1998-12 to 2019-01 | CBOE via datahub.io (`datasets/finance-vix`) |

**Target.** Daily Garman-Klass variance, 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2, annualised (x252). It measures open-to-close variation, so its level (14.8% volatility on average) is below close-to-close volatility (19.1%), which includes overnight moves. Forecast targets are the log of its average over the next 1, 5 and 22 trading days.

**Why not realised variance from intraday data?** It is the better measure and is not freely redistributable. Range-based estimators are the standard substitute; they are noisier, which lowers every model's R2 roughly equally.

**Sample.** The bundled OHLC file ends in 2018, so the test period (2005-2018) includes the 2008 crisis and the 2018 volatility spike and stops before 2020. `scripts/fetch_data.py` rebuilds the files; any OHLC source with the same columns can be dropped in to extend the sample.
