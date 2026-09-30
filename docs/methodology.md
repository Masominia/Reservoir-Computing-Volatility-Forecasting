# Methodology

## 1. Reservoir computing in one paragraph

Take a nonlinear dynamical system with many degrees of freedom and fading memory, drive it with the input signal, and do not train it. Its state at time t is a rich nonlinear function of the recent input history. Train only a linear readout of that state, by ridge regression. All of the "learning" is one linear solve, which is why the idea suits physical hardware: the reservoir can be a bucket of water, a spintronic oscillator or, most successfully, an optical system. Photonic reservoirs built from one laser, one modulator and a fibre delay loop process signals at gigahertz rates (Appeltant et al. 2011; Larger et al. 2012; Paquot et al. 2012; Brunner et al. 2013).

## 2. The two reservoirs

**Echo state network** (Jaeger 2001). N leaky tanh units:

    x_t = (1 - a) x_{t-1} + a tanh(W x_{t-1} + W_in u_t + b),

W sparse Gaussian (10% density) rescaled to spectral radius rho, W_in uniform in [-s, s], bias uniform in [-0.1, 0.1]. Three knobs with physical meaning: rho sets the memory length (rho < 1 gives the echo state property: the state forgets its initial condition), the leak a sets the time scale of the units, and s sets how nonlinear the response is.

**Photonic ring reservoir.** The discrete-time model of an opto-electronic delay-line reservoir. One nonlinear node, a Mach-Zehnder modulator whose transmission is a sine of its drive, feeds a delay loop; within one delay time the input is multiplied by a fixed random mask, creating N "virtual nodes" along the loop. With the input clock slightly detuned from the loop delay, each virtual node is coupled to its neighbour one round trip earlier:

    x_{i,t} = sin(alpha x_{i-1,t-1} + beta (M u_t)_i + phi),

alpha the feedback gain, beta the input gain, phi the modulator bias, M a random +/-1 mask. The coupling matrix is a cyclic shift: a ring. The code is 10 lines and maps one to one onto hardware parameters.

**Readout.** Ridge regression of the target on [state, inputs, 1] with an unpenalised intercept; the penalty is expressed relative to the mean feature variance. Forecasts average an ensemble of 10 independently drawn reservoirs.

## 3. Physics checks of the implementation

* **Mackey-Glass.** dx/dt = 0.2 x(t - 17) / (1 + x(t - 17)^10) - 0.1 x(t), integrated by RK4 with step 0.1 and sampled every 10 steps; chaotic. A readout is trained to predict the value h steps ahead directly from the current state. A linear autoregression on 50 lags is the benchmark: it fails beyond a few steps, the reservoirs do not.
* **Linear memory capacity** (Jaeger 2002). For iid input, MC_k is the squared correlation between u_{t-k} and its best linear reconstruction from x_t; the sum over k cannot exceed N. It rises as the gain approaches 1 and falls beyond, where the autonomous dynamics become unstable ("edge of chaos"). The ring is close to the optimal linear memory (70 of 100 at unit gain) and collapses abruptly above it; the random tanh network has a shorter memory and degrades gradually.
* **Echo state property**: unit test that two different initial conditions converge under the same input.

## 4. Volatility forecasting design

* **Target**: y_t^(h) = log of the mean Garman-Klass variance over days t+1..t+h, h = 1, 5, 22.
* **Reservoir inputs**: today's log variance, return and log VIX squared, standardised with means and scales from the pre-2005 sample only.
* **Benchmarks**, all as "features + linear readout" so that the comparison isolates the features:
  - random walk (mean log variance of the last h days);
  - HAR (Corsi 2009): daily, 5-day and 22-day mean log variance;
  - HAR-X: HAR + log VIX squared + positive and negative parts of the last return (leverage effect). This is the benchmark with the same information as the reservoirs;
  - GARCH(1,1) (Gaussian QML, variance targeting, analytic h-day forecast) and RiskMetrics EWMA, each mapped to the target by a log-linear calibration because the target excludes overnight variance.
* **Walk-forward**: at the start of each year 2005-2018 every readout (and the GARCH parameters) is re-estimated on all rows whose target was fully observed before that date; a unit test verifies that perturbing later targets leaves the forecasts unchanged.
* **Hyperparameters**: grid search (36 echo-state and 18 ring configurations, 4 ridge penalties) with readouts fitted on 1999-2002 and scored on 2003-2004, per horizon. Nothing from the test period is used. The reservoir weights themselves are never trained.
* **Scores**: MSE of log variance; QLIKE on variance, with the log-normal correction exp(prediction + residual variance / 2) from each model's own training residuals; out-of-sample R2; Diebold-Mariano tests with Newey-West variance (h - 1 lags) and the Harvey-Leybourne-Newbold correction.

## 5. Reading the results

* At 1 day both reservoirs beat HAR-X significantly (MSE -3.6%, QLIKE -4%); at 5 days they tie with it; at 22 days they lose to it (+10% MSE) and only match plain HAR.
* A likely reading (not tested separately here): the reservoir's edge is nonlinear interaction between today's variance, return and VIX, a short-horizon effect. Its weakness is memory: volatility has long memory (autocorrelation 0.36 at 66 days), HAR is handed a 22-day average, while a 200-node reservoir retains tens of steps of a noisy input at best. Adding HAR features to the readout does not fix the 22-day horizon: with about 210 regressors and a ridge penalty selected for the reservoir, the readout is too flexible for a target with few independent observations.
* A 50/50 pool of HAR-X and the echo state network is the best or near-best model at 1 and 5 days and within 2% of HAR-X at 22 days.
* Size: 25 nodes already deliver the 1-day gain. The task is low-dimensional; a hardware reservoir would not need to be large.
* Seeds: single reservoirs differ by less than 0.01 in MSE ratio at 1 day and up to 0.08 at 22 days; averaging removes most of it.

## 6. Limitations

* One asset, range-based variance, sample ending in 2018.
* Hyperparameters selected once on a short validation window (2003-2004).
* The gain over HAR-X is statistically clear at 1 day and economically small; no trading or risk-capital evaluation is attempted.
* The ring reservoir is the idealised discrete model: no noise, no finite bandwidth, no quantisation of the readout, all of which matter in hardware.
* No comparison with trained recurrent networks (LSTM), which cost orders of magnitude more to fit; the literature finds them at best comparable to HAR-type models on this problem.

## References

* Jaeger, H. (2001), "The echo state approach to analysing and training recurrent neural networks", GMD Report 148; (2002) "Short term memory in echo state networks", GMD Report 152.
* Maass, W., Natschläger, T. and Markram, H. (2002), "Real-time computing without stable states", *Neural Computation*.
* Appeltant, L. et al. (2011), "Information processing using a single dynamical node as complex system", *Nature Communications*.
* Larger, L. et al. (2012), "Photonic information processing beyond Turing: an optoelectronic implementation of reservoir computing", *Optics Express*.
* Paquot, Y. et al. (2012), "Optoelectronic reservoir computing", *Scientific Reports*.
* Brunner, D. et al. (2013), "Parallel photonic information processing at gigabyte per second data rates using transient states", *Nature Communications*.
* Rodan, A. and Tino, P. (2011), "Minimum complexity echo state network", *IEEE Transactions on Neural Networks*.
* Corsi, F. (2009), "A simple approximate long-memory model of realized volatility", *Journal of Financial Econometrics*.
* Garman, M. and Klass, M. (1980), "On the estimation of security price volatilities from historical data", *Journal of Business*.
* Patton, A. (2011), "Volatility forecast comparison using imperfect volatility proxies", *Journal of Econometrics*.
