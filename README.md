# Online Data-Assimilating Virtual Brain Twin (`vbt-01-online-assimilation`)

Sequential Bayesian tracking of a neural mass model's parameters from streaming EEG-like signals.

## Research Gap & Question
* **Gap:** Current virtual brain models are static after offline fitting; none tracks and updates a specific brain's latent parameters continuously as new measurements arrive.
* **Question:** Can an Ensemble Kalman Filter (EnKF) track a slowly drifting excitability parameter in a Jansen-Rit neural mass model from noisy, sub-sampled EEG-like observations, and how does tracking error scale with drift velocity?

## Verification Gate
1. **Sanity Check:** Simulated Jansen-Rit output exhibits non-constant oscillatory dynamics ($\text{std}(y) > 0.1\text{ mV}$) with zero NaNs.
2. **Planted-Signal Recovery:** On synthetic streaming data with a known sinusoidal/linear drift in excitatory synaptic gain $A(t)$, the online EnKF tracks the true parameter trajectory with $\text{RMSE} < 50\%$ of a static offline baseline fit.
