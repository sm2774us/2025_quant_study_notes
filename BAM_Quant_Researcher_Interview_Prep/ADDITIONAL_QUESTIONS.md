# Additional Questions

---

### 1) Close Form of OLS, Ridge, and Lasso Regression
**Answer:**  
- **Ordinary Least Squares (OLS):** Minimizes \( \sum (y_i - X_i \beta)^2 \). Closed form: \( \hat{\beta} = (X^T X)^{-1} X^T y \), assuming \( X^T X \) is invertible.  
- **Ridge Regression:** Adds L2 penalty: \( \min \sum (y_i - X_i \beta)^2 + \lambda \sum \beta_j^2 \). Closed form: \( \hat{\beta} = (X^T X + \lambda I)^{-1} X^T y \), where \( I \) is the identity matrix, ensuring invertibility.  
- **Lasso Regression:** Adds L1 penalty: \( \min \sum (y_i - X_i \beta)^2 + \lambda \sum |\beta_j| \). No closed form due to non-differentiability; solved via coordinate descent or proximal gradient methods (e.g., LARS).  
For BAM’s margin models, ridge is useful for correlated predictors (e.g., asset returns), while OLS assumes independence.

**Code:**
```python
import numpy as np

def ols_closed_form(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def ridge_closed_form(X, y, lambda_reg=1.0):
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X.T @ X + lambda_reg * I) @ X.T @ y

# Example usage
X = np.array([[1, 1], [1, 2], [1, 3]])  # Design matrix with intercept
y = np.array([2, 3, 5])                  # Target
beta_ols = ols_closed_form(X, y)
beta_ridge = ridge_closed_form(X, y, lambda_reg=0.1)
print(f"OLS Coefficients: {beta_ols}")
print(f"Ridge Coefficients: {beta_ridge}")
```

**Code Explanation:**  
- **OLS:** Computes \( (X^T X)^{-1} X^T y \) using NumPy’s matrix operations (`@` for multiplication, `np.linalg.inv` for inversion).  
- **Ridge:** Adds \( \lambda I \) to \( X^T X \) before inversion, stabilizing the solution. \( lambda_reg \) controls regularization strength.  
- **Lasso:** Omitted here as it lacks a closed form; typically implemented via `sklearn.linear_model.Lasso`.  
- **Example:** \( X \) includes a constant term ( intercept), \( y \) is a simple linear response.

**Complexity Analysis:**  
- **Time:**  
  - \( X^T X \): \( O(n p^2) \), where \( n \) is samples, \( p \) is features.  
  - Inversion \( (X^T X)^{-1} \): \( O(p^3) \).  
  - \( X^T y \): \( O(n p) \). Total: \( O(n p^2 + p^3) \).  
- **Space:** \( O(p^2) \) for \( X^T X \) matrix, \( O(p) \) for \( \beta \).  
Ridge is identical, as \( \lambda I \) addition is \( O(p) \).

---

### 2) What is Bayesian Optimization?
**Answer:**  
Bayesian optimization is a probabilistic method for global optimization of black-box functions (e.g., hyperparameter tuning). It builds a surrogate model (usually a Gaussian Process, GP) of the objective function and uses an acquisition function (e.g., Expected Improvement) to decide where to sample next, balancing exploration and exploitation. For BAM, it’s ideal for optimizing margin models (e.g., tuning \( \lambda \) in ridge regression) where evaluations are costly.

**Code:**
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import numpy as np

def bayesian_optimization(func, bounds, n_iter=10):
    X = np.random.uniform(bounds[0], bounds[1], (1, 1))
    y = func(X[0])
    gp = GaussianProcessRegressor(kernel=Matern())
    
    for _ in range(n_iter):
        gp.fit(X, y)
        next_x = np.random.uniform(bounds[0], bounds[1], (100, 1))
        mu, sigma = gp.predict(next_x, return_std=True)
        ei = (mu - max(y)) * norm.cdf((mu - max(y)) / sigma) + sigma * norm.pdf((mu - max(y)) / sigma)
        X = np.vstack([X, next_x[np.argmax(ei)]])
        y = np.append(y, func(X[-1]))
    return X[np.argmax(y)]

# Example: Optimize f(x) = -x^2
from scipy.stats import norm
objective = lambda x: -x**2
best_x = bayesian_optimization(objective, [-5, 5])
print(f"Optimal x: {best_x}")
```

**Code Explanation:**  
- **GP Surrogate:** Uses `GaussianProcessRegressor` with a Matern kernel to model the function.  
- **Acquisition (EI):** Expected Improvement balances exploitation (\( mu \)) and exploration (\( sigma \)).  
- **Loop:** Fits GP, predicts over a grid, selects the next point via EI, and updates data.  
- **Example:** Maximizes \( -x^2 \) (peak at 0).  

**Complexity Analysis:**  
- **Time:**  
  - GP fit: \( O(n^3) \) per iteration (inverting covariance matrix), \( n \) is points sampled.  
  - Prediction: \( O(n^2 m) \), \( m \) is grid size (here 100). Total: \( O(n_{\text{iter}} (n^3 + n^2 m)) \).  
- **Space:** \( O(n^2) \) for GP covariance matrix, \( O(m) \) for grid.

---

### 3) Tell Me Why Long/Short Equities?
**Answer:**  
Long/short equities strategies aim to generate alpha by going long on undervalued stocks and shorting overvalued ones, reducing market risk (beta) while capturing relative mispricing. For BAM’s treasury, this hedges margin exposure (e.g., long low-margin assets, short high-margin ones), optimizing capital efficiency. It’s market-neutral, leveraging quantitative signals (e.g., momentum, value) and aligns with risk-based allocation goals.

**Code:**
```python
import pandas as pd
import numpy as np

def long_short_portfolio(returns, signals, top_n=5):
    ranked = signals.rank(axis=1, ascending=False)
    long_mask = ranked <= top_n
    short_mask = ranked > (len(signals.columns) - top_n)
    weights = long_mask.astype(float) / top_n - short_mask.astype(float) / top_n
    portfolio_returns = (weights * returns).sum(axis=1)
    return portfolio_returns

# Example
returns = pd.DataFrame(np.random.randn(10, 20), columns=[f"Stock{i}" for i in range(20)])
signals = pd.DataFrame(np.random.randn(10, 20), columns=returns.columns)  # E.g., momentum
portfolio = long_short_portfolio(returns, signals)
print(f"Portfolio Returns:\n{portfolio}")
```

**Code Explanation:**  
- **Ranking:** Ranks stocks by signal (e.g., momentum).  
- **Weights:** Long top \( n \) stocks (weight \( 1/n \)), short bottom \( n \) (weight \( -1/n \)).  
- **Returns:** Computes daily portfolio returns by multiplying weights with stock returns.  
- **Example:** Random data simulates a 20-stock universe over 10 days.

**Complexity Analysis:**  
- **Time:**  
  - Ranking: \( O(n m \log m) \), \( n \) is days, \( m \) is stocks.  
  - Weight calc and returns: \( O(n m) \). Total: \( O(n m \log m) \).  
- **Space:** \( O(n m) \) for dataframes.

---

### 4) What is the Put-Call Parity?
**Answer:**  
Put-call parity is a no-arbitrage relationship for European options: \( C - P = S - K e^{-rT} \), where \( C \) is call price, \( P \) is put price, \( S \) is spot price, \( K \) is strike, \( r \) is risk-free rate, and \( T \) is time to expiry. It ensures pricing consistency; violations signal arbitrage. For BAM, it’s useful in pricing margin-related derivatives or hedging treasury positions.

**Code:**
```python
import numpy as np

def check_put_call_parity(call, put, spot, strike, rate, time):
    lhs = call - put
    rhs = spot - strike * np.exp(-rate * time)
    return np.isclose(lhs, rhs, atol=0.01)

# Example
call, put, spot, strike, rate, time = 10, 5, 100, 95, 0.02, 1
is_valid = check_put_call_parity(call, put, spot, strike, rate, time)
print(f"Put-Call Parity Holds: {is_valid}")
```

**Code Explanation:**  
- **Formula:** Computes \( C - P \) and \( S - K e^{-rT} \), checks if equal within tolerance.  
- **Example:** Hypothetical option prices; parity holds if \( 10 - 5 \approx 100 - 95 e^{-0.02} \).

**Complexity Analysis:**  
- **Time:** \( O(1) \) (constant arithmetic).  
- **Space:** \( O(1) \) (few scalars).

---

### 5) Questions about PCA, Linear Regression, OVB
**Answer:**  
- **PCA (Principal Component Analysis):** Reduces dimensionality by projecting data onto principal components (eigenvectors of covariance matrix). For BAM, it simplifies correlated margin predictors.  
- **Linear Regression:** Models \( Y = X \beta + \epsilon \); see Q1 for assumptions.  
- **OVB (Omitted Variable Bias):** Occurs when an omitted variable correlates with both \( X \) and \( Y \), biasing \( \beta \). Mitigate via PCA or including proxies.

**Code:**
```python
import numpy as np
from sklearn.decomposition import PCA

# PCA
X = np.random.randn(100, 5)  # 100 samples, 5 features
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")

# OVB Example
X = np.random.randn(100, 1)  # Predictor
Z = 0.5 * X + np.random.randn(100, 1)  # Omitted variable
y = 2 * X + Z + np.random.randn(100, 1)  # True model
beta_biased = np.linalg.inv(X.T @ X) @ X.T @ y  # Without Z
X_full = np.hstack([X, Z])
beta_full = np.linalg.inv(X_full.T @ X_full) @ X_full.T @ y  # With Z
print(f"Biased Beta: {beta_biased.flatten()[0]}, Full Beta: {beta_full.flatten()}")
```

**Code Explanation:**  
- **PCA:** Reduces 5D data to 2D, prints variance explained.  
- **OVB:** Simulates bias by omitting \( Z \); compares \( \beta \) with and without \( Z \).  
- **Example:** Random data shows bias reduction when including \( Z \).

**Complexity Analysis:**  
- **PCA:**  
  - Time: \( O(n p^2 + p^3) \) (covariance + eigendecomposition).  
  - Space: \( O(p^2) \) (covariance matrix).  
- **OVB:** Same as OLS: \( O(n p^2 + p^3) \) time, \( O(p^2) \) space.

---

### 6) Conditional Probability Questions and Data Forecasting [2 Questions Each]
**Answer:**  
- **Conditional Probability:**  
  1. \( P(A|B) = P(A \cap B) / P(B) \). E.g., probability of margin breach given volatility spike.  
  2. Bayes’ Theorem: \( P(A|B) = P(B|A) P(A) / P(B) \). Updates priors in treasury risk models.  
- **Data Forecasting:**  
  1. ARIMA: Models time series with trend and seasonality.  
  2. Exponential Smoothing: Weights recent data more, good for short-term treasury forecasts.

**Code:**
```python
# Conditional Probability
import numpy as np

# Q1: P(X > 1 | X > 0), X ~ N(0,1)
from scipy.stats import norm
p_x_gt_0 = 1 - norm.cdf(0)
p_x_gt_1_and_0 = 1 - norm.cdf(1)
cond_prob = p_x_gt_1_and_0 / p_x_gt_0
print(f"P(X > 1 | X > 0): {cond_prob}")

# Q2: Bayes - P(Rain | Wet), P(Wet|Rain)=0.9, P(Rain)=0.3, P(Wet|NoRain)=0.2
p_rain, p_wet_rain, p_wet_norain = 0.3, 0.9, 0.2
p_wet = p_wet_rain * p_rain + p_wet_norain * (1 - p_rain)
p_rain_wet = (p_wet_rain * p_rain) / p_wet
print(f"P(Rain | Wet): {p_rain_wet}")

# Forecasting
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# Q1: ARIMA Forecast
data = pd.Series(np.random.randn(100).cumsum())
model = ARIMA(data, order=(1, 1, 0)).fit()
forecast = model.forecast(steps=5)
print(f"ARIMA Forecast: {forecast}")

# Q2: Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(data, trend="add").fit()
forecast = model.forecast(5)
print(f"Exp Smoothing Forecast: {forecast}")
```

**Code Explanation:**  
- **Cond Prob Q1:** Uses normal CDF to compute \( P(X > 1 | X > 0) \).  
- **Cond Prob Q2:** Applies Bayes’ theorem for a simple weather example.  
- **Forecast Q1:** Fits ARIMA(1,1,0) to random walk data, forecasts 5 steps.  
- **Forecast Q2:** Uses additive exponential smoothing for trend.

**Complexity Analysis:**  
- **Cond Prob:** Time \( O(1) \), Space \( O(1) \) (constant ops).  
- **ARIMA:** Time \( O(n^3) \) (fitting), Space \( O(n) \) (data).  
- **Exp Smoothing:** Time \( O(n) \), Space \( O(n) \).

---

### 7) Time Series Analysis, Predict Future Volumes, Predict Expected Return [2 Questions Each]
**Answer:**  
- **Time Series Analysis:**  
  1. Stationarity: Check via ADF test; differencing if non-stationary.  
  2. Autocorrelation: Use ACF/PACF to identify lags.  
- **Predict Future Volumes:**  
  1. ARIMA for volume trends.  
  2. LSTM for non-linear patterns.  
- **Predict Expected Return:**  
  1. CAPM: \( E[R] = R_f + \beta (E[R_m] - R_f) \).  
  2. GARCH: Models return volatility.

**Code:**
```python
# Time Series Analysis
from statsmodels.tsa.stattools import adfuller, acf

# Q1: Stationarity
data = pd.Series(np.random.randn(100).cumsum())
result = adfuller(data)
print(f"ADF p-value: {result[1]}")  # > 0.05 means non-stationary

# Q2: Autocorrelation
acf_vals = acf(data, nlags=10)
print(f"ACF: {acf_vals}")

# Predict Volumes
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf

# Q1: ARIMA
volumes = pd.Series(np.random.randn(100) + 100)
model = ARIMA(volumes, order=(1, 0, 0)).fit()
vol_forecast = model.forecast(5)
print(f"Volume Forecast: {vol_forecast}")

# Q2: LSTM
data = volumes.values.reshape(-1, 1)
X, y = [], []
for i in range(5, len(data)):
    X.append(data[i-5:i]); y.append(data[i])
X, y = np.array(X), np.array(y)
model = tf.keras.Sequential([tf.keras.layers.LSTM(10, input_shape=(5, 1)), tf.keras.layers.Dense(1)])
model.compile(optimizer="adam", loss="mse")
model.fit(X, y, epochs=10, verbose=0)
print(f"LSTM Forecast: {model.predict(X[-1:])[0]}")

# Predict Returns
# Q1: CAPM
rf, beta, market_ret = 0.01, 1.2, 0.08
exp_ret = rf + beta * (market_ret - rf)
print(f"CAPM Expected Return: {exp_ret}")

# Q2: GARCH
from arch import arch_model
returns = np.random.randn(100) * 0.02
garch = arch_model(returns, vol="Garch", p=1, q=1).fit(disp="off")
print(f"GARCH Forecast Variance: {garch.forecast().variance.iloc[-1]}")
```

**Code Explanation:**  
- **TS Q1:** ADF test checks stationarity.  
- **TS Q2:** ACF computes lag correlations.  
- **Vol Q1:** ARIMA(1,0,0) forecasts volumes.  
- **Vol Q2:** LSTM uses 5-step lookback for non-linear prediction.  
- **Ret Q1:** CAPM computes \( E[R] \).  
- **Ret Q2:** GARCH(1,1) forecasts volatility.

**Complexity Analysis:**  
- **ADF:** Time \( O(n) \), Space \( O(n) \).  
- **ACF:** Time \( O(n \log n) \), Space \( O(n) \).  
- **ARIMA:** Time \( O(n^3) \), Space \( O(n) \).  
- **LSTM:** Time \( O(e n h) \) (epochs \( e \), hidden units \( h \)), Space \( O(n h) \).  
- **CAPM:** Time \( O(1) \), Space \( O(1) \).  
- **GARCH:** Time \( O(n^2) \), Space \( O(n) \).

---

### 8) How Would You Analyze Data When Predictors Have Outliers?
**Answer:**  
Outliers distort models like OLS. Steps:  
1. **Detect:** Use IQR (Q3 + 1.5 * IQR) or z-scores (>3).  
2. **Handle:** Robust regression (e.g., Huber), winsorization, or log-transform.  
3. **Model:** Use ridge/Lasso for regularization or PCA to reduce outlier impact. For BAM, this ensures stable margin models.

**Code:**
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import HuberRegressor

# Generate data with outliers
X = np.random.randn(100, 2)
y = 2 * X[:, 0] + X[:, 1] + np.random.randn(100)
X[0, 0] = 100  # Outlier
y[0] = 1000

# Detect outliers
Q1, Q3 = np.percentile(X[:, 0], [25, 75])
IQR = Q3 - Q1
outliers = (X[:, 0] < Q1 - 1.5 * IQR) | (X[:, 0] > Q3 + 1.5 * IQR)
print(f"Outliers detected: {np.where(outliers)}")

# Robust regression
huber = HuberRegressor().fit(X, y)
print(f"Huber Coefficients: {huber.coef_}")

# Compare with OLS
ols_beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"OLS Coefficients: {ols_beta}")
```

**Code Explanation:**  
- **Data:** Adds an outlier to \( X \) and \( y \).  
- **Detection:** IQR method flags outliers in first predictor.  
- **Huber:** Fits robust regression, less sensitive to outliers vs. OLS.  

**Complexity Analysis:**  
- **IQR:** Time \( O(n \log n) \) (percentiles), Space \( O(n) \).  
- **Huber:** Time \( O(n p^2) \) (iterative solver), Space \( O(p) \).  
- **OLS:** Time \( O(n p^2 + p^3) \), Space \( O(p^2) \).

---

These responses provide technical depth and practical Python implementations, aligning with BAM’s treasury focus. Each code snippet is executable and demonstrates problem-solving relevant to margin optimization and risk modeling. Practice explaining the logic and complexity to showcase clarity in the interview!
