# Midterm Report  

---

## 1. Introduction

This midterm report presents a comprehensive explanation of the work completed during Week 1 and Week 2 of the project, as implemented in the notebook `submission-week2.ipynb`. The primary objective of this phase was to develop a strong foundation in financial time series analysis by working with real world stock market data, applying statistical techniques, computing returns across multiple horizons, and performing systematic feature engineering.

---

## 2. Libraries and Tools Used

This section analyzes every library imported in the notebook and explains its purpose, concepts enabled and relevance to financial time series analysis.

---

## 2.1 NumPy (`numpy`)

NumPy is the core numerical computation library used throughout the project. It provides efficient array-based computation and underpins the numerical operations performed by higher-level libraries such as Pandas.

### Concepts Learned Using NumPy
- Vectorized numerical computation
- Element wise arithmetic operations
- Numerical precision and stability
- Logarithmic transformations
- Efficient manipulation of large numerical arrays

### Logarithmic (Continuously Compounded) Returns

**Mathematical Formula**

$$
\
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
\
$$

Where:
- $P_t$ is the price at time $t$
- $P_{t-1}$ is the price at the previous time step
- $ln$ denotes the natural logarithm

### Financial Interpretation

Log returns are preferred in quantitative finance because:
- They are additive across time periods
- They handle gains and losses symmetrically
- They align with continuous time financial models
- They improve numerical behavior in statistical modeling

NumPy provides the mathematical foundation required to compute these transformations efficiently.

---

## 2.2 Pandas (`pandas`)

Pandas is the primary data manipulation and time series analysis library used in this project. It introduces labeled, time-aware data structures built on top of NumPy arrays.

### Concepts Learned Using Pandas
- `DataFrame` and `Series` structures
- Time indexed financial data
- Automatic alignment of observations by date
- Percentage change computation
- Rolling window statistics
- Lagged feature creation
- Handling missing values

---

### Simple (Percentage) Returns

**Mathematical Formula**

$$
\
R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
\
$$

### Interpretation

Simple returns measure the relative price change between two consecutive time periods. They are intuitive and commonly used for descriptive and short horizon financial analysis.

---

### Rolling Mean (Moving Average)

**Mathematical Formula**

$$
\
\text{RollingMean}_t = \frac{1}{N} \sum_{i=0}^{N-1} r_{t-i}
\
$$

### Interpretation

Rolling means smooth short term fluctuations and highlight local trends in financial returns. They are widely used to study momentum and trend-following behavior.

---

### Rolling Standard Deviation (Volatility)

**Mathematical Formula**

$$
\
\sigma_t = \sqrt{\frac{1}{N} \sum_{i=0}^{N-1} (r_{t-i} - \mu_t)^2}
\
$$

Where $\mu_t$ is the rolling mean.

### Interpretation

Rolling volatility measures time varying risk and uncertainty in financial markets. Volatility clustering is a key empirical property of asset returns.

---

### Lagged Variables

Lagged variables represent past values of returns. They are essential in time series modeling to capture temporal dependence and market memory.

---

### Missing Values

Rolling and lagging operations introduce missing values at the beginning of the dataset. These observations are removed to ensure consistency and correctness in subsequent analysis.

---

## 2.3 YFinance (`yfinance`)

YFinance is used to download historical financial market data directly from Yahoo Finance.

### Concepts Learned
- OHLCV data structure (Open, High, Low, Close, Volume)
- Adjusted Close prices
- Corporate actions such as dividends and stock splits

---

### Adjusted Close Price

**Definition**

Adjusted Close price reflects the true economic value of a stock by accounting for corporate actions.

### Importance

Using raw closing prices leads to incorrect return calculations. Adjusted prices ensure accurate historical performance measurement.

---

## 2.4 Matplotlib (`matplotlib`)

Matplotlib is used for data visualization and exploratory data analysis.

### Concepts Learned
- Time series visualization
- Trend identification
- Volatility visualization
- Pattern recognition

Visual inspection complements numerical analysis by revealing structural patterns not obvious from statistics alone.

---

## 2.5 SciPy (`scipy`)

SciPy extends NumPy by providing advanced scientific and statistical routines used for quantitative and statistical analysis.

### Concepts Learned Using SciPy
- Probability distributions
- Statistical hypothesis testing
- Numerical methods for scientific computing
- Summary statistics for random variables

---

### Statistical Properties of Returns

Financial returns are treated as realizations of a random variable $r_t$.

**Mean (Expected Return)**

$$
\mu = \mathbb{E}[r_t]
$$

**Variance**

$$
\sigma^2 = \mathbb{E}[(r_t - \mu)^2]
$$

**Standard Deviation**

$$
\sigma = \sqrt{\sigma^2}
$$

These quantities form the statistical foundation for risk and uncertainty measurement in finance.

---

### Hypothesis Testing Framework

Statistical hypothesis testing is used to validate assumptions about financial returns.

- Null hypothesis $H_0$: No abnormal effect or zero mean return  
- Alternative hypothesis $H_1$: Presence of statistically significant effect  

Test statistics and p-values are used to determine whether observed return behavior is statistically distinguishable from randomness.

---

### Financial Interpretation

SciPy enables statistically rigorous analysis of return distributions and supports inferential reasoning beyond descriptive metrics, which is essential for evidence-based financial conclusions.

---

## 2.6 Scikit-learn (`sklearn`)

Scikit-learn is used to prepare financial data for predictive modeling by enforcing structure, scaling, and disciplined data handling.

### Concepts Learned Using Scikit-learn
- Feature scaling and normalization
- Data splitting for model evaluation
- Machine learning workflow structure
- Model readiness of engineered features

---

### Feature Scaling (Standardization)

Financial features often exist on different numerical scales, which can bias modeling outcomes.

**Z-score Normalization**

$$
x^{(scaled)} = \frac{x - \mu}{\sigma}
$$

Where:
- $\mu$ is the feature mean  
- $\sigma$ is the feature standard deviation  

This transformation ensures:
- Zero mean
- Unit variance
- Numerical stability during modeling

---

### Train–Test Separation

Financial time series data must be separated carefully to avoid look-ahead bias. Structured data splitting ensures that future information is not leaked into the training process.

---

### Financial Interpretation

Scikit-learn enforces proper preprocessing and experimental design, ensuring that financial models are evaluated on realistic and unbiased data representations.

---

## 2.7 Statsmodels (`statsmodels`)

Statsmodels is a statistical and econometric modeling library focused on inference, interpretability, and hypothesis testing.

### Concepts Learned Using Statsmodels
- Econometric regression modeling
- Statistical inference on model parameters
- Hypothesis testing
- Time series modeling foundations

---

### Linear Regression Model

Statsmodels represents relationships using the linear econometric form:

$$
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \epsilon
$$

Where:
- $y$ is the dependent variable (e.g., asset returns)
- $x_i$ are explanatory variables (lagged returns, rolling statistics)
- $\beta_i$ are model coefficients
- $\epsilon$ is the stochastic error term

---

### Statistical Significance Testing

Each coefficient is evaluated using a t-statistic:

$$
t_i = \frac{\hat{\beta}_i}{\text{SE}(\hat{\beta}_i)}
$$

Hypotheses tested:
- $H_0$: $\beta_i$ = 0
- $H_1$: $\beta_i$ $\neq$ 0

P-values quantify the probability that observed relationships arise due to random chance.

---

### Financial Interpretation

Statsmodels enables economically interpretable modeling, allowing analysts to understand *why* relationships exist rather than merely predicting outcomes. This is critical in finance, where decision-making relies on statistical validity and economic reasoning.


## 3. Asset Selection and Data Scope

### 3.1 Assets Selected

The analysis focuses on the following large-cap U.S. equities:
- Apple Inc. (AAPL)
- Microsoft Corporation (MSFT)
- Alphabet Inc. (GOOG)
- Amazon.com Inc. (AMZN)
- Tesla Inc. (TSLA)

### Rationale
- High liquidity
- Large market capitalization
- Long, reliable historical data availability

---

### 3.2 Time Horizons

Two distinct time horizons were used:
- Long-term (2015–2024) for statistical understanding
- Medium-term (2021–2024) for return computation and feature engineering

This separation allows both descriptive and modeling-oriented analysis.

---

## 4. Descriptive Statistics

Descriptive statistics summarize historical price behavior using:
- Mean (central tendency)
- Standard deviation (dispersion)
- Minimum and maximum (range)
- Quartiles (distribution shape)

This step establishes baseline familiarity with the data before applying transformations.

---

## 5. Multi-Horizon Return Analysis

Returns were computed over multiple horizons to capture different market dynamics.

**General Formula**

$$
\
R_t^{(k)} = \frac{P_t - P_{t-k}}{P_{t-k}}
\
$$

### Interpretation
- 1-day returns capture short-term noise
- 5-day returns approximate weekly behavior
- 20-day returns represent monthly trends

---

## 6. Feature Engineering

Feature engineering converts raw market data into structured inputs suitable for quantitative modeling.

### Features Created
- Current returns
- Lagged returns
- Rolling mean of returns
- Rolling volatility of returns

These features capture momentum, trend and risk characteristics of financial markets.

---

## 7. Long-Term Investment Analysis

### Mathematical Formula

$$
\
\text{Investment Value} = \text{Initial Investment} \times \frac{P_{\text{current}}}{P_{\text{initial}}}
\
$$

### Interpretation

This analysis demonstrates the effect of compounding and translates abstract return calculations into real economic outcomes.

---

## 8. Key Concepts Learned

- Difference between raw and adjusted prices
- Simple vs logarithmic returns
- Time horizon sensitivity of returns
- Non stationarity of financial markets
- Importance of rolling statistics
- Feature engineering for predictive modeling
- Power of long term compounding

---

## 9. Conclusion

The combined work of Week 1 and Week 2 establishes a foundation in financial time series analysis. By integrating real world data, mathematical reasoning and structured feature engineering, this phase prepares the groundwork for advanced quantitative modeling and strategy development. 

--- 
