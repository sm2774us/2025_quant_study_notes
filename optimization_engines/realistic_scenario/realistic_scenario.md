# Mathematical Deep Dive: Treasury Trading Optimization at Balyasny Asset Management L.P.

This document explores the mathematics behind a treasury trading optimization problem for a Senior Quantitative Researcher at Balyasny Asset Management L.P. (BAM), a multi-strategy quant fund, as of February 19, 2025.

## Scenario Overview

Optimize a $250M treasury portfolio to maximize risk-adjusted net spread (yield minus funding cost) minus transaction costs, using 2024 real yield curve data (5 YR, 7 YR, 10 YR, 20 YR, 30 YR). The model reflects BAM’s systematic, relative-value approach in fixed income.

### Objective Function
Maximize net spread minus transaction costs:

$$S = \sum_{i=1}^{n} w_i (r_i - r_f_i) - c_i |w_i|$$

Where:
- $w_i$: Portfolio weight (positive for long, negative for short).
- $r_i$: Treasury yield (e.g., 2.00% for 5 YR).
- $r_f_i$: Repo rate (e.g., 1.95% for 5 YR).
- $c_i$: Transaction cost per unit weight (e.g., 0.001 for 5 YR).
- $|w_i|$: Absolute weight for costs on both sides.
- $n = 5$: Number of maturities.

Since we maximize, this becomes:

$$\text{Maximize: } \mathbf{w}^T (\mathbf{r} - \mathbf{r}_f) - \mathbf{c}^T |\mathbf{w}|$$

### Constraints
1. **Capital Constraint**: Total leverage limited:
   $$\sum_{i=1}^{n} |w_i| V_i \leq K$$
   Where $V_i = 50M$ (notional per unit), $K = 250M$.

2. **CVaR Constraint**: 95% Conditional Value-at-Risk (e.g., $3M):
   $$\alpha + \\frac{1}{1 - 0.95} \\mathbb{E} [(\\mathbf{w}^T \\mathbf{\\sigma} Z)^+ ] \\leq 3$$
   Approximated via sample scenarios $Z$.

3. **Liquidity Constraint**: Position size respects daily liquidity:
   $$|w_i| V_i \\leq L_i$$
   Where $L_i$ is liquidity (e.g., 100M for 5 YR).

4. **Haircut**: Minimum collateral:
   $$|w_i| \\geq h_i = 0.02$$

## Formulation
This is a **Convex Optimization Problem** (linear objective, convex constraints):

$$\text{Maximize: } \\mathbf{w}^T (\mathbf{r} - \\mathbf{r}_f) - \\mathbf{c}^T |\mathbf{w}|$$
$$\text{Subject to: }$$
$$\\mathbf{1}^T |\mathbf{w}| V \\leq 250$$
$$\\alpha + \\frac{1}{0.05} \\frac{1}{m} \\sum_{j=1}^{m} (\\mathbf{w}^T \\mathbf{\\sigma} Z_j)^+ \\leq 3$$
$$|\mathbf{w}| V \\leq \\mathbf{L}$$
$$|\mathbf{w}| \\geq 0.02$$

Vectors:
- $\mathbf{w}$: Weights.
- $\mathbf{r}$: Yields.
- $\mathbf{r}_f$: Repo rates.
- $\mathbf{c}$: Costs.
- $\mathbf{\\sigma}$: Volatilities.
- $\mathbf{L}$: Liquidity limits.

## Mathematical Insights
- **Convexity**: Linear objective with convex CVaR (piecewise linear) and linear constraints ensure a unique optimum.
- **Relative Value**: Long positions where $r_i > r_f_i$, short where $r_f_i > r_i$, adjusted for costs and risk.
- **CVaR**: More conservative than VaR, aligning with Balyasny’s risk-aware quant culture.

## Solver Approach
Balyasny might use MOSEK or Gurobi for QP/LP with CVaR approximated via scenario sampling:
- **KKT Conditions**:
  $$(\\mathbf{r} - \\mathbf{r}_f) - \\mathbf{c} \\text{sign}(\\mathbf{w}) + \\lambda \\mathbf{1} + \\boldsymbol{\\mu} \\mathbf{\\sigma} - \\boldsymbol{\\nu} = 0$$

## Practical Notes
- **High Frequency**: Real-time data feeds adjust $\\mathbf{r}$ and $\\mathbf{r}_f$.
- **Multi-Strategy**: Integrates with BAM’s broader equity and FX models.
- **Execution**: Liquidity constraints reflect BAM’s focus on tradable edges.
