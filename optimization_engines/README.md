# Mathematical Deep Dive: Treasury Trading Optimization

This document explores the mathematics behind the convex optimization problem used to minimize funding costs in treasury trading, as implemented with solvers like MOSEK, Gurobi, CPLEX, SCIP, and CVXOPT. The problem leverages 2024 U.S. Treasury real yield curve data.

## Problem Statement

We aim to allocate weights $w_i$ to $n$ treasury maturities (e.g., 5 YR, 7 YR, 10 YR, 20 YR, 30 YR) to minimize total funding costs, accounting for interest rates, transfer fees, and haircut constraints.

### Objective Function
The total funding cost to minimize is:

$$C = \sum_{i=1}^{n} (w_i \cdot r_i + f_i)$$

Where:
- $w_i$: Weight (allocation) for treasury $i$ (decision variable, $0 \leq w_i \leq 1$).
- $r_i$: Real yield rate for treasury $i$ (e.g., 2.02% = 0.0202).
- $f_i$: Transfer fee for treasury $i$ (fixed at 0.001 or 0.1%).
- $n$: Number of maturities (here, $n = 5$).

Since $f_i$ is constant, we can rewrite the objective as:

$$C = \sum_{i=1}^{n} w_i r_i + \sum_{i=1}^{n} f_i$$

The second term, $\sum f_i$, is a constant offset (e.g., $0.001 \cdot 5 = 0.005$), so minimizing $C$ is equivalent to minimizing:

$$C' = \sum_{i=1}^{n} w_i r_i$$

### Constraints
1. **Budget Constraint**: Full allocation of weights:
   $$\sum_{i=1}^{n} w_i = 1$$
2. **Haircut Constraint**: Minimum allocation due to haircuts (e.g., 2%):
   $$w_i \geq h_i \quad \forall i, \text{ where } h_i = 0.02$$
3. **Non-negativity**: Weights cannot be negative:
   $$w_i \geq 0 \quad \forall i$$

Since $h_i = 0.02 > 0$, the non-negativity constraint is subsumed by the haircut constraint.

## Mathematical Formulation
This is a **Linear Programming (LP)** problem:

$$\text{Minimize: } C = \sum_{i=1}^{n} w_i r_i + \text{constant}$$
$$\text{Subject to: }$$
$$\sum_{i=1}^{n} w_i = 1$$
$$w_i \geq 0.02 \quad \forall i = 1, \dots, n$$

### Canonical LP Form
In vector notation:
- $\mathbf{w} = [w_1, w_2, \dots, w_n]^T$: Weight vector.
- $\mathbf{r} = [r_1, r_2, \dots, r_n]^T$: Yield vector.
- $\mathbf{f} = [f_1, f_2, \dots, f_n]^T$: Fee vector.
- $\mathbf{h} = [0.02, 0.02, \dots, 0.02]^T$: Haircut vector.

Objective:
$$\text{Minimize: } \mathbf{r}^T \mathbf{w} + \mathbf{f}^T \mathbf{1}$$

Constraints:
$$\mathbf{1}^T \mathbf{w} = 1$$
$$\mathbf{w} \geq \mathbf{h}$$

Where $\mathbf{1}$ is a vector of ones, and $\geq$ is element-wise.

## Feasibility and Optimality
- **Feasibility**: The problem is feasible if $\sum h_i \leq 1$. Here, $n = 5$, $h_i = 0.02$, so $5 \cdot 0.02 = 0.1 \leq 1$, which holds.
- **Convexity**: The objective is linear, and the constraints form a convex polytope (linear equalities and inequalities), ensuring a unique global minimum at a vertex (by LP theory).
- **Optimal Solution**: The solution often allocates most weight to the treasury with the lowest $r_i$ (to minimize $\mathbf{r}^T \mathbf{w}$), constrained by $w_i \geq 0.02$ and $\sum w_i = 1$.

For example, if $\mathbf{r} = [0.0202, 0.0213, 0.0224, 0.0240, 0.0247]$:
- Allocate $w_1 = 0.92$ (max possible after $4 \cdot 0.02 = 0.08$ to others) to the lowest rate (0.0202).
- Set $w_2 = w_3 = w_4 = w_5 = 0.02$.
- Cost: $0.92 \cdot 0.0202 + 0.02 \cdot (0.0213 + 0.0224 + 0.0240 + 0.0247) + 0.005 = 0.02386$.

## Solver Mechanics
All solvers (MOSEK, Gurobi, etc.) use variants of the **Simplex Method** or **Interior Point Methods**:
- **Simplex**: Moves along the polytope’s vertices to the optimal corner (e.g., $[0.92, 0.02, 0.02, 0.02, 0.02]$).
- **Interior Point**: Solves the KKT conditions iteratively within the feasible region.

### KKT Conditions
For this LP:
1. **Stationarity**: $\mathbf{r} + \lambda \mathbf{1} - \boldsymbol{\mu} = 0$, where $\lambda$ is the Lagrange multiplier for equality, $\boldsymbol{\mu} \geq 0$ for inequalities.
2. **Primal Feasibility**: $\mathbf{1}^T \mathbf{w} = 1$, $\mathbf{w} \geq \mathbf{h}$.
3. **Dual Feasibility**: $\boldsymbol{\mu} \geq 0$.
4. **Complementary Slackness**: $\mu_i (w_i - h_i) = 0$.

The solution satisfies these, with $\mu_i > 0$ where $w_i = 0.02$ (active constraints).

## Conclusion
This LP efficiently allocates funds to minimize costs, leveraging the linearity of the objective and constraints. The solvers’ differences lie in implementation (e.g., MOSEK’s conic flexibility isn’t needed here), but the math ensures the same optimal solution.
