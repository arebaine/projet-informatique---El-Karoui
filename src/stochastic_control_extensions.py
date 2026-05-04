"""
stochastic_control_extensions.py
=================================

Extensions to ``stochastic_control_core`` that fill the gaps noted in the report
(Barclais & Rebaine) relative to Bachouch, Huré, Langrené, Pham (2020).

Contents
--------
1. **Lipschitz approximation** of the discontinuous terminal cost (Test 2 γ=0).
   The paper uses ``g_N(x) = g(x)`` off the plateau and ``-N*x`` on it, which
   enables the Y&R algorithm to converge and also stabilises the neural
   approximations near the discontinuity.

2. **Y&R algorithm** (Richou 2010, 2011) — the quadratic-BSDE solver that is the
   baseline reference in the paper Table 1. Implemented as a simple backward
   regression on a polynomial basis.

3. **Optimal quantization grids** for the standard Gaussian law in 1D and 2D,
   loaded from precomputed tables. Used by Qknn and Hybrid-LaterQ in the paper.

4. **Pre-training helper** (Section 2.2.3) that warm-starts neural networks
   across time steps with reduced learning rate for stability.

5. **Optimal non-uniform grids for Qknn** (Section 3.5): mean + quantile
   structure as described in the paper for the microgrid problem.

All additions import from ``stochastic_control_core`` and are fully compatible
with the existing Solution / Problem abstractions.
"""

from __future__ import annotations

import math
from functools import lru_cache
import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.polynomial.hermite import hermgauss

import stochastic_control_core as scc
from stochastic_control_core import (
    Tensor,
    SemilinearPDEProblem,
    StochasticControlProblem,
    Quantizer,
    Solution,
    SolverLogs,
    to_tensor,
    set_seed,
    default_device,
)


# ============================================================
# 1. Lipschitz approximation of the discontinuous terminal cost
# ============================================================

def lipschitz_terminal_cost_test2(gamma: float, N_slope: float = 40.0) -> Callable[[Tensor], Tensor]:
    """
    Build the Lipschitz version of the paper's Test 2 terminal cost.

    The original terminal cost is
        g(x) = -x^γ * 1_{0≤x≤1} - 1_{x≥1}
    which is *continuous* for γ ∈ (0, 1] but *discontinuous at x=0* when γ=0.
    For γ=0 the exact value is V(0,0) = -1.

    Richou's algorithm (Y&R) and the DNN approximators need a Lipschitz
    terminal condition to converge cleanly at the discontinuity. The paper
    uses the piecewise approximation (equations (13)–(15) in Richou 2011):

        g_N(x) =  g(x)     if x not in [0, N^{-1/(1-γ)}]
                 -N * x    otherwise.

    For γ=0 we interpret ``N^{-1/(1-γ)} = N^{-1} = 1/N_slope`` and the Lipschitz
    replacement on the plateau is a steep linear ramp ``-N_slope * x``.

    Parameters
    ----------
    gamma : float
        Exponent γ ∈ [0, 1].
    N_slope : float
        Lipschitz constant controlling how aggressively we ramp near x=0.
        The paper used the number of time steps N for this (≈40).

    Returns
    -------
    g : Callable[[Tensor], Tensor]
        Terminal cost function, accepts (B,1) or (B,d) tensors.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie in [0, 1].")

    if gamma <= 0.0:
        # Purely discontinuous case: -1_{x>=0}  → Lipschitz ramp on [0, 1/N]
        x_break = 1.0 / max(N_slope, 1e-6)

        def g(x: Tensor) -> Tensor:
            x1 = x[:, 0]
            out = torch.zeros_like(x1)
            # Below 0 : 0 (as in the original g, since -x^0*1_{0<=x<=1} evaluates to -1 on x>=0)
            # In (0, 1/N]: steep linear ramp from 0 down to -1
            # Above 1/N and below 1: plateau at -1  (original -x^0 = -1)
            # Above 1: -1  (original indicator)
            mask_ramp = (x1 > 0.0) & (x1 <= x_break)
            mask_plat = x1 > x_break
            out = torch.where(mask_ramp, -N_slope * x1, out)
            out = torch.where(mask_plat, -torch.ones_like(x1), out)
            return out
        return g

    # γ = 1 : the terminal cost is already globally Lipschitz, so there is no
    # need to introduce the Richou-style linearisation. This special case also
    # avoids the numerical overflow caused by 1 / (1 - gamma).
    if abs(gamma - 1.0) <= 1e-12:
        def g(x: Tensor) -> Tensor:
            x1 = x[:, 0]
            out = torch.zeros_like(x1)
            mask_unit = (x1 >= 0.0) & (x1 <= 1.0)
            mask_plat = x1 > 1.0
            out = torch.where(mask_unit, -x1, out)
            out = torch.where(mask_plat, -torch.ones_like(x1), out)
            return out
        return g

    # 0 < γ < 1 : original g is continuous but not Lipschitz at 0, so we add a
    # linearisation on [0, x_break] to tame the singular derivative of x^γ.
    denom = max(1.0 - gamma, 1e-6)
    x_break = 1.0 / (max(N_slope, 1e-6) ** (1.0 / denom))
    # At x_break: original value is -(x_break)^gamma. The slope of the linear
    # piece is chosen so that -slope * x_break = -x_break^gamma.
    slope = x_break ** (gamma - 1.0)

    def g(x: Tensor) -> Tensor:
        x1 = x[:, 0]
        out = torch.zeros_like(x1)
        mask_ramp = (x1 > 0.0) & (x1 <= x_break)
        mask_orig = (x1 > x_break) & (x1 <= 1.0)
        mask_plat = x1 > 1.0
        out = torch.where(mask_ramp, -slope * x1, out)
        out = torch.where(
            mask_orig,
            -(torch.clamp(x1, min=1e-12) ** gamma),
            out,
        )
        out = torch.where(mask_plat, -torch.ones_like(x1), out)
        return out
    return g


def make_test2_lipschitz_problem(
    gamma: float,
    T: float = 1.0,
    N: int = 40,
    N_slope: Optional[float] = None,
    action_bounds: Tuple[float, float] = (-3.0, 3.0),
    train_domain: Optional[Tuple[float, float]] = (-1.5, 2.0),
) -> SemilinearPDEProblem:
    """
    Like ``scc.make_test2_semilinear_problem`` but with the Lipschitz
    terminal cost — required for the γ=0 case and numerically
    cleaner for small γ.
    """
    if N_slope is None:
        N_slope = float(N)   # paper heuristic

    g = lipschitz_terminal_cost_test2(gamma=gamma, N_slope=N_slope)

    def sampler(n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        if train_domain is not None:
            lo, hi = train_domain
            u = torch.rand(batch_size, 1, device=device)
            return lo + (hi - lo) * u
        sigma = max(0.35, math.sqrt(max(n * T / N, 1e-8)))
        return sigma * torch.randn(batch_size, 1, device=device)

    return SemilinearPDEProblem(
        state_dim=1,
        action_dim=1,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=action_bounds,
        terminal_cost_fn=g,
        training_sampler=sampler,
        name=f"SemilinearPDE-Test2-Lipschitz-gamma={gamma}-N_slope={N_slope:.1f}",
    )


# ============================================================
# 2. Y&R algorithm (Richou 2010, 2011) — baseline benchmark
# ============================================================

@dataclass
class YRConfig:
    """Configuration for the Y&R algorithm (polynomial regression for BSDE)."""
    n_paths: int = 50_000
    poly_degree: int = 5       # polynomial basis for conditional expectations
    n_mc_inner: int = 1        # one forward sample per path per step
    seed: int = 1234
    device: str = field(default_factory=default_device)
    verbose: bool = True


def _polynomial_basis(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Build a 1D polynomial basis: [1, x, x^2, ..., x^degree].
    x: (B,) or (B,1) array.  Returns (B, degree+1).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return np.stack([x ** k for k in range(degree + 1)], axis=1)


class YRSolver:
    """
    Richou (2011)'s algorithm for quadratic BSDEs.

    Solves the semilinear PDE
        ∂v/∂t + Δv - |∇v|² = 0, v(T, x) = g(x)
    via the BSDE (Y, Z) with driver f(y, z) = |z|², by a backward
    discretisation with smart handling of the quadratic growth:

        Y_n = E[g_N(X_N) | F_n] + sum_{k=n}^{N-1} h * |Z_k|²_truncated

    The conditional expectations are approximated by **polynomial regression**
    on Monte Carlo paths.

    Only meant for the 1D semilinear PDE benchmark (Test 2 in the paper).
    """

    def __init__(self, config: YRConfig) -> None:
        self.config = config
        set_seed(config.seed)

    def solve(self, problem: SemilinearPDEProblem) -> Solution:
        if problem.state_dim != 1 or problem.action_dim != 1:
            raise ValueError("YRSolver only supports 1D semilinear PDE problems.")

        N = problem.horizon
        h = problem.dt
        P = self.config.n_paths
        K = self.config.poly_degree

        # Forward simulate under the driverless dynamics:
        #   dX = sqrt(2) dW   =>   X_{n+1} = X_n + sqrt(2h) eps
        # The Y&R algorithm does NOT know the optimal control a priori,
        # so the forward paths follow the "free" SDE.
        rng = np.random.default_rng(self.config.seed)
        X = np.zeros((P, N + 1), dtype=np.float64)   # X[:, n] = X_n
        X[:, 0] = 0.0
        for n in range(N):
            X[:, n + 1] = X[:, n] + math.sqrt(2.0 * h) * rng.standard_normal(P)

        # Backward pass: Y_N = g(X_N), and
        #   Z_n ≈ E[Y_{n+1} * ΔW / h | X_n]
        #   Y_n = E[Y_{n+1} | X_n] + h * min(|Z_n|², cap)
        # where cap is a truncation for quadratic growth (Richou, Theorem 4.14).
        # Richou's truncation grows polynomially with N.
        cap = 5.0 * math.log(max(N, 2))   # Richou's standard choice

        # Terminal condition — use original g (callable on Tensor)
        x_N_t = torch.from_numpy(X[:, N:N+1].astype(np.float32))
        g_vals = problem.terminal_cost(x_N_t).detach().cpu().numpy().astype(np.float64)
        Y = np.zeros((P, N + 1), dtype=np.float64)
        Y[:, N] = g_vals

        # Store polynomial coefficients so we can query v(t,x) later.
        coefs_Y: List[np.ndarray] = [np.zeros(K + 1) for _ in range(N + 1)]
        coefs_Z: List[np.ndarray] = [np.zeros(K + 1) for _ in range(N)]

        # Terminal fit
        Phi_N = _polynomial_basis(X[:, N], K)
        coefs_Y[N], *_ = np.linalg.lstsq(Phi_N, Y[:, N], rcond=None)

        pbar_iter = reversed(range(N))
        try:
            from tqdm.auto import tqdm
            if self.config.verbose:
                pbar_iter = tqdm(list(pbar_iter), desc="YR ← n", leave=False)
        except ImportError:
            pass

        for n in pbar_iter:
            # Regression target for Z: E[Y_{n+1} * ΔW / h | X_n]
            dW = (X[:, n + 1] - X[:, n]) / math.sqrt(2.0 * h)  # ~ N(0,1)
            Phi_n = _polynomial_basis(X[:, n], K)

            # Z_n  ≈ E[ Y_{n+1} * sqrt(2/h) * eps | X_n ] / sqrt(2)
            # More stable: regress Y_{n+1} * eps / sqrt(h/2) on X_n.
            target_Z = Y[:, n + 1] * dW / math.sqrt(h)
            coefs_Z[n], *_ = np.linalg.lstsq(Phi_n, target_Z, rcond=None)
            Z_est = Phi_n @ coefs_Z[n]

            # Truncate |Z|^2 for stability (Richou 2011, Section 4)
            Z_trunc_sq = np.minimum(Z_est ** 2, cap)

            # Y_n = E[Y_{n+1} | X_n] + h * |Z_n|^2
            target_Y = Y[:, n + 1] + h * Z_trunc_sq
            coefs_Y[n], *_ = np.linalg.lstsq(Phi_n, target_Y, rcond=None)
            Y[:, n] = Phi_n @ coefs_Y[n]

        # Build a Solution object.
        def make_value(n: int) -> Callable[[Tensor], Tensor]:
            c = coefs_Y[n]
            def val(x: Tensor) -> Tensor:
                x_np = x.detach().cpu().numpy().reshape(-1)
                Phi = _polynomial_basis(x_np, K)
                out = Phi @ c
                return torch.from_numpy(out.astype(np.float32)).to(x.device)
            return val

        def make_policy(n: int) -> Callable[[Tensor], Tensor]:
            # Optimal control for this PDE is a = -∇v = -Z  (heuristically).
            # We use the Z polynomial approximation.
            c = coefs_Z[n]
            def pol(x: Tensor) -> Tensor:
                x_np = x.detach().cpu().numpy().reshape(-1)
                Phi = _polynomial_basis(x_np, K)
                out = -(Phi @ c)
                return torch.from_numpy(out.astype(np.float32)).to(x.device).unsqueeze(1)
            return pol

        policies = [make_policy(n) for n in range(N)]
        values = [make_value(n) for n in range(N)]

        return Solution(
            method="Y&R",
            policies=policies,
            value_functions=values,
            policy_nets=[None] * N,
            value_nets=[None] * N,
            logs=SolverLogs(meta={"method": "Y&R", "poly_degree": K, "n_paths": P}),
            problem_name=problem.name,
        )


# ============================================================
# 3. Optimal quantization grids for standard Gaussian
# ============================================================
# The paper uses precomputed optimal L^2 quantizers from
#   http://www.quantize.maths-fi.com
# For dimension 1 with K=21 we use the Lloyd-Max solution.
# We provide the Lloyd-Max algorithm to generate such grids,
# plus Gauss-Hermite as the default fallback.


@lru_cache(maxsize=32)
def lloyd_max_quantizer_1d(K: int = 21, n_iter: int = 100, seed: int = 0) -> Quantizer:
    """Fast Lloyd-Max quantizer of ``N(0,1)`` with ``K`` points.

    The previous implementation used a 400k-point Monte Carlo sample every
    time a script requested a grid.  That made even ``--fast`` look slow for
    the wrong reason.  In one dimension the Lloyd update is analytic: for a
    Voronoi interval ``[a,b]`` the optimal centroid is

        E[Z | a < Z < b] = (phi(a)-phi(b))/(Phi(b)-Phi(a)).

    This produces the same type of L2 quantizer used in the paper, is
    deterministic, and is cached across repeated calls.  The ``seed`` argument
    is kept for backward compatibility but is not used.
    """
    from scipy.stats import norm

    quantiles = np.linspace(1.0 / (2 * K), 1.0 - 1.0 / (2 * K), K)
    points = norm.ppf(quantiles)

    for _ in range(n_iter):
        bins = 0.5 * (points[:-1] + points[1:])
        edges = np.concatenate([[-np.inf], bins, [np.inf]])
        mass = norm.cdf(edges[1:]) - norm.cdf(edges[:-1])
        numer = norm.pdf(edges[:-1]) - norm.pdf(edges[1:])
        new_points = numer / np.maximum(mass, 1e-14)
        if np.max(np.abs(new_points - points)) < 1e-10:
            points = new_points
            break
        points = np.sort(new_points)

    bins = 0.5 * (points[:-1] + points[1:])
    edges = np.concatenate([[-np.inf], bins, [np.inf]])
    weights = norm.cdf(edges[1:]) - norm.cdf(edges[:-1])
    weights = weights / weights.sum()
    return Quantizer(points=points[:, None].astype(np.float32), weights=weights.astype(np.float32), dim=1)


def optimal_quantizer_2d(K: int = 100, n_iter: int = 60, seed: int = 0) -> Quantizer:
    """
    2D competitive learning VQ optimal quantizer of N(0, I_2) with K points.
    """
    rng = np.random.default_rng(seed)
    # Initialize using a random sample
    points = rng.standard_normal((K, 2))
    sample = rng.standard_normal((300_000, 2))

    for it in range(n_iter):
        # Voronoi assignment
        dists = np.linalg.norm(sample[:, None, :] - points[None, :, :], axis=2)
        idx = dists.argmin(axis=1)
        new_points = np.copy(points)
        for k in range(K):
            mask = idx == k
            if mask.any():
                new_points[k] = sample[mask].mean(axis=0)
        if np.max(np.abs(new_points - points)) < 1e-6:
            break
        points = new_points

    # Weights
    dists = np.linalg.norm(sample[:, None, :] - points[None, :, :], axis=2)
    idx = dists.argmin(axis=1)
    weights = np.bincount(idx, minlength=K).astype(np.float64)
    weights = weights / weights.sum()

    return Quantizer(points=points.astype(np.float32), weights=weights, dim=2)


def best_quantizer(K: int = 21, dim: int = 1, method: str = "auto") -> Quantizer:
    """
    Convenience factory: returns the best quantizer available for the given dimension.

    method = "auto" chooses Gauss-Hermite for small K (cheap) and Lloyd-Max for
    K >= 15 (as the paper does).
    """
    if method == "gauss_hermite" or (method == "auto" and dim == 1 and K < 15):
        return scc.gauss_hermite_quantizer_1d(K=K)
    if dim == 1:
        return lloyd_max_quantizer_1d(K=K)
    if dim == 2:
        return optimal_quantizer_2d(K=K)
    raise ValueError(f"No optimal quantizer available for dim={dim}.")


def scale_quantizer(q: Quantizer, scale: Union[float, np.ndarray]) -> Quantizer:
    """Return a copy of ``q`` scaled to the law of the actual noise.

    Qknn expects the quantizer points to have the same distribution as the
    ``eps`` argument passed to ``problem.dynamics``.  In the gas and microgrid
    experiments the dynamics uses shocks ``sigma * Z`` whereas
    ``best_quantizer`` returns a quantizer for ``Z ~ N(0,1)``.  Forgetting this
    scaling was the main reason the Section 3.4/3.5 outputs looked erratic.
    """
    scale_arr = np.asarray(scale, dtype=np.float32)
    return Quantizer(points=np.asarray(q.points, dtype=np.float32) * scale_arr,
                     weights=np.asarray(q.weights, dtype=np.float32).copy(),
                     dim=q.dim)


# ============================================================
# 4. Pre-training / transfer-learning helper (Section 2.2.3)
# ============================================================

def transfer_warm_start_value(
    target_net: nn.Module,
    source_net: Optional[nn.Module],
    lr_reduce_factor: float = 0.1,
) -> Optional[float]:
    """
    Warm-start ``target_net`` from ``source_net`` (same architecture) by copying
    the weights and returning the *suggested* reduced learning rate.

    Used by Hybrid-Now / Hybrid-LaterQ between consecutive time steps to
    exploit the temporal continuity of V_n (Section 2.2.3 of the paper).
    """
    if source_net is None:
        return None
    try:
        target_net.load_state_dict(copy.deepcopy(source_net.state_dict()))
        return lr_reduce_factor
    except Exception:
        # silently skip if architectures don't match
        return None


# ============================================================
# 5. Paper-style non-uniform grids for Qknn (Section 3.5)
# ============================================================

def paper_style_grids_for_microgrid(
    problem: "scc.MicrogridProblem",
    n_C: int = 51,
    n_R: int = 51,
    device: str = "cpu",
) -> List[np.ndarray]:
    """
    Paper-style non-uniform Qknn grids for the microgrid:

        Γ_n = Γ_C × {0,1} × Γ_R^n

    The important point is that Γ_R^n must be an *optimal quantization grid*
    for the law of R_n, not raw Gauss-Hermite quadrature abscissas.  The old
    version used ``method="gauss_hermite"``, whose extreme nodes are far too
    large (around ±14 for 61 nodes), so the grid wasted almost all points in
    states that the AR(1) residual demand never visits.
    """
    C_min = problem.C_min if hasattr(problem, "C_min") else 0.0
    C_max = problem.C_max
    Gamma_C = np.linspace(C_min, C_max, n_C, dtype=np.float32)

    q_R = best_quantizer(K=n_R, dim=1)  # Lloyd-Max/optimal, not Gauss-Hermite nodes
    base_R = np.asarray(q_R.points).reshape(-1).astype(np.float32)

    grids: List[np.ndarray] = []
    for n in range(problem.horizon + 1):
        mean_n = problem.R_bar + (problem.rho ** n) * (problem.r0 - problem.R_bar)
        if problem.rho ** 2 < 1.0:
            var_n = (problem.sigma_R ** 2) * (1.0 - problem.rho ** (2 * n)) / (1.0 - problem.rho ** 2)
        else:
            var_n = (problem.sigma_R ** 2) * n
        std_n = math.sqrt(max(var_n, 1e-8))
        Gamma_R = mean_n + std_n * base_R

        CC, MM, RR = np.meshgrid(
            Gamma_C,
            np.array([0.0, 1.0], dtype=np.float32),
            Gamma_R,
            indexing="ij",
        )
        grids.append(np.stack([CC.ravel(), MM.ravel(), RR.ravel()], axis=1).astype(np.float32))
    return grids


def reachable_inventory_levels_gas(problem: "scc.GasStorageProblem", n: int) -> np.ndarray:
    """Inventory levels reachable from C0 after n discrete gas-storage decisions."""
    vals = {round(float(problem.c0), 10)}
    for _ in range(int(n)):
        nxt = set()
        for c in vals:
            for delta in (float(problem.ain), 0.0, -float(problem.aout)):
                cn = c + delta
                if problem.C_min - 1e-10 <= cn <= problem.C_max + 1e-10:
                    nxt.add(round(cn, 10))
        vals = nxt
    return np.asarray(sorted(vals), dtype=np.float32)


def paper_style_grids_for_gas_storage(
    problem: "scc.GasStorageProblem",
    n_P: int = 51,
    n_C: int = 51,
    p0: float = 4.0,
    reachable_inventory: bool = True,
    price_std_multiplier: float = 2.20,
) -> List[np.ndarray]:
    """
    Paper-style grids for the gas-storage Qknn experiment.

    Corrections compared with the previous version:
      * use a true Gaussian quantizer (Lloyd-Max) for price locations, not
        Gauss-Hermite quadrature nodes;
      * use a price spread wide enough to reproduce the paper's Fig. 9
        axes, without the excessive shocks caused by scaling the noise twice;
      * by default use the inventory levels reachable from C0.  This is what
        creates the horizontal, triangular waiting regions visible in Fig. 9.
    """
    p_bar = problem.p_bar
    beta = problem.beta
    sigma_p = problem.sigma_p

    q_P = best_quantizer(K=n_P, dim=1)
    base_P = np.asarray(q_P.points).reshape(-1).astype(np.float32)

    grids: List[np.ndarray] = []
    for n in range(problem.horizon + 1):
        mean_n = p_bar + (beta ** n) * (p0 - p_bar)
        if n == 0:
            std_n = 1e-4
        else:
            var_n = sigma_p ** 2 * (1.0 - beta ** (2 * n)) / max(1.0 - beta ** 2, 1e-8)
            std_n = math.sqrt(max(var_n, 1e-8))

        Gamma_P_n = mean_n + price_std_multiplier * std_n * base_P
        Gamma_P_n = np.clip(Gamma_P_n, 0.5, 15.0).astype(np.float32)

        if reachable_inventory:
            Gamma_C = reachable_inventory_levels_gas(problem, n)
        else:
            Gamma_C = np.linspace(problem.C_min, problem.C_max, n_C, dtype=np.float32)

        # Guard against very small n where the reachable set contains one point:
        # add a few neighbouring points for interpolation, but keep reachable
        # levels dominant so the figure keeps the paper's discrete C structure.
        if len(Gamma_C) < min(5, n_C):
            pad = np.linspace(max(problem.C_min, problem.c0 - problem.aout * max(n, 1)),
                              min(problem.C_max, problem.c0 + problem.ain * max(n, 1)),
                              min(5, n_C), dtype=np.float32)
            Gamma_C = np.unique(np.concatenate([Gamma_C, pad]).astype(np.float32))

        PP, CC = np.meshgrid(Gamma_P_n, Gamma_C, indexing="ij")
        grids.append(np.stack([PP.ravel(), CC.ravel()], axis=1).astype(np.float32))

    return grids


# ============================================================
# 6. Rich summary helper
# ============================================================

def summary_dict(sol: Solution, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a dict summarising a Solution for easy logging."""
    out: Dict[str, Any] = {
        "method": sol.method,
        "problem": sol.problem_name,
        "horizon": len(sol.policies),
    }
    # Add final-epoch loss if available
    if sol.logs.policy_losses:
        out["policy_loss_final"] = float(np.mean([lst[-1] for lst in sol.logs.policy_losses if lst]))
    if sol.logs.value_losses:
        out["value_loss_final"] = float(np.mean([lst[-1] for lst in sol.logs.value_losses if lst]))
    if extra:
        out.update(extra)
    return out


# ============================================================
# 7. Utility: relative error helper
# ============================================================

def relative_error(estimate: float, reference: float) -> float:
    """Safe relative error |a - b| / max(|b|, eps)."""
    return abs(estimate - reference) / max(abs(reference), 1e-8)


# ============================================================
# 8. Benchmark evaluation wrapper
# ============================================================

def evaluate_semilinear_at_x0(
    sol: Solution,
    problem: SemilinearPDEProblem,
    x0: np.ndarray,
    n_paths: int = 10_000,
    seed: int = 1234,
) -> Dict[str, Any]:
    """Convenience: forward-MC the learned policy from x0 and compute statistics."""
    dev = default_device()
    roll = scc.rollout_policy(problem, sol, x0=x0, n_paths=n_paths, seed=seed, device=dev)
    return {
        "mean": roll["mean"],
        "std": roll["std"],
        "n_paths": n_paths,
    }


if __name__ == "__main__":
    # Simple smoke test
    print("Extensions module loaded successfully.")
    g = lipschitz_terminal_cost_test2(gamma=0.0, N_slope=40.0)
    xs = torch.tensor([[-1.0], [-0.01], [0.0], [0.005], [0.025], [0.5], [1.5]])
    print("g(x) for γ=0 (Lipschitz):")
    print(g(xs))

    q = best_quantizer(K=11, dim=1)
    print("Gauss-Hermite K=11 points:", q.points.reshape(-1))
    print("Weights sum:", q.weights.sum())
