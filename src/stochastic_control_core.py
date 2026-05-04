
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union, Any

import numpy as np
import math
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree
from numpy.polynomial.hermite import hermgauss

try:
    from tqdm.auto import tqdm as _tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

def _make_tqdm(iterable, desc: str = "", verbose: bool = True, **kwargs):
    """Wrap iterable with tqdm when available and verbose=True.

    The experiment scripts keep ``verbose=True`` by default so long neural/Qknn
    runs expose progress in notebooks and terminals. ``mininterval`` and
    ``dynamic_ncols`` keep the UI responsive without printing too much.
    """
    if verbose and _TQDM_AVAILABLE:
        kwargs.setdefault("dynamic_ncols", True)
        kwargs.setdefault("mininterval", 1.0)
        return _tqdm(iterable, desc=desc, **kwargs)
    return iterable


def progress_iter(iterable, desc: str = "", enabled: bool = True, **kwargs):
    """Public progress-bar helper used by scripts."""
    return _make_tqdm(iterable, desc=desc, verbose=enabled, **kwargs)


Tensor = torch.Tensor


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(x: Union[np.ndarray, Tensor, float, int], device: Optional[str] = None, dtype: torch.dtype = torch.float32) -> Tensor:
    if isinstance(x, Tensor):
        return x.to(device=device, dtype=dtype) if device is not None else x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad_(True)


def l2_regularization(module: nn.Module) -> Tensor:
    params = list(module.parameters())
    device = params[0].device if params else "cpu"
    reg = torch.tensor(0.0, device=device)
    for p in params:
        reg = reg + (p ** 2).sum()
    return reg


# ----------------------------
# Problem abstraction
# ----------------------------

@dataclass
class StochasticControlProblem:
    state_dim: int
    action_dim: int
    horizon: int
    T: float
    action_mode: str = "continuous"  # "continuous" or "discrete"
    action_bounds: Optional[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]] = None
    discrete_actions: Optional[np.ndarray] = None
    noise_dim: Optional[int] = None
    name: str = "GenericControlProblem"

    def __post_init__(self) -> None:
        if self.noise_dim is None:
            self.noise_dim = self.state_dim
        if self.action_mode not in {"continuous", "discrete"}:
            raise ValueError("action_mode must be 'continuous' or 'discrete'")
        if self.action_mode == "discrete" and self.discrete_actions is None:
            raise ValueError("For discrete control problems, provide discrete_actions.")
        if self.action_mode == "discrete":
            da = np.asarray(self.discrete_actions, dtype=np.float32)
            if da.ndim == 1:
                da = da[:, None]
            self.discrete_actions = da.astype(np.float32)

    @property
    def dt(self) -> float:
        return self.T / self.horizon

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        raise NotImplementedError

    def sample_noise(self, batch_size: int, device: Optional[str] = None) -> Tensor:
        eps = torch.randn(batch_size, self.noise_dim, device=device)
        return eps

    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        raise NotImplementedError

    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        raise NotImplementedError

    def terminal_cost(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def project_action(self, a: Tensor) -> Tensor:
        if self.action_mode == "discrete":
            raise ValueError("project_action is only for continuous controls.")
        if self.action_bounds is None:
            return a
        low, high = self.action_bounds
        low_t = to_tensor(low, device=a.device).view(1, -1)
        high_t = to_tensor(high, device=a.device).view(1, -1)
        return torch.clamp(a, low_t, high_t)

    def action_grid(self, num: int = 101) -> np.ndarray:
        if self.action_mode == "discrete":
            return np.asarray(self.discrete_actions)
        if self.action_dim != 1:
            raise ValueError("Default action_grid is implemented only for 1D controls. Pass a custom grid for multi-dim controls.")
        if self.action_bounds is None:
            raise ValueError("Provide action_bounds or a custom action grid.")
        low, high = self.action_bounds
        return np.linspace(float(np.asarray(low).reshape(-1)[0]), float(np.asarray(high).reshape(-1)[0]), num).reshape(-1, 1)


# ----------------------------
# Example problems from the paper
# ----------------------------

@dataclass
class SemilinearPDEProblem(StochasticControlProblem):
    terminal_cost_fn: Optional[Callable[[Tensor], Tensor]] = None
    training_sampler: Optional[Callable[[int, int, Optional[str]], Tensor]] = None
    diffusion_scale: float = math.sqrt(2.0)
    drift_scale: float = 2.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state_dim != self.action_dim:
            raise ValueError("For the semilinear PDE benchmark, state_dim and action_dim are expected to match.")

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        if self.training_sampler is not None:
            return self.training_sampler(n, batch_size, device)
        sigma = math.sqrt(max(n * self.T / self.horizon, 1e-8))
        return sigma * torch.randn(batch_size, self.state_dim, device=device)

    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        h = self.dt
        return x + self.drift_scale * h * a + self.diffusion_scale * math.sqrt(h) * eps

    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        return self.dt * (a ** 2).sum(dim=1)

    def terminal_cost(self, x: Tensor) -> Tensor:
        if self.terminal_cost_fn is None:
            raise ValueError("Provide terminal_cost_fn.")
        return self.terminal_cost_fn(x)


@dataclass
class LinearQuadraticProblem(StochasticControlProblem):
    B: np.ndarray = field(default_factory=lambda: np.eye(1))
    C: np.ndarray = field(default_factory=lambda: np.ones((1, 1)))
    D_list: List[np.ndarray] = field(default_factory=lambda: [np.eye(1)])
    Q: np.ndarray = field(default_factory=lambda: np.eye(1))
    P: np.ndarray = field(default_factory=lambda: np.eye(1))
    lam: float = 1.0
    training_cov: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # noise_dim = number of independent Brownian motions = len(D_list).
        # Must be set before super().__post_init__() which would overwrite it with state_dim.
        self.noise_dim = len(self.D_list)
        super().__post_init__()

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        if self.training_cov is None:
            cov = np.eye(self.state_dim)
        else:
            cov = np.asarray(self.training_cov, dtype=np.float32)
        z = np.random.multivariate_normal(mean=np.zeros(self.state_dim), cov=cov, size=batch_size).astype(np.float32)
        return torch.tensor(z, device=device)

    def sample_noise(self, batch_size: int, device: Optional[str] = None) -> Tensor:
        return torch.randn(batch_size, len(self.D_list), device=device)

    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        h = self.dt
        B = to_tensor(self.B, device=x.device)
        C = to_tensor(self.C, device=x.device)
        drift = x @ B.T + a @ C.T
        out = x + h * drift
        if eps.ndim == 1:
            eps = eps[:, None]
        for j, D in enumerate(self.D_list):
            D_t = to_tensor(D, device=x.device)
            out = out + math.sqrt(h) * eps[:, j:j+1] * (a @ D_t.T)
        return out

    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        Q = to_tensor(self.Q, device=x.device)
        quad_x = (x * (x @ Q.T)).sum(dim=1)
        quad_a = self.lam * (a ** 2).sum(dim=1)
        return self.dt * (quad_x + quad_a)

    def terminal_cost(self, x: Tensor) -> Tensor:
        P = to_tensor(self.P, device=x.device)
        return (x * (x @ P.T)).sum(dim=1)


# ----------------------------
# Exact / benchmark helpers
# ----------------------------

def semilinear_closed_form_mc(
    g: Callable[[Tensor], Tensor],
    x: Union[np.ndarray, Tensor],
    t: float,
    T: float,
    n_mc: int = 100000,
    seed: int = 1234,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Computes v(t,x) = -log E[exp(-g(x + sqrt(2) W_{T-t}))] by Monte Carlo.
    """
    set_seed(seed)
    device = device or default_device()
    x_t = to_tensor(x, device=device)
    if x_t.ndim == 1:
        x_t = x_t[None, :]
    d = x_t.shape[1]
    z = torch.randn(n_mc, d, device=device) * math.sqrt(T - t)
    vals = []
    batch = 4096
    for i in range(0, n_mc, batch):
        noise = z[i:i+batch]
        shifted = x_t[:, None, :] + math.sqrt(2.0) * noise[None, :, :]
        shifted = shifted.reshape(-1, d)
        g_vals = g(shifted).reshape(x_t.shape[0], -1)
        vals.append(torch.exp(-g_vals))
    expo = torch.cat(vals, dim=1).mean(dim=1)
    v = -torch.log(expo)
    return v.detach().cpu().numpy()


def solve_lq_riccati(problem: LinearQuadraticProblem, n_grid: int = 200) -> Dict[str, Any]:
    d = problem.state_dim
    m = problem.action_dim
    B = np.asarray(problem.B, dtype=float)
    C = np.asarray(problem.C, dtype=float)
    Q = np.asarray(problem.Q, dtype=float)
    P = np.asarray(problem.P, dtype=float)
    D_list = [np.asarray(D, dtype=float) for D in problem.D_list]
    lamI = problem.lam * np.eye(m)

    def rhs_rev(s: float, y: np.ndarray) -> np.ndarray:
        K = y.reshape(d, d)
        M = lamI.copy()
        for D in D_list:
            M = M + D.T @ K @ D
        Minv = np.linalg.inv(M)
        dK = B.T @ K + K @ B + Q - K @ C @ Minv @ C.T @ K
        return dK.reshape(-1)

    y0 = P.reshape(-1)
    sol = solve_ivp(rhs_rev, [0.0, problem.T], y0, dense_output=True, rtol=1e-6, atol=1e-8)
    s_grid = np.linspace(0.0, problem.T, n_grid)
    K_rev = sol.sol(s_grid).T.reshape(n_grid, d, d)
    t_grid = problem.T - s_grid

    def K_of_t(t: float) -> np.ndarray:
        idx = np.searchsorted(np.sort(t_grid), t)
        ts = np.sort(t_grid)
        Ks = K_rev[np.argsort(t_grid)]
        if idx <= 0:
            return Ks[0]
        if idx >= len(ts):
            return Ks[-1]
        t0, t1 = ts[idx - 1], ts[idx]
        w = (t - t0) / max(t1 - t0, 1e-12)
        return (1.0 - w) * Ks[idx - 1] + w * Ks[idx]

    def value(t: float, x: np.ndarray) -> np.ndarray:
        K = K_of_t(t)
        x = np.atleast_2d(x)
        return np.einsum("bi,ij,bj->b", x, K, x)

    def policy(t: float, x: np.ndarray) -> np.ndarray:
        K = K_of_t(t)
        M = lamI.copy()
        for D in D_list:
            M = M + D.T @ K @ D
        Minv = np.linalg.inv(M)
        x = np.atleast_2d(x)
        return -(x @ K.T @ C @ Minv)

    # Expose values sampled on the discrete grid t_n = n * dt for notebook compatibility.
    dt = problem.T / problem.horizon
    K_grid = [K_of_t(n * dt) for n in range(problem.horizon + 1)]
    feedback_grid = []
    for n in range(problem.horizon):
        K = K_grid[n]
        M = lamI.copy()
        for D in D_list:
            M = M + D.T @ K @ D
        Minv = np.linalg.inv(M)
        feedback_grid.append(Minv @ C.T @ K)

    return {"K_of_t": K_of_t, "value": value, "policy": policy, "solve_ivp": sol, "K": K_grid, "feedback": feedback_grid}


# ----------------------------
# Quantization helpers
# ----------------------------

@dataclass
class Quantizer:
    points: np.ndarray
    weights: np.ndarray
    dim: int

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.points.ndim == 1:
            self.points = self.points[:, None]
        if self.points.shape[0] != self.weights.shape[0]:
            raise ValueError("points and weights must have the same first dimension.")
        if self.points.shape[1] != self.dim:
            raise ValueError("points dimension mismatch.")
        self.weights = self.weights / self.weights.sum()

    def to_torch(self, device: Optional[str] = None) -> Tuple[Tensor, Tensor]:
        return to_tensor(self.points, device=device), to_tensor(self.weights, device=device)


def gauss_hermite_quantizer_1d(K: int = 21) -> Quantizer:
    """
    Approximate N(0,1) expectation using Gauss-Hermite quadrature:
      E[f(Z)] = 1/sqrt(pi) sum_i w_i f(sqrt(2) x_i), Z~N(0,1)
    """
    xs, ws = hermgauss(K)
    points = np.sqrt(2.0) * xs
    weights = ws / np.sqrt(np.pi)
    return Quantizer(points=points[:, None], weights=weights, dim=1)


# ----------------------------
# Neural building blocks
# ----------------------------

class RunningNormalizer(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.eps = eps

    def update(self, x: Tensor) -> None:
        if x.ndim != 2:
            return
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean.copy_(new_mean.detach())
        self.var.copy_(new_var.detach())
        self.count.copy_(tot_count.detach())

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_sizes: Sequence[int], activation: str = "elu", normalize_input: bool = True) -> None:
        super().__init__()
        self.normalizer = RunningNormalizer(in_dim) if normalize_input else None
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(make_activation(activation))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.normalizer is not None:
            if self.training:
                self.normalizer.update(x.detach())
            x = self.normalizer(x)
        return self.net(x)


class ContinuousPolicyNet(nn.Module):
    def __init__(self, problem: StochasticControlProblem, hidden_sizes: Sequence[int], activation: str = "elu", normalize_input: bool = True) -> None:
        super().__init__()
        self.problem = problem
        self.body = MLP(problem.state_dim, problem.action_dim, hidden_sizes, activation, normalize_input)

    def forward(self, x: Tensor) -> Tensor:
        raw = self.body(x)
        if self.problem.action_bounds is None:
            return raw
        low, high = self.problem.action_bounds
        low_t = to_tensor(low, device=x.device).view(1, -1)
        high_t = to_tensor(high, device=x.device).view(1, -1)
        return low_t + 0.5 * (torch.tanh(raw) + 1.0) * (high_t - low_t)


class DiscretePolicyNet(nn.Module):
    def __init__(self, problem: StochasticControlProblem, hidden_sizes: Sequence[int], activation: str = "elu", normalize_input: bool = True) -> None:
        super().__init__()
        if problem.action_mode != "discrete":
            raise ValueError("DiscretePolicyNet requires a discrete-action problem.")
        self.problem = problem
        self.L = problem.discrete_actions.shape[0]
        self.body = MLP(problem.state_dim, self.L, hidden_sizes, activation, normalize_input)

    def logits(self, x: Tensor) -> Tensor:
        return self.body(x)

    def forward(self, x: Tensor) -> Tensor:
        logits = self.logits(x)
        # For finite-action problems with state-dependent constraints (gas
        # storage, microgrid discretisations), remove impossible actions before
        # the softmax.  Masking only at greedy-evaluation time is too late: the
        # training loss can otherwise put probability mass on infeasible actions
        # near the inventory boundaries, which is one of the reasons the maps
        # become visually incoherent.
        mask_matrix_fn = getattr(self.problem, "admissible_action_matrix_torch", None)
        if mask_matrix_fn is not None:
            mask = mask_matrix_fn(x)
            logits = logits.masked_fill(~mask, -1e9)
        return torch.softmax(logits, dim=1)

    def greedy_action(self, x: Tensor) -> Tensor:
        """Return the highest-probability feasible discrete action.

        For constrained finite-action problems such as gas storage, a plain
        argmax can select an action that violates the state constraint near
        C_min/C_max.  That produced huge Q+ penalties in rollouts and made the
        neural estimates look much worse than the paper.  When the problem
        provides ``admissible_action_mask_np``, mask infeasible actions before
        taking the argmax; otherwise fall back to the standard greedy rule.
        """
        probs = self.forward(x)
        mask_fn = getattr(self.problem, "admissible_action_mask_np", None)
        if mask_fn is None:
            idx = probs.argmax(dim=1)
            actions = to_tensor(self.problem.discrete_actions, device=x.device)
            return actions[idx]

        x_np = x.detach().cpu().numpy()
        probs_np = probs.detach().cpu().numpy()
        acts_np = np.asarray(self.problem.discrete_actions, dtype=np.float32)
        if acts_np.ndim == 1:
            acts_np = acts_np[:, None]
        B = x_np.shape[0]
        feasible_cols = []
        for a in acts_np:
            a_batch = np.repeat(a.reshape(1, -1), B, axis=0).astype(np.float32)
            feasible_cols.append(np.asarray(mask_fn(x_np, a_batch), dtype=bool).reshape(B, 1))
        feasible = np.concatenate(feasible_cols, axis=1)
        masked = np.where(feasible, probs_np, -np.inf)
        no_feasible = ~np.isfinite(masked).any(axis=1)
        if np.any(no_feasible):
            masked[no_feasible] = probs_np[no_feasible]
        idx_np = np.argmax(masked, axis=1)
        out = acts_np[idx_np]
        return to_tensor(out, device=x.device, dtype=x.dtype)


class ValueNet(nn.Module):
    def __init__(self, problem: StochasticControlProblem, hidden_sizes: Sequence[int], activation: str = "elu", normalize_input: bool = True) -> None:
        super().__init__()
        self.body = MLP(problem.state_dim, 1, hidden_sizes, activation, normalize_input)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x).squeeze(-1)


# ----------------------------
# Solution containers
# ----------------------------

@dataclass
class SolverLogs:
    policy_losses: List[List[float]] = field(default_factory=list)
    value_losses: List[List[float]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    method: str
    policies: List[Callable[[Tensor], Tensor]]
    value_functions: List[Callable[[Tensor], Tensor]]
    policy_nets: List[Optional[nn.Module]]
    value_nets: List[Optional[nn.Module]]
    logs: SolverLogs
    problem_name: str

    def policy(self, n: int, x: Tensor) -> Tensor:
        return self.policies[n](x)

    def value(self, n: int, x: Tensor) -> Tensor:
        return self.value_functions[n](x)


# ----------------------------
# Evaluation helpers
# ----------------------------

@torch.no_grad()
def rollout_policy(
    problem: StochasticControlProblem,
    solution: Solution,
    x0: Union[np.ndarray, Tensor],
    n_paths: int = 10000,
    seed: int = 1234,
    device: Optional[str] = None,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> Dict[str, Any]:
    set_seed(seed)
    device = device or default_device()
    x = to_tensor(x0, device=device)
    if x.ndim == 1:
        x = x[None, :]
    x = x.repeat(n_paths, 1)
    total = torch.zeros(n_paths, device=device)
    trajectory = [x.detach().cpu()]
    # Keep rollouts silent by default: they are frequent and short-horizon,
    # and their per-time-step tqdm bars clutter notebooks. Algorithm-level
    # progress bars still show where the run is.
    for n in range(problem.horizon):
        a = solution.policy(n, x)
        total = total + problem.running_cost(x, a, n)
        eps = problem.sample_noise(n_paths, device=device)
        x = problem.dynamics(x, a, eps, n)
        trajectory.append(x.detach().cpu())
    total = total + problem.terminal_cost(x)
    return {
        "costs": total.detach().cpu().numpy(),
        "mean": float(total.mean().item()),
        "std": float(total.std(unbiased=False).item()),
        "trajectory": trajectory,
    }


# ----------------------------
# Base neural solver
# ----------------------------

@dataclass
class NeuralSolverConfig:
    hidden_sizes: Sequence[int] = (64, 64, 64)
    activation: str = "elu"
    learning_rate: float = 1e-3
    batch_size: int = 512
    epochs: int = 50
    n_batches_per_epoch: int = 10   # gradient steps per epoch (paper uses mini-batches)
    l2_reg: float = 1e-4
    transfer_warm_start: bool = True
    device: str = field(default_factory=default_device)
    verbose: bool = True
    seed: int = 1234
    grad_clip: Optional[float] = 5.0
    # Optional controls used by finite-action Hybrid-Now. Keeping them here
    # avoids hard-coded expensive quadrature in fast runs.
    hybrid_train_noise: Optional[int] = None
    hybrid_eval_noise: Optional[int] = None


class BaseNeuralSolver:
    def __init__(self, config: NeuralSolverConfig) -> None:
        self.config = config
        set_seed(config.seed)

    def _train_loop(
        self,
        module: nn.Module,
        loss_closure: Callable[[], Tensor],
        lr: Optional[float] = None,
        desc: str = "",
    ) -> List[float]:
        module.train()
        opt = torch.optim.Adam(module.parameters(), lr=lr or self.config.learning_rate)
        losses = []
        n_steps = self.config.epochs * self.config.n_batches_per_epoch
        # Minimal progress policy: do not create a tqdm bar for every mini-batch.
        # Colab renders nested/cleared bars as many persistent lines, which makes
        # long gas-storage runs unreadable. Caller-level bars still show the
        # current algorithm and backward time index; this loop stays silent.
        for _ in range(n_steps):
            opt.zero_grad()
            loss = loss_closure()
            if self.config.l2_reg > 0:
                loss = loss + self.config.l2_reg * l2_regularization(module)
            loss.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        return losses

    def _new_continuous_policy(self, problem: StochasticControlProblem) -> nn.Module:
        if problem.name == "OptionHedging":
            return HedgingPolicyNet(
                problem,
                self.config.hidden_sizes,
                self.config.activation,
            ).to(self.config.device)
    
        return ContinuousPolicyNet(
            problem,
            self.config.hidden_sizes,
            self.config.activation,
        ).to(self.config.device)
    def _new_discrete_policy(self, problem: StochasticControlProblem) -> DiscretePolicyNet:
        return DiscretePolicyNet(problem, self.config.hidden_sizes, self.config.activation).to(self.config.device)

    def _new_value_net(self, problem: StochasticControlProblem) -> nn.Module:
        if problem.name == "OptionHedging":
            return HedgingValueNet(
                problem,
                self.config.hidden_sizes,
                self.config.activation,
            ).to(self.config.device)
    
        return ValueNet(
            problem,
            self.config.hidden_sizes,
            self.config.activation,
        ).to(self.config.device)

    def _maybe_copy(self, target: nn.Module, source: Optional[nn.Module]) -> None:
        if self.config.transfer_warm_start and source is not None:
            target.load_state_dict(copy.deepcopy(source.state_dict()))

    def _terminal_value(self, problem: StochasticControlProblem) -> Callable[[Tensor], Tensor]:
        return lambda x: problem.terminal_cost(x)

    def _freeze_solution_modules(self, modules: Sequence[Optional[nn.Module]]) -> None:
        for m in modules:
            if m is not None:
                freeze_module(m)


# ----------------------------
# Algorithm 1: NNContPI
# ----------------------------

class NNContPISolver(BaseNeuralSolver):
    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "continuous":
            raise ValueError("NNContPI requires continuous controls.")
        N = problem.horizon
        policy_nets: List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        logs = SolverLogs(meta={"method": "NNContPI", "config": dataclasses.asdict(self.config)})

        for n in _make_tqdm(reversed(range(N)), desc="NNContPI -> n", verbose=self.config.verbose, total=N):
            net = self._new_continuous_policy(problem)
            self._maybe_copy(net, policy_nets[n + 1] if n + 1 < N else None)

            future_nets = [m for m in policy_nets[n + 1:] if m is not None]
            self._freeze_solution_modules(future_nets)

            # Capture n, net, and the current snapshot of policy_nets by value
            def make_loss(step_n: int, current_net: nn.Module, p_nets: list) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    a = current_net(x)
                    total = problem.running_cost(x, a, step_n)
                    x = problem.dynamics(x, a, eps, step_n)
                    for k in range(step_n + 1, N):
                        fnet = p_nets[k]
                        assert fnet is not None
                        a = fnet(x)
                        total = total + problem.running_cost(x, a, k)
                        eps = problem.sample_noise(self.config.batch_size, self.config.device)
                        x = problem.dynamics(x, a, eps, k)
                    total = total + problem.terminal_cost(x)
                    return total.mean()
                return closure

            losses = self._train_loop(net, make_loss(n, net, policy_nets), desc=f"  n={n}")
            logs.policy_losses.insert(0, losses)
            freeze_module(net)
            policy_nets[n] = net
            policies[n] = lambda x, net=net: net(x)

        value_fns = [lambda x, problem=problem, policies=policies, n=n: policy_value_mc(problem, policies, x, start_n=n) for n in range(N)]
        return Solution(
            method="NNContPI",
            policies=policies,
            value_functions=value_fns,
            policy_nets=policy_nets,
            value_nets=[None] * N,
            logs=logs,
            problem_name=problem.name,
        )


# ----------------------------
# Algorithm 2: ClassifPI
# ----------------------------

class ClassifPISolver(BaseNeuralSolver):
    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "discrete":
            raise ValueError("ClassifPI requires a discrete-action problem.")
        N = problem.horizon
        action_values = to_tensor(problem.discrete_actions, device=self.config.device)
        L = action_values.shape[0]
        policy_nets: List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        logs = SolverLogs(meta={"method": "ClassifPI", "config": dataclasses.asdict(self.config)})

        for n in _make_tqdm(reversed(range(N)), desc="ClassifPI -> n", verbose=self.config.verbose, total=N):
            net = self._new_discrete_policy(problem)
            self._maybe_copy(net, policy_nets[n + 1] if n + 1 < N else None)
            future_nets = [m for m in policy_nets[n + 1:] if m is not None]
            self._freeze_solution_modules(future_nets)

            def make_loss(step_n: int, current_net: nn.Module, p_nets: list) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x0 = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    probs = current_net(x0)  # [B, L]
                    all_costs = []
                    for ell in range(L):
                        a = action_values[ell].view(1, -1).expand(self.config.batch_size, -1)
                        total = problem.running_cost(x0, a, step_n)
                        eps = problem.sample_noise(self.config.batch_size, self.config.device)
                        x = problem.dynamics(x0, a, eps, step_n)
                        for k in range(step_n + 1, N):
                            fnet = p_nets[k]
                            assert fnet is not None
                            a = fnet.greedy_action(x)
                            total = total + problem.running_cost(x, a, k)
                            eps = problem.sample_noise(self.config.batch_size, self.config.device)
                            x = problem.dynamics(x, a, eps, k)
                        total = total + problem.terminal_cost(x)
                        all_costs.append(total)
                    costs = torch.stack(all_costs, dim=1)  # [B, L]
                    return (probs * costs).sum(dim=1).mean()
                return closure

            losses = self._train_loop(net, make_loss(n, net, policy_nets), desc=f"  n={n}")
            logs.policy_losses.insert(0, losses)
            freeze_module(net)
            policy_nets[n] = net
            policies[n] = lambda x, net=net: net.greedy_action(x)

        value_fns = [lambda x, problem=problem, policies=policies, n=n: policy_value_mc(problem, policies, x, start_n=n) for n in range(N)]
        return Solution(
            method="ClassifPI",
            policies=policies,
            value_functions=value_fns,
            policy_nets=policy_nets,
            value_nets=[None] * N,
            logs=logs,
            problem_name=problem.name,
        )


# ----------------------------
# Algorithm 3: Hybrid-Now
# ----------------------------

class HybridNowSolver(BaseNeuralSolver):
    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "continuous":
            raise ValueError("Hybrid-Now requires continuous controls.")
        N = problem.horizon
        policy_nets: List[Optional[nn.Module]] = [None] * N
        value_nets:  List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        value_functions: List[Callable[[Tensor], Tensor]] = [self._terminal_value(problem) for _ in range(N)]
        V_next = self._terminal_value(problem)
        next_value_net = None
        logs = SolverLogs(meta={"method": "Hybrid-Now", "config": dataclasses.asdict(self.config)})

        for n in _make_tqdm(reversed(range(N)), desc="Hybrid-Now -> n", verbose=self.config.verbose, total=N):
            policy_net = self._new_continuous_policy(problem)
            self._maybe_copy(policy_net, policy_nets[n + 1] if n + 1 < N else None)
            if next_value_net is not None:
                freeze_module(next_value_net)

            # Capture V_next, policy_net, and n by value via factory arguments
            def make_policy_loss(pnet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    a = pnet(x)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    x_next = problem.dynamics(x, a, eps, step_n)
                    return (problem.running_cost(x, a, step_n) + Vn1(x_next)).mean()
                return closure

            policy_losses = self._train_loop(policy_net, make_policy_loss(policy_net, V_next, n), desc=f"  n={n}")
            logs.policy_losses.insert(0, policy_losses)
            freeze_module(policy_net)
            policy_nets[n] = policy_net
            policies[n] = lambda x, net=policy_net: net(x)

            value_net = self._new_value_net(problem)
            self._maybe_copy(value_net, next_value_net)

            def make_value_loss(pnet: nn.Module, vnet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    with torch.no_grad():
                        a = pnet(x)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    x_next = problem.dynamics(x, a, eps, step_n)
                    target = (problem.running_cost(x, a, step_n) + Vn1(x_next)).detach()
                    pred = vnet(x)
                    return ((target - pred) ** 2).mean()
                return closure

            value_losses = self._train_loop(value_net, make_value_loss(policy_net, value_net, V_next, n), desc=f"  n={n}")
            logs.value_losses.insert(0, value_losses)
            freeze_module(value_net)
            value_nets[n] = value_net
            value_functions[n] = lambda x, net=value_net: net(x)
            V_next = value_functions[n]
            next_value_net = value_net

        return Solution(
            method="Hybrid-Now",
            policies=policies,
            value_functions=value_functions,
            policy_nets=policy_nets,
            value_nets=value_nets,
            logs=logs,
            problem_name=problem.name,
        )



# ----------------------------
# Hybrid-Now / Hybrid-LaterQ variants for finite action spaces
# ----------------------------

class DiscreteHybridNowSolver(BaseNeuralSolver):
    """Hybrid-Now for a finite action set using direct finite-action greedification.

    The former V5 implementation trained a soft policy network and then used
    its greedy action.  For gas storage this could repeatedly withdraw and
    trigger the terminal inventory penalty, giving values around -20.  Since
    the action set is finite, the correct Hybrid-Now analogue is to learn the
    value function and choose the next action by direct minimisation over the
    three feasible actions.
    """

    def _q_values(
        self,
        problem: StochasticControlProblem,
        Vn1: Callable[[Tensor], Tensor],
        x: Tensor,
        step_n: int,
        n_noise: int = 3,
    ) -> Tensor:
        action_values = to_tensor(problem.discrete_actions, device=x.device, dtype=x.dtype)
        if action_values.ndim == 1:
            action_values = action_values[:, None]
        B = x.shape[0]
        viability_fn = getattr(problem, "terminal_viability_action_matrix_torch", None)
        mask_matrix_fn = getattr(problem, "admissible_action_matrix_torch", None)
        if viability_fn is not None:
            feasible = viability_fn(x, step_n)
        else:
            feasible = mask_matrix_fn(x) if mask_matrix_fn is not None else None

        # Optional economic regularisation. Gas storage is a low-dimensional
        # discrete-control problem where raw Hybrid-Now can massively
        # undervalue inventory and repeatedly withdraw gas. The GasStorageProblem
        # now provides a shadow-inventory baseline and a small smooth action
        # regulariser; both are absent for other examples.
        action_penalty_fn = getattr(problem, "hybrid_action_regularization_torch", None)

        cols: List[Tensor] = []
        m = max(1, int(n_noise))
        # Deterministic Gauss-Hermite integration for 1D Gaussian-noise examples.
        if getattr(problem, "noise_dim", None) == 1 and hasattr(problem, "sigma_p"):
            gh_x, gh_w = np.polynomial.hermite.hermgauss(m)
            eps_points = (math.sqrt(2.0) * float(getattr(problem, "sigma_p")) * gh_x).astype(np.float32)
            eps_weights = (gh_w / math.sqrt(math.pi)).astype(np.float32)
        elif getattr(problem, "noise_dim", None) == 1 and hasattr(problem, "sigma_R"):
            gh_x, gh_w = np.polynomial.hermite.hermgauss(m)
            eps_points = (math.sqrt(2.0) * float(getattr(problem, "sigma_R")) * gh_x).astype(np.float32)
            eps_weights = (gh_w / math.sqrt(math.pi)).astype(np.float32)
        else:
            eps_points = None
            eps_weights = np.full(m, 1.0 / float(m), dtype=np.float32)

        for ell in range(action_values.shape[0]):
            a = action_values[ell].view(1, -1).expand(B, -1)
            q = problem.running_cost(x, a, step_n)
            if action_penalty_fn is not None:
                q = q + action_penalty_fn(x, a, step_n)
            exp_val = torch.zeros_like(q)
            for j in range(m):
                if eps_points is None:
                    eps = problem.sample_noise(B, x.device)
                else:
                    eps = torch.full((B, 1), float(eps_points[j]), device=x.device, dtype=x.dtype)
                x_next = problem.dynamics(x, a, eps, step_n)
                next_val = Vn1(x_next)
                # Broad numerical guard only. Do not clip at -5: with the gas
                # inventory baseline, negative values are meaningful surplus-gas
                # continuation values.
                next_val = torch.nan_to_num(next_val, nan=1e6, posinf=1e6, neginf=-1e6)
                next_val = torch.clamp(next_val, min=-100.0, max=200.0)
                exp_val = exp_val + float(eps_weights[j]) * next_val
            q = q + exp_val
            if feasible is not None:
                q = torch.where(feasible[:, ell], q, torch.full_like(q, 1e8))
            cols.append(q)
        return torch.stack(cols, dim=1)

    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "discrete":
            raise ValueError("DiscreteHybridNowSolver requires a discrete-action problem.")
        N = problem.horizon
        value_nets: List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        value_functions: List[Callable[[Tensor], Tensor]] = [self._terminal_value(problem) for _ in range(N)]
        V_next = self._terminal_value(problem)
        next_value_net = None
        logs = SolverLogs(meta={"method": "DiscreteHybrid-Now", "config": dataclasses.asdict(self.config)})
        # Fast mode used to hard-code 9/15 Gauss-Hermite points, which made
        # gas storage Hybrid-Now extremely slow. The script now passes smaller
        # values for --fast and larger values for --no-fast.
        train_noise = int(self.config.hybrid_train_noise) if self.config.hybrid_train_noise is not None else (7 if self.config.epochs * self.config.n_batches_per_epoch >= 20 else 5)
        eval_noise = int(self.config.hybrid_eval_noise) if self.config.hybrid_eval_noise is not None else 11

        baseline_fn = getattr(problem, "hybrid_value_baseline_torch", None)

        def baseline(step_n: int, x: Tensor) -> Tensor:
            if baseline_fn is None:
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            return baseline_fn(step_n, x)

        def make_full_value(step_n: int, net: nn.Module) -> Callable[[Tensor], Tensor]:
            def value(x: Tensor) -> Tensor:
                return baseline(step_n, x) + net(x)
            return value

        for n in _make_tqdm(reversed(range(N)), desc="DiscreteHybrid-Now -> n", verbose=self.config.verbose, total=N):
            if next_value_net is not None:
                freeze_module(next_value_net)
            V_after = V_next
            value_net = self._new_value_net(problem)
            self._maybe_copy(value_net, next_value_net)

            def make_value_loss(vnet: nn.Module, Vn1: Callable[[Tensor], Tensor], step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    with torch.no_grad():
                        qvals = self._q_values(problem, Vn1, x, step_n, n_noise=train_noise)
                        target = torch.min(qvals, dim=1).values.detach()
                        target_residual = target - baseline(step_n, x)
                        target_residual = torch.clamp(target_residual, min=-50.0, max=50.0)
                    pred = vnet(x)
                    return ((target_residual - pred) ** 2).mean()
                return closure

            value_losses = self._train_loop(value_net, make_value_loss(value_net, V_after, n), desc=f"  n={n}")
            logs.value_losses.insert(0, value_losses)
            logs.policy_losses.insert(0, [])
            freeze_module(value_net)
            value_nets[n] = value_net
            value_functions[n] = make_full_value(n, value_net)

            def make_policy(step_n: int, Vn1: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
                def policy(x: Tensor) -> Tensor:
                    qvals = self._q_values(problem, Vn1, x, step_n, n_noise=eval_noise)
                    idx = torch.argmin(qvals, dim=1)
                    actions = to_tensor(problem.discrete_actions, device=x.device, dtype=x.dtype)
                    if actions.ndim == 1:
                        actions = actions[:, None]
                    return actions[idx]
                return policy

            policies[n] = make_policy(n, V_after)
            V_next = value_functions[n]
            next_value_net = value_net

        return Solution(
            method="Hybrid-Now",
            policies=policies,
            value_functions=value_functions,
            policy_nets=[None] * N,
            value_nets=value_nets,
            logs=logs,
            problem_name=problem.name,
        )
class DiscreteHybridLaterQSolver(BaseNeuralSolver):
    """Hybrid-LaterQ analogue for finite action problems."""

    def __init__(self, config: NeuralSolverConfig, quantizer: Quantizer) -> None:
        super().__init__(config)
        self.quantizer = quantizer

    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "discrete":
            raise ValueError("DiscreteHybridLaterQSolver requires a discrete-action problem.")
        N = problem.horizon
        action_values = to_tensor(problem.discrete_actions, device=self.config.device)
        L = action_values.shape[0]
        e_pts, e_w = self.quantizer.to_torch(self.config.device)

        policy_nets: List[Optional[nn.Module]] = [None] * N
        value_nets: List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        value_functions: List[Callable[[Tensor], Tensor]] = [self._terminal_value(problem) for _ in range(N)]
        V_next = self._terminal_value(problem)
        next_value_net = None
        logs = SolverLogs(meta={"method": "DiscreteHybrid-LaterQ", "config": dataclasses.asdict(self.config), "K": len(self.quantizer.weights)})

        for n in _make_tqdm(reversed(range(N)), desc="DiscreteHybrid-LaterQ → n", verbose=self.config.verbose, total=N):
            policy_net = self._new_discrete_policy(problem)
            self._maybe_copy(policy_net, policy_nets[n + 1] if n + 1 < N else None)
            if next_value_net is not None:
                freeze_module(next_value_net)

            def make_policy_loss(pnet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    probs = pnet(x)
                    cols = []
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    for ell in range(L):
                        a = action_values[ell].view(1, -1).expand(self.config.batch_size, -1)
                        x_next = problem.dynamics(x, a, eps, step_n)
                        cols.append(problem.running_cost(x, a, step_n) + Vn1(x_next))
                    costs = torch.stack(cols, dim=1)
                    return (probs * costs).sum(dim=1).mean()
                return closure

            policy_losses = self._train_loop(policy_net, make_policy_loss(policy_net, V_next, n), desc=f"  n={n}")
            logs.policy_losses.insert(0, policy_losses)
            freeze_module(policy_net)
            policy_nets[n] = policy_net
            policies[n] = lambda x, net=policy_net: net.greedy_action(x)

            interp_net = self._new_value_net(problem)
            self._maybe_copy(interp_net, next_value_net)

            def make_interp_loss(pnet: DiscretePolicyNet, inet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    with torch.no_grad():
                        a = pnet.greedy_action(x)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    x_next = problem.dynamics(x, a, eps, step_n)
                    target = Vn1(x_next).detach()
                    pred = inet(x_next)
                    return ((target - pred) ** 2).mean()
                return closure

            value_losses = self._train_loop(interp_net, make_interp_loss(policy_net, interp_net, V_next, n), desc=f"  n={n}")
            logs.value_losses.insert(0, value_losses)
            freeze_module(interp_net)
            value_nets[n] = interp_net

            def make_value_fn(step_n: int, pnet: DiscretePolicyNet, inet: nn.Module, ep: Tensor, ew: Tensor) -> Callable[[Tensor], Tensor]:
                def Vn(x: Tensor) -> Tensor:
                    a = pnet.greedy_action(x)
                    B, K = x.shape[0], ep.shape[0]
                    x_rep = x.unsqueeze(1).expand(B, K, -1).reshape(B * K, problem.state_dim)
                    a_rep = a.unsqueeze(1).expand(B, K, -1).reshape(B * K, problem.action_dim)
                    e_rep = ep.unsqueeze(0).expand(B, K, -1).reshape(B * K, ep.shape[-1])
                    x_next = problem.dynamics(x_rep, a_rep, e_rep, step_n)
                    vals = inet(x_next).reshape(B, K)
                    expected = (vals * ew.view(1, K)).sum(dim=1)
                    return problem.running_cost(x, a, step_n) + expected
                return Vn

            value_functions[n] = make_value_fn(n, policy_net, interp_net, e_pts, e_w)
            V_next = value_functions[n]
            next_value_net = interp_net

        return Solution(
            method="Hybrid-LaterQ",
            policies=policies,
            value_functions=value_functions,
            policy_nets=policy_nets,
            value_nets=value_nets,
            logs=logs,
            problem_name=problem.name,
        )


# ----------------------------
# Algorithm 4: Hybrid-LaterQ
# ----------------------------

class HybridLaterQSolver(BaseNeuralSolver):
    def __init__(self, config: NeuralSolverConfig, quantizer: Quantizer) -> None:
        super().__init__(config)
        self.quantizer = quantizer

    def solve(self, problem: StochasticControlProblem) -> Solution:
        if problem.action_mode != "continuous":
            raise ValueError("Hybrid-LaterQ requires continuous controls.")
        N = problem.horizon
        e_pts, e_w = self.quantizer.to_torch(self.config.device)
        policy_nets: List[Optional[nn.Module]] = [None] * N
        value_nets:  List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        value_functions: List[Callable[[Tensor], Tensor]] = [self._terminal_value(problem) for _ in range(N)]
        V_next = self._terminal_value(problem)
        next_value_net = None
        logs = SolverLogs(meta={"method": "Hybrid-LaterQ", "config": dataclasses.asdict(self.config), "K": len(self.quantizer.weights)})

        for n in _make_tqdm(reversed(range(N)), desc="Hybrid-LaterQ -> n", verbose=self.config.verbose, total=N):
            policy_net = self._new_continuous_policy(problem)
            self._maybe_copy(policy_net, policy_nets[n + 1] if n + 1 < N else None)
            if next_value_net is not None:
                freeze_module(next_value_net)

            def make_policy_loss(pnet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    a = pnet(x)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    x_next = problem.dynamics(x, a, eps, step_n)
                    return (problem.running_cost(x, a, step_n) + Vn1(x_next)).mean()
                return closure

            policy_losses = self._train_loop(policy_net, make_policy_loss(policy_net, V_next, n), desc=f"  n={n}")
            logs.policy_losses.insert(0, policy_losses)
            freeze_module(policy_net)
            policy_nets[n] = policy_net
            policies[n] = lambda x, net=policy_net: net(x)

            # Interpolation net: learn to approximate V_{n+1}(x_{n+1}).
            # We regress V_{n+1}(x_{n+1}) onto x_{n+1} directly.
            interp_net = self._new_value_net(problem)
            self._maybe_copy(interp_net, next_value_net)

            def make_interp_loss(pnet: nn.Module, inet: nn.Module, Vn1: Callable, step_n: int) -> Callable[[], Tensor]:
                def closure() -> Tensor:
                    x = problem.sample_training_states(step_n, self.config.batch_size, self.config.device)
                    with torch.no_grad():
                        a = pnet(x)
                    eps = problem.sample_noise(self.config.batch_size, self.config.device)
                    x_next = problem.dynamics(x, a, eps, step_n)
                    target = Vn1(x_next).detach()
                    pred = inet(x_next)
                    return ((target - pred) ** 2).mean()
                return closure

            value_losses = self._train_loop(interp_net, make_interp_loss(policy_net, interp_net, V_next, n), desc=f"  n={n}")
            logs.value_losses.insert(0, value_losses)
            freeze_module(interp_net)
            value_nets[n] = interp_net

            def make_value_fn(
                step_n: int,
                pnet: nn.Module,
                inet: nn.Module,
                ep: Tensor,
                ew: Tensor,
            ) -> Callable[[Tensor], Tensor]:
                def Vn(x: Tensor) -> Tensor:
                    a = pnet(x)
                    B, K = x.shape[0], ep.shape[0]
                    x_rep = x.unsqueeze(1).expand(B, K, -1).reshape(B * K, problem.state_dim)
                    a_rep = a.unsqueeze(1).expand(B, K, -1).reshape(B * K, problem.action_dim)
                    # ep has shape (K, quantizer.dim); use ep.shape[-1], not problem.noise_dim,
                    # so this works even when noise_dim != quantizer.dim (e.g. LQ with D_list).
                    e_rep = ep.unsqueeze(0).expand(B, K, -1).reshape(B * K, ep.shape[-1])
                    x_next = problem.dynamics(x_rep, a_rep, e_rep, step_n)
                    vals = inet(x_next).reshape(B, K)
                    expected = (vals * ew.view(1, K)).sum(dim=1)
                    return problem.running_cost(x, a, step_n) + expected
                return Vn

            value_functions[n] = make_value_fn(n, policy_net, interp_net, e_pts, e_w)
            V_next = value_functions[n]
            next_value_net = interp_net

        return Solution(
            method="Hybrid-LaterQ",
            policies=policies,
            value_functions=value_functions,
            policy_nets=policy_nets,
            value_nets=value_nets,
            logs=logs,
            problem_name=problem.name,
        )


# ----------------------------
# Algorithm 5: Qknn
# ----------------------------

@dataclass
class QknnConfig:
    action_candidates: Optional[np.ndarray] = None
    state_grids: Optional[List[np.ndarray]] = None
    k_neighbors: int = 1
    device: str = field(default_factory=default_device)
    verbose: bool = True


class QknnSolver:
    def __init__(self, config: QknnConfig, quantizer: Quantizer) -> None:
        self.config = config
        self.quantizer = quantizer

    def _interp_knn(
        self,
        grid: np.ndarray,
        values: np.ndarray,
        xq: np.ndarray,
        k: int = 1,
        tree: Optional[cKDTree] = None,
    ) -> np.ndarray:
        """k-nearest-neighbour interpolation with optional cached KD-tree."""
        grid = np.asarray(grid, dtype=float)
        values = np.asarray(values, dtype=float)
        xq = np.asarray(xq, dtype=float)
        if grid.ndim == 1:
            grid = grid[:, None]
        if xq.ndim == 1:
            xq = xq[:, None]

        k_eff = min(int(k), grid.shape[0])
        if tree is None:
            tree = cKDTree(grid)
        dists, idx = tree.query(xq, k=k_eff)
        if k_eff == 1:
            return values[np.asarray(idx, dtype=int)]

        idx = np.asarray(idx, dtype=int)
        dists = np.asarray(dists, dtype=float)
        picked_vals = values[idx]
        weights = 1.0 / np.maximum(dists, 1e-8)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return (weights * picked_vals).sum(axis=1)

    def solve(self, problem: StochasticControlProblem) -> Solution:
        if self.config.state_grids is None:
            raise ValueError("Provide state_grids for Qknn.")
        if self.config.action_candidates is None:
            self.config.action_candidates = problem.action_grid()

        state_grids = [np.asarray(g, dtype=np.float32) for g in self.config.state_grids]
        actions = np.asarray(self.config.action_candidates, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]

        e_pts = np.asarray(self.quantizer.points, dtype=np.float32)
        e_w = np.asarray(self.quantizer.weights, dtype=np.float32)

        N = problem.horizon
        V_grid: List[Optional[np.ndarray]] = [None] * (N + 1)
        A_grid: List[Optional[np.ndarray]] = [None] * N
        Q_grid: List[Optional[np.ndarray]] = [None] * N
        # Cache one KD-tree per state grid.  The old implementation rebuilt the
        # tree inside every (time, action, noise) loop.
        trees = [
            cKDTree(np.asarray(g, dtype=float) if np.asarray(g).ndim > 1 else np.asarray(g, dtype=float)[:, None])
            for g in state_grids
        ]

        terminal_grid = state_grids[-1]
        xN = to_tensor(terminal_grid, device=self.config.device)
        if xN.ndim == 1:
            xN = xN[:, None]
        V_grid[N] = problem.terminal_cost(xN).detach().cpu().numpy().astype(np.float32)

        mask_fn = getattr(problem, "admissible_action_mask_np", None)

        for n in _make_tqdm(
            reversed(range(N)),
            desc="Qknn → n",
            verbose=self.config.verbose,
            total=N,
        ):
            grid_n = state_grids[n]
            grid_np1 = state_grids[n + 1]
            tree_np1 = trees[n + 1]
            V_np1 = V_grid[n + 1]
            assert V_np1 is not None

            G = len(grid_n)
            qvals = np.full((G, len(actions)), np.inf, dtype=np.float32)
            z_t = to_tensor(grid_n, device=self.config.device)
            if z_t.ndim == 1:
                z_t = z_t[:, None]

            penalized_cols: List[np.ndarray] = []
            for j, a in enumerate(actions):
                a_batch = np.repeat(a.reshape(1, -1), G, axis=0).astype(np.float32)
                a_t = to_tensor(a_batch, device=self.config.device)
                running = problem.running_cost(z_t, a_t, n).detach().cpu().numpy().astype(np.float32)

                expected = np.zeros(G, dtype=np.float32)
                for ell in range(len(e_w)):
                    e_batch = np.repeat(e_pts[ell].reshape(1, -1), G, axis=0).astype(np.float32)
                    e_t = to_tensor(e_batch, device=self.config.device)
                    xq = problem.dynamics(z_t, a_t, e_t, n).detach().cpu().numpy()
                    expected += float(e_w[ell]) * self._interp_knn(
                        grid_np1, V_np1, xq, k=self.config.k_neighbors, tree=tree_np1
                    ).astype(np.float32)

                col = running + expected
                penalized_cols.append(col.copy())
                if mask_fn is not None:
                    admissible = np.asarray(mask_fn(grid_n, a_batch), dtype=bool).reshape(-1)
                    col[~admissible] = np.inf
                qvals[:, j] = col

            no_feasible = ~np.isfinite(qvals).any(axis=1)
            if no_feasible.any():
                penalized = np.stack(penalized_cols, axis=1)
                qvals[no_feasible] = penalized[no_feasible]

            best_idx = np.nanargmin(qvals, axis=1)
            A_grid[n] = actions[best_idx].astype(np.float32)
            V_grid[n] = qvals[np.arange(G), best_idx].astype(np.float32)
            Q_grid[n] = qvals.copy()

        def make_policy(n: int) -> Callable[[Tensor], Tensor]:
            grid_n = np.asarray(state_grids[n], dtype=np.float32)
            q_n = np.asarray(Q_grid[n], dtype=np.float32)
            a_grid_n = np.asarray(A_grid[n], dtype=np.float32)
            acts_n = np.asarray(actions, dtype=np.float32)
            tree_n = trees[n]

            def policy(x: Tensor) -> Tensor:
                x_np = x.detach().cpu().numpy()

                q_interp_cols = []
                for j in range(q_n.shape[1]):
                    qj = self._interp_knn(
                        grid_n, q_n[:, j], x_np, k=self.config.k_neighbors, tree=tree_n
                    )
                    q_interp_cols.append(qj.reshape(-1, 1))
                q_interp = np.concatenate(q_interp_cols, axis=1)

                finite_q = np.where(np.isfinite(q_interp), q_interp, np.inf)
                mask_fn = getattr(problem, "admissible_action_mask_np", None)
                if mask_fn is not None:
                    feasible_cols = []
                    for a in acts_n:
                        a_batch = np.repeat(a.reshape(1, -1), x_np.shape[0], axis=0).astype(np.float32)
                        feasible_cols.append(np.asarray(mask_fn(x_np, a_batch), dtype=bool).reshape(-1, 1))
                    feasible = np.concatenate(feasible_cols, axis=1)
                    finite_q = np.where(feasible, finite_q, np.inf)
                all_bad = ~np.isfinite(finite_q).any(axis=1)
                best_idx = np.argmin(finite_q, axis=1)
                out = acts_n[best_idx].copy()
                if all_bad.any():
                    _, idx_nn = tree_n.query(x_np[all_bad], k=1)
                    out[all_bad] = a_grid_n[np.asarray(idx_nn, dtype=int)]
                project_np = getattr(problem, "project_action_np", None)
                if project_np is not None:
                    out = np.asarray(project_np(x_np, out), dtype=np.float32)
                    if out.ndim == 1:
                        out = out[:, None]
                return to_tensor(out, device=x.device, dtype=x.dtype)

            return policy

        def make_value(n: int) -> Callable[[Tensor], Tensor]:
            grid_n = np.asarray(state_grids[n], dtype=np.float32)
            vals_n = np.asarray(V_grid[n], dtype=np.float32)
            tree_n = trees[n]

            def value(x: Tensor) -> Tensor:
                x_np = x.detach().cpu().numpy()
                out = self._interp_knn(grid_n, vals_n, x_np, k=self.config.k_neighbors, tree=tree_n)
                return to_tensor(out, device=x.device, dtype=x.dtype)

            return value

        policies = [make_policy(n) for n in range(N)]
        value_functions = [make_value(n) for n in range(N)]

        return Solution(
            method="Qknn",
            policies=policies,
            value_functions=value_functions,
            policy_nets=[None] * N,
            value_nets=[None] * N,
            logs=SolverLogs(meta={
                "method": "Qknn",
                "k_neighbors": self.config.k_neighbors,
                "state_grids": state_grids,
                "actions": actions,
                "A_grid": A_grid,
                "Q_grid": Q_grid,
                "V_grid": V_grid,
            }),
            problem_name=problem.name,
        )


# ----------------------------
# Monte Carlo policy value helper
# ----------------------------

def policy_value_mc(
    problem: StochasticControlProblem,
    policies_or_solution: Union[Sequence[Callable[[Tensor], Tensor]], Solution],
    x: Optional[Tensor] = None,
    start_n: int = 0,
    n_mc: int = 256,
    *,
    x0: Optional[Union[np.ndarray, Tensor]] = None,
    n_paths: Optional[int] = None,
    seed: int = 1234,
    device: Optional[str] = None,
    verbose: bool = False,
) -> Union[Tensor, Dict[str, Any]]:
    """
    Two modes for backward compatibility:

    1) Internal low-level mode used by value functions:
       policy_value_mc(problem, policies, x=<torch tensor>, start_n=0, n_mc=256)
       -> returns a torch Tensor of shape [batch].

    2) Notebook / experiment mode:
       policy_value_mc(problem, solution, x0=<numpy or tensor>, n_paths=..., seed=..., device=...)
       -> returns a dict with mean/std/costs/trajectory.
    """
    # High-level notebook mode
    if isinstance(policies_or_solution, Solution):
        solution = policies_or_solution
        if x0 is None and x is None:
            raise ValueError("Provide x0 (or x) when calling policy_value_mc with a Solution.")
        x0_use = x0 if x0 is not None else x
        return rollout_policy(
            problem,
            solution,
            x0=x0_use,
            n_paths=int(n_paths or n_mc),
            seed=seed,
            device=device,
            verbose=verbose,
        )

    # Low-level internal mode
    policies = policies_or_solution
    if x is None:
        if x0 is None:
            raise ValueError("Provide x (Tensor) in low-level mode, or x0 in high-level mode.")
        x = to_tensor(x0, device=device)
    device = x.device
    B = x.shape[0]
    x_rep = x.repeat_interleave(int(n_mc), dim=0)
    total = torch.zeros(B * int(n_mc), device=device)
    for n in range(start_n, problem.horizon):
        a = policies[n](x_rep)
        total = total + problem.running_cost(x_rep, a, n)
        eps = problem.sample_noise(B * int(n_mc), device=device)
        x_rep = problem.dynamics(x_rep, a, eps, n)
    total = total + problem.terminal_cost(x_rep)
    return total.view(B, int(n_mc)).mean(dim=1)


# ----------------------------
# High-level experiment helpers
# ----------------------------

def make_test1_semilinear_problem(d: int = 100, T: float = 1.0, N: int = 20) -> SemilinearPDEProblem:
    def g(x: Tensor) -> Tensor:
        return torch.log(0.5 * (1.0 + (x ** 2).sum(dim=1)))
    return SemilinearPDEProblem(
        state_dim=d,
        action_dim=d,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=None,
        terminal_cost_fn=g,
        name=f"SemilinearPDE-Test1-d={d}",
    )


def make_test2_semilinear_problem(
    gamma: float,
    T: float = 1.0,
    N: int = 40,
    action_bounds: Tuple[float, float] = (-3.0, 3.0),
    sigma_floor: float = 0.35,
    train_domain: Optional[Tuple[float, float]] = None,
) -> SemilinearPDEProblem:
    def g(x: Tensor) -> Tensor:
        x1 = x[:, 0]
        out = torch.zeros_like(x1)
        mask1 = (x1 >= 0.0) & (x1 <= 1.0)
        mask2 = x1 >= 1.0
        if gamma == 0.0:
            out[mask1] = -1.0
        else:
            out[mask1] = -(torch.clamp(x1[mask1], min=1e-12) ** gamma)
        out[mask2] = -1.0
        return out
    def sampler(n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        if train_domain is not None:
            lo, hi = train_domain
            u = torch.rand(batch_size, 1, device=device)
            return lo + (hi - lo) * u
        sigma = max(float(sigma_floor), math.sqrt(max(n * T / N, 1e-8)))
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
        name=f"SemilinearPDE-Test2-gamma={gamma}",
    )


def make_discrete_test2_problem(
    gamma: float,
    action_values: np.ndarray,
    T: float = 1.0,
    N: int = 40,
) -> SemilinearPDEProblem:
    base = make_test2_semilinear_problem(gamma=gamma, T=T, N=N, action_bounds=None)
    base.action_mode = "discrete"
    av = np.asarray(action_values, dtype=np.float32)
    if av.ndim == 1:
        av = av[:, None]
    base.discrete_actions = av
    base.action_dim = av.shape[1]
    return base


def make_lq_problem(d: int = 10, T: float = 1.0, N: int = 20) -> LinearQuadraticProblem:
    B = np.eye(d, dtype=np.float32)
    C = np.ones((d, 1), dtype=np.float32)
    D_list = []
    for j in range(d):
        Dj = np.zeros((d, 1), dtype=np.float32)
        Dj[j, 0] = 1.0
        D_list.append(Dj)
    Q = np.eye(d, dtype=np.float32)
    P = np.eye(d, dtype=np.float32)
    return LinearQuadraticProblem(
        state_dim=d,
        action_dim=1,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=(-6.0, 6.0),
        B=B,
        C=C,
        D_list=D_list,
        Q=Q,
        P=P,
        lam=1.0,
        name=f"LQ-d={d}",
    )


def make_state_grids_from_sampler(problem: StochasticControlProblem, n_points: int = 200, seed: int = 1234) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    grids = []
    for n in range(problem.horizon + 1):
        if hasattr(problem, "sample_training_states"):
            x = problem.sample_training_states(min(n, problem.horizon - 1), n_points, device="cpu").detach().cpu().numpy()
        else:
            x = rng.normal(size=(n_points, problem.state_dim)).astype(np.float32)
        x = np.asarray(x, dtype=np.float32)
        if problem.state_dim == 1:
            x = np.sort(x.reshape(-1))
        grids.append(x)
    return grids


# ============================================================
# ADDITIONS: missing factories, exact solvers, extra problems
# ============================================================

# ----------------------------
# semilinear_closed_form_mc: extended signature (n_paths alias, d unused)
# ----------------------------

def _semilinear_closed_form_mc_v2(
    g: Callable[[Tensor], Tensor],
    x: Union[np.ndarray, Tensor],
    t: float,
    T: float,
    n_mc: int = 100000,
    seed: int = 1234,
    device: Optional[str] = None,
    *,
    n_paths: Optional[int] = None,
    d: Optional[int] = None,   # ignored ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“ kept for notebook compatibility
) -> np.ndarray:
    """v(t,x) = -log E[exp(-g(x + sqrt(2) W_{T-t}))].  Alias: n_paths=n_mc."""
    return semilinear_closed_form_mc(g, x, t, T, n_mc=n_paths if n_paths is not None else n_mc, seed=seed, device=device)


# Monkey-patch the public name so existing calls work.
semilinear_closed_form_mc.__wrapped__ = semilinear_closed_form_mc  # keep original
import functools as _functools

@_functools.wraps(semilinear_closed_form_mc)
def semilinear_closed_form_mc(  # noqa: F811
    g: Callable[[Tensor], Tensor],
    x: Union[np.ndarray, Tensor],
    t: float = 0.0,
    T: float = 1.0,
    n_mc: int = 100000,
    seed: int = 1234,
    device: Optional[str] = None,
    *,
    n_paths: Optional[int] = None,
    d: Optional[int] = None,
) -> np.ndarray:
    """
    Closed-form Monte Carlo value for the semilinear PDE:
        v(t,x) = -log E[exp(-g(x + sqrt(2) W_{T-t}))]
    Parameters
    ----------
    g         : terminal cost, accepts a (B, d) Tensor, returns (B,) Tensor.
    x         : evaluation point(s), shape (d,) or (B, d).
    t         : current time.
    T         : terminal time.
    n_mc      : number of Monte Carlo samples (alias: n_paths).
    """
    n_use = n_paths if n_paths is not None else n_mc
    set_seed(seed)
    dev = device or default_device()
    x_t = to_tensor(x, device=dev)
    if x_t.ndim == 1:
        x_t = x_t[None, :]
    d_dim = x_t.shape[1]
    z = torch.randn(n_use, d_dim, device=dev) * math.sqrt(max(T - t, 1e-12))
    vals = []
    batch = 4096
    for i in range(0, n_use, batch):
        noise = z[i : i + batch]
        shifted = x_t[:, None, :] + math.sqrt(2.0) * noise[None, :, :]
        shifted = shifted.reshape(-1, d_dim)
        g_vals = g(shifted).reshape(x_t.shape[0], -1)
        vals.append(torch.exp(-g_vals))
    expo = torch.cat(vals, dim=1).mean(dim=1)
    v = -torch.log(expo)
    return v.detach().cpu().numpy()


# ----------------------------
# solve_discrete_lq_exact  (discrete-time Riccati, wraps solve_lq_riccati)
# ----------------------------

def solve_discrete_lq_exact(problem: "LinearQuadraticProblem") -> Dict[str, Any]:
    """
    Solve the LQ problem exactly using a discrete-time Riccati recursion.
    Returns a dict with keys: K (list of matrices, K[n] at step n),
    gains (list of feedback gain matrices), value (callable), policy (callable),
    and 'solution' (a Solution object that can be used like a learned solution).
    """
    d = problem.state_dim
    m = problem.action_dim
    h = problem.dt
    B = np.asarray(problem.B, dtype=float)
    C = np.asarray(problem.C, dtype=float)
    Q = np.asarray(problem.Q, dtype=float)
    P_mat = np.asarray(problem.P, dtype=float)
    D_list = [np.asarray(D, dtype=float) for D in problem.D_list]
    lamI = problem.lam * np.eye(m)

    N = problem.horizon
    # Backward Riccati: K[N] = P, then K[n] = Q*h + A'K[n+1]A - ...
    # Continuous-time matrices are embedded in the Euler discretisation:
    # x_{n+1} = (I + h*B)x + h*C*a + sqrt(h)*D*a*eps
    A_d = np.eye(d) + h * B
    C_d = h * C
    D_d = [math.sqrt(h) * D for D in D_list]

    K = [None] * (N + 1)
    K[N] = P_mat.copy()
    gains = [None] * N

    for n in reversed(range(N)):
        Kn1 = K[n + 1]
        # R = lam*I + sum_j D_j' K_{n+1} D_j
        R = h * lamI + C_d.T @ Kn1 @ C_d
        for Dj in D_d:
            R = R + Dj.T @ Kn1 @ Dj
        Rinv = np.linalg.inv(R)
        # Feedback gain G = Rinv * C_d' * K_{n+1} * A_d
        G = Rinv @ C_d.T @ Kn1 @ A_d
        gains[n] = G
        # Riccati update
        K[n] = h * Q + A_d.T @ Kn1 @ A_d - A_d.T @ Kn1 @ C_d @ Rinv @ C_d.T @ Kn1 @ A_d

    def value_fn(n: int, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        Kn = K[n]
        return np.einsum("bi,ij,bj->b", x, Kn, x)

    def policy_fn(n: int, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        return -(x @ gains[n].T)

    # Build a Solution object for seamless rollout_policy usage.
    # G and Kn are stored as numpy arrays and moved to x.device at call time
    # to avoid device-mismatch errors when x is on CUDA.

    def make_pol(n: int) -> Callable[[Tensor], Tensor]:
        G_np = gains[n].astype(np.float32)
        def pol(x: Tensor) -> Tensor:
            G = to_tensor(G_np, device=x.device)
            return -(x @ G.T)
        return pol

    def make_val(n: int) -> Callable[[Tensor], Tensor]:
        K_np = K[n].astype(np.float32)
        def val(x: Tensor) -> Tensor:
            Kn = to_tensor(K_np, device=x.device)
            return (x * (x @ Kn.T)).sum(dim=1)
        return val

    policies = [make_pol(n) for n in range(N)]
    value_functions = [make_val(n) for n in range(N)]
    sol = Solution(
        method="Riccati-exact",
        policies=policies,
        value_functions=value_functions,
        policy_nets=[None] * N,
        value_nets=[None] * N,
        logs=SolverLogs(meta={"method": "Riccati-exact"}),
        problem_name=problem.name,
    )

    return {
        "K": K,
        "P": K,           # alias (notebook uses exact_di['P'])
        "gains": gains,
        "value": value_fn,
        "policy": policy_fn,
        "solution": sol,
    }


# ----------------------------
# Double integrator LQ problem
# ----------------------------

def make_double_integrator_problem(N: int = 30, T: float = 1.0, lam: float = 0.1) -> "LinearQuadraticProblem":
    """
    2D double integrator: state x=(position, velocity), control a=force.
    Dynamics: x1' = x1 + h*x2, x2' = x2 + h*a + sqrt(h)*eps.
    Cost: integral of (x1^2 + lam*a^2) dt + x.T P x.
    """
    # Encode as LinearQuadraticProblem with B, C, D_list.
    # x_{n+1} = (I + h*B)x + h*C*a + sqrt(h)*D*a*eps
    # B = [[0,1],[0,0]], C = [[0],[1]], D = [[0],[1]] (noise only on velocity)
    B = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    C = np.array([[0.0], [1.0]], dtype=np.float32)
    D_list = [np.array([[0.0], [1.0]], dtype=np.float32)]
    Q = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    return LinearQuadraticProblem(
        state_dim=2,
        action_dim=1,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=(-5.0, 5.0),
        B=B,
        C=C,
        D_list=D_list,
        Q=Q,
        P=P,
        lam=lam,
        name="DoubleIntegratorLQ",
    )


# ----------------------------
# Option hedging problem (paper Section 3.3)
# ----------------------------

@dataclass
class OptionHedgingProblem(StochasticControlProblem):
    """
    State: x = (W, P) where W = wealth, P = stock price.
    Control: a = number of shares held (continuous, 1D).
    Returns are trinomial: r+ / 0 / r-.
    Cost: E[(h(P_N) - W_N)^2]  (no running cost).
    """
    p0: float = 100.0
    strike: float = 100.0
    r_plus: float = 0.05
    r_minus: float = -0.05
    pi_plus: float = 0.6
    pi_zero: float = 0.1
    pi_minus: float = 0.3

    def __post_init__(self) -> None:
        self.noise_dim = 1  # a single return draw; set before super() so it is not overwritten
        
        super().__post_init__()

    @property
    def returns(self) -> List[float]:
        return [self.r_plus, 0.0, self.r_minus]

    @property
    def return_probs(self) -> List[float]:
        return [self.pi_plus, self.pi_zero, self.pi_minus]

    def sample_noise(self, batch_size: int, device: Optional[str] = None) -> Tensor:
        # Sample from the trinomial return distribution.
        probs = torch.tensor([self.pi_plus, self.pi_zero, self.pi_minus], device=device)
        r_vals = torch.tensor([self.r_plus, 0.0, self.r_minus], device=device)
        idx = torch.multinomial(probs.expand(batch_size, -1), num_samples=1).squeeze(1)
        return r_vals[idx].unsqueeze(1)

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        # Price evolves around p0; wealth around the hedging cost (~a few units).
        p = self.p0 * torch.exp(0.15 * torch.randn(batch_size, 1, device=device))
        p = torch.clamp(p, self.p0 * 0.5, self.p0 * 2.0)
        # Wealth centered around the ATM call price (~5 for p0=100, strike=100)
        w = self.p0 * 0.05 + self.p0 * 0.1 * torch.randn(batch_size, 1, device=device)
        return torch.cat([w, p], dim=1)

    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        w = x[:, 0:1]
        p = x[:, 1:2]
        r = eps  # shape (B,1) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â return rate
        # a is the number of shares (delta); wealth change = a * p * r
        w_new = w + a * r
        p_new = p * (1.0 + r)
        p_new = torch.clamp(p_new, 1.0, self.p0 * 5.0)
        return torch.cat([w_new, p_new], dim=1)

    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device)

    def terminal_cost(self, x: Tensor) -> Tensor:
        w = x[:, 0]
        p = x[:, 1]
        payoff = torch.clamp(p - self.strike, min=0.0)
        return (payoff - w) ** 2


def make_option_hedging_problem(
    N: int = 6,
    p0: float = 100.0,
    strike: float = 100.0,
    r_plus: float = 0.05,
    r_minus: float = -0.05,
    pi_plus: float = 0.6,
    pi_minus: float = 0.3,
) -> OptionHedgingProblem:
    return OptionHedgingProblem(
        state_dim=2,
        action_dim=1,
        horizon=N,
        T=float(N),
        action_mode="continuous",
        # Delta for a call option is in [0, 1]; allow slight overshoot for learning.
        action_bounds=None,
        discrete_actions=None,
        p0=p0,
        strike=strike,
        r_plus=r_plus,
        r_minus=r_minus,
        pi_plus=pi_plus,
        pi_zero=1.0 - pi_plus - pi_minus,
        pi_minus=pi_minus,
        name="OptionHedging",
    )


def solve_option_hedging_exact(
    N: int = 6,
    p0: float = 100.0,
    strike: float = 100.0,
    r_plus: float = 0.05,
    r_minus: float = -0.05,
    pi_plus: float = 0.6,
    pi_minus: float = 0.3,
) -> Dict[str, Any]:
    """
    Exact DP solution for the trinomial option hedging problem.
    Returns dict with keys: hedging_price, value (callable), policy (callable).
    """
    pi_zero = 1.0 - pi_plus - pi_minus
    returns = np.array([r_plus, 0.0, r_minus], dtype=float)
    probs = np.array([pi_plus, pi_zero, pi_minus], dtype=float)
    nu_bar = (probs * returns).sum()
    M2 = (probs * returns ** 2).sum()

    # Terminal: K_N=1, Z_N(p) = payoff, C_N(p) = payoff^2
    # We discretise on a price tree for compactness.
    def payoff(p: np.ndarray) -> np.ndarray:
        return np.maximum(p - strike, 0.0)

    # Closed-form backward recursion via the BKL formula.
    # K_n = K_{n+1}*(1 - nu_bar^2/M2) -- scalar
    # Z_n(p) = E[Z_{n+1}(p*(1+R))] - nu_bar/M2 * E[Z_{n+1}(p*(1+R))*R]
    # We represent Z_n and C_n as functions evaluated by polynomial approximation
    # on a large grid of prices.

    p_grid = np.linspace(50.0, 200.0, 501)

    K = np.ones(N + 1)
    for n in reversed(range(N)):
        K[n] = K[n + 1] * (1.0 - nu_bar ** 2 / M2)

    Z = [None] * (N + 1)
    Cval = [None] * (N + 1)
    EZR_cache = [None] * N
    
    Z[N] = payoff(p_grid).copy()
    Cval[N] = payoff(p_grid) ** 2
    
    for n in reversed(range(N)):
        EZ = np.zeros_like(p_grid)
        EZR = np.zeros_like(p_grid)
        EC = np.zeros_like(p_grid)
    
        for r, prob in zip(returns, probs):
            p_next = p_grid * (1.0 + r)
            z_next = np.interp(p_next, p_grid, Z[n + 1])
            c_next = np.interp(p_next, p_grid, Cval[n + 1])
    
            EZ += prob * z_next
            EZR += prob * z_next * r
            EC += prob * c_next
    
        Z[n] = EZ - (nu_bar / M2) * EZR
        Cval[n] = EC - (EZR ** 2) / (K[n + 1] * M2)
        EZR_cache[n] = EZR
    
    # Hedging price = Z_0(p0) / K_0
    z0 = float(np.interp(p0, p_grid, Z[0]))
    hedging_price = z0 / K[0]
    
    def value(n: int, w: np.ndarray, p: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        p = np.asarray(p, dtype=float)
        Kn = K[n]
        Zn = np.interp(p, p_grid, Z[n])
        Cn = np.interp(p, p_grid, Cval[n])
        return Kn * w ** 2 - 2.0 * Zn * w + Cn
    
    def policy(n: int, w: np.ndarray, p: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        p = np.asarray(p, dtype=float)
        EZR = np.interp(p, p_grid, EZR_cache[n])
        return EZR / (K[n + 1] * M2) - (nu_bar / M2) * w

    

    return {
        "hedging_price": hedging_price,
        "value": value,
        "policy": policy,
        "K": K,
        "Z_grid": Z,
        "p_grid": p_grid,
        "C_grid": Cval
    }
class HedgingPolicyNet(nn.Module):
    """
    Paper ansatz:
        a(w, p) = b0(p) + b1(p) * w
    """
    def __init__(
        self,
        problem: StochasticControlProblem,
        hidden_sizes: Sequence[int],
        activation: str = "elu",
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.problem = problem
        self.body = MLP(1, 2, hidden_sizes, activation, normalize_input)

    def forward(self, x: Tensor) -> Tensor:
        w = x[:, 0:1]
        p = x[:, 1:2]

        coeff = self.body(p)
        a = coeff[:, 0:1] + coeff[:, 1:2] * w

        if self.problem.action_bounds is not None:
            low, high = self.problem.action_bounds
            low_t = to_tensor(low, device=x.device).view(1, -1)
            high_t = to_tensor(high, device=x.device).view(1, -1)
            a = torch.clamp(a, low_t, high_t)

        return a


class HedgingValueNet(nn.Module):
    """
    Paper ansatz:
        V(w, p) = c0(p) + c1(p) * w + c2(p) * w^2
    """
    def __init__(
        self,
        problem: StochasticControlProblem,
        hidden_sizes: Sequence[int],
        activation: str = "elu",
        normalize_input: bool = True,
    ) -> None:
        super().__init__()
        self.body = MLP(1, 3, hidden_sizes, activation, normalize_input)

    def forward(self, x: Tensor) -> Tensor:
        w = x[:, 0]
        p = x[:, 1:2]

        coeff = self.body(p)
        c0 = coeff[:, 0]
        c1 = coeff[:, 1]
        c2 = torch.nn.functional.softplus(coeff[:, 2]) + 1e-6

        return c0 + c1 * w + c2 * w ** 2

# ----------------------------
# Gas storage problem (paper Section 3.4)
# ----------------------------

@dataclass
class GasStorageProblem(StochasticControlProblem):
    """
    State: (P, C) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â gas price (mean-reverting) and inventory levelaction.
    Control: discrete {-1 (inject), 0 (hold), +1 (withdraw)}.
    Objective: MAXIMISE total profit ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬ÃƒÂ¢Ã¢â‚¬Å¾Ã‚Â¢ we minimise the negated profit.
    """
    ain: float = 0.06
    aout: float = 0.25
    bin_rate: float = 0.06
    bout_rate: float = 0.25
    C_min: float = 0.0
    C_max: float = 8.0
    c0: float = 4.0
    p0: float = 4.0
    p_bar: float = 5.0
    beta: float = 0.5
    sigma_p: float = math.sqrt(0.05)
    mu_penalty: float = 2.0
    Ki: float = 0.01  # storage cost coefficient

    def __post_init__(self) -> None:
        self.noise_dim = 1
        super().__post_init__()

    def sample_noise(self, batch_size: int, device: Optional[str] = None) -> Tensor:
        return self.sigma_p * torch.randn(batch_size, 1, device=device)

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        # Mixture distribution for the neural solvers.  The original marginal
        # law is essential around the actually visited states, but using only
        # this law gives very poor extrapolation on the decision maps.  The
        # uniform component makes the value residual well-trained on the full
        # (P,C) plotting/DP region.
        mean_n = self.p_bar + (self.beta ** n) * (self.p0 - self.p_bar)
        if n == 0:
            std_n = 1e-3
        else:
            var_n = self.sigma_p ** 2 * (1.0 - self.beta ** (2 * n)) / max(1.0 - self.beta ** 2, 1e-8)
            std_n = math.sqrt(max(var_n, 1e-8))

        m_marginal = int(0.60 * batch_size)
        m_uniform = batch_size - m_marginal

        P_m = mean_n + std_n * torch.randn(m_marginal, 1, device=device)
        P_m = torch.clamp(P_m, 2.5, 7.5)
        P_u = 3.0 + 4.0 * torch.rand(m_uniform, 1, device=device)
        P = torch.cat([P_m, P_u], dim=0)

        C = self.C_min + (self.C_max - self.C_min) * torch.rand(batch_size, 1, device=device)
        return torch.cat([P, C], dim=1)

    def hybrid_value_baseline_torch(self, n: int, x: Tensor) -> Tensor:
        """Economic baseline for the finite-action Hybrid-Now value net.

        The raw Hybrid-Now value net was the source of the very negative values
        seen in V5/V7: it undervalued inventory, so the greedy policy withdrew
        over and over and paid the terminal shortage penalty.  This baseline
        gives the network an explicit shadow value for inventory; the network
        only learns the residual Bellman correction.
        """
        P = x[:, 0]
        C = x[:, 1]
        if n >= self.horizon:
            return self.terminal_cost(x)

        remaining = max(int(self.horizon - n), 0)
        # Replacement/sale reference price: a short-run conditional mean.
        # This creates the correct economics: inject when P is below its future
        # mean and withdraw when P is above it.
        shadow = self.p_bar + self.beta * (P - self.p_bar)
        shadow = torch.clamp(shadow, min=0.5, max=15.0)

        linear_inventory = shadow * (self.c0 - C)

        # If even maximum injections cannot restore C to c0 by maturity, add
        # the unavoidable part of the terminal penalty.  The linear inventory
        # term already prices ordinary refillable deficits.
        max_reachable_C = C + remaining * self.ain
        unavoidable_shortage = torch.relu(self.c0 - max_reachable_C)
        terminal_mean = self.p_bar + (self.beta ** remaining) * (P - self.p_bar)
        terminal_mean = torch.clamp(terminal_mean, min=0.5, max=15.0)
        unavoidable_penalty = torch.relu(self.mu_penalty * terminal_mean - shadow) * unavoidable_shortage

        # Approximate carrying cost.  It discourages pathological overfilling
        # while keeping the baseline smooth.
        carry = self.Ki * C * float(remaining) * 0.5
        return linear_inventory + unavoidable_penalty + carry

    def hybrid_action_regularization_torch(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        """Small smooth guard used only by the stabilised Hybrid-Now.

        It is not a replacement for the Bellman target; it only makes the
        finite-action greedy step conservative when inventory is below c0 and
        the network residual is still noisy.  It vanishes for store and remains
        small compared with market cashflows.
        """
        C = x[:, 1:2]
        P = x[:, 0:1]
        inject, store, withdraw = self._gas_action_weights(a)
        C_after = C + inject * self.ain - withdraw * self.aout
        remaining_after = max(int(self.horizon - n - 1), 0)
        shortage_after = torch.relu(self.c0 - C_after)
        # Stronger close to maturity, weaker early.
        maturity_weight = 1.0 / math.sqrt(max(remaining_after + 1, 1))
        low_price_guard = torch.relu(self.p_bar - P)
        return (0.15 * maturity_weight * shortage_after * (1.0 + low_price_guard)).squeeze(1)

    def _h(self, c: Tensor, a_idx: int) -> Tensor:
        if a_idx == 1:    # inject
            delta = self.ain * torch.ones_like(c)
        elif a_idx == -1: # withdraw
            delta = -self.aout * torch.ones_like(c)
        else:             # hold
            delta = torch.zeros_like(c)
        return delta

    def _action_weights(self, a: Tensor):
        """
        Convention papier/figures:
        a = -1 : injection
        a =  0 : store
        a = +1 : withdraw
        """
        a_flat = a.reshape(-1) if a.ndim > 1 else a
    
        # Cas discret : Qknn / ClassifPI
        if self.action_mode == "discrete":
            inject = (a_flat < -0.5).float().unsqueeze(1)
            withdraw = (a_flat > 0.5).float().unsqueeze(1)
            store = 1.0 - inject - withdraw
            return inject, store, withdraw
    
        # Cas continu : Hybrid-Now, relaxation diffÃƒÆ’Ã‚Â©rentiable
        tau = 12.0
        scores = torch.cat(
            [
                -tau * (a_flat + 1.0).pow(2).unsqueeze(1),  # proche de -1 => inject
                -tau * (a_flat).pow(2).unsqueeze(1),        # proche de 0  => store
                -tau * (a_flat - 1.0).pow(2).unsqueeze(1),  # proche de +1 => withdraw
            ],
            dim=1,
        )
        weights = torch.softmax(scores, dim=1)
        inject = weights[:, 0:1]
        store = weights[:, 1:2]
        withdraw = weights[:, 2:3]
        return inject, store, withdraw
    def _gas_action_weights(self, a: Tensor):
        """
        Convention du papier :
          a = -1 -> injection
          a =  0 -> store
          a = +1 -> withdraw
    
        Pour Qknn/ClassifPI : décision dure.
        Pour Hybrid-Now : relaxation différentiable.
        """
        a_flat = a.reshape(-1) if a.ndim > 1 else a
    
        if self.action_mode == "continuous":
            # Relaxation différentiable pour Hybrid-Now.
            # Le réseau sort a dans [-1,1], et on l'interprète
            # comme mélange souple entre injection/store/withdraw.
            tau = 10.0
            scores = torch.cat(
                [
                    -tau * (a_flat + 1.0).pow(2).unsqueeze(1),  # injection, a≈-1
                    -tau * (a_flat).pow(2).unsqueeze(1),        # store, a≈0
                    -tau * (a_flat - 1.0).pow(2).unsqueeze(1),  # withdraw, a≈+1
                ],
                dim=1,
            )
            weights = torch.softmax(scores, dim=1)
            inject = weights[:, 0:1]
            store = weights[:, 1:2]
            withdraw = weights[:, 2:3]
        else:
            # Décision discrète exacte pour Qknn/ClassifPI.
            inject = (a_flat < -0.5).float().unsqueeze(1)
            withdraw = (a_flat > 0.5).float().unsqueeze(1)
            store = 1.0 - inject - withdraw
    
        return inject, store, withdraw
    
    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        P = x[:, 0:1]
        C = x[:, 1:2]
    
        inject, store, withdraw = self._gas_action_weights(a)
    
        C_after = C + inject * self.ain - withdraw * self.aout
        C_new = torch.clamp(C_after, self.C_min, self.C_max)
    
        P_new = self.p_bar * (1.0 - self.beta) + self.beta * P + eps
        P_new = torch.clamp(P_new, 0.5, 15.0)
    
        return torch.cat([P_new, C_new], dim=1)
    
    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        """
        Coût = -gain.
    
        Convention :
          a = -1 -> injection
          a =  0 -> store
          a = +1 -> withdraw
        """
        P = x[:, 0:1]
        C = x[:, 1:2]
    
        inject, store, withdraw = self._gas_action_weights(a)
    
        storage_cost = self.Ki * C
    
        gain = (
            inject * (-self.bin_rate * P - storage_cost)
            + store * (-storage_cost)
            + withdraw * (self.bout_rate * P - storage_cost)
        ).squeeze(1)
    
        C_after = C + inject * self.ain - withdraw * self.aout
    
        if self.action_mode == "continuous":
            # Pénalité différentiable pour Hybrid-Now.
            penalty = 1e4 * (
                torch.relu(self.C_min - C_after).pow(2)
                + torch.relu(C_after - self.C_max).pow(2)
            ).squeeze(1)
        else:
            # Pénalité dure pour Qknn/ClassifPI.
            infeasible = (C_after < self.C_min - 1e-8) | (C_after > self.C_max + 1e-8)
            penalty = 1e6 * infeasible.float().squeeze(1)
    
        return -gain + penalty

    def admissible_action_mask_np(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """State-dependent admissibility mask for Qknn.

        The paper optimizes over actions that keep the inventory in
        [C_min, C_max].  The convention used in the figures is
        a=-1 injection, a=0 store, a=+1 withdraw.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        a_arr = np.asarray(a, dtype=np.float32)
        if a_arr.ndim == 1:
            a_arr = a_arr[:, None] if a_arr.shape[0] != x_arr.shape[0] else a_arr.reshape(-1, 1)

        c = x_arr[:, 1]
        aa = a_arr[:, 0]
        delta = np.zeros_like(aa, dtype=np.float32)
        delta[aa < -0.5] = self.ain
        delta[aa > 0.5] = -self.aout
        c_next = c + delta
        return (c_next >= self.C_min - 1e-8) & (c_next <= self.C_max + 1e-8)

    def admissible_action_matrix_torch(self, x: Tensor) -> Tensor:
        """Feasibility matrix [batch, n_actions] for actions [-1, 0, +1].

        Convention used by the figures in the project:
          -1 -> injection, 0 -> store, +1 -> withdraw.
        """
        actions = to_tensor(self.discrete_actions, device=x.device, dtype=x.dtype).view(1, -1)
        C = x[:, 1:2]
        delta = torch.zeros(x.shape[0], actions.shape[1], device=x.device, dtype=x.dtype)
        delta = torch.where(actions < -0.5, torch.full_like(delta, self.ain), delta)
        delta = torch.where(actions > 0.5, torch.full_like(delta, -self.aout), delta)
        C_next = C + delta
        return (C_next >= self.C_min - 1e-8) & (C_next <= self.C_max + 1e-8)

    def terminal_viability_action_matrix_torch(self, x: Tensor, n: int) -> Tensor:
        """Safety mask used by neural finite-action gas-storage solvers.

        It keeps the physical inventory constraint and, near maturity, removes
        actions after which it is impossible to refill the cavern to the initial
        inventory c0 before terminal time. This prevents the value-network
        bootstrap from repeatedly withdrawing gas and then paying a huge
        terminal penalty.
        """
        base = self.admissible_action_matrix_torch(x)
        actions = to_tensor(self.discrete_actions, device=x.device, dtype=x.dtype).view(1, -1)
        C = x[:, 1:2]
        delta = torch.zeros(x.shape[0], actions.shape[1], device=x.device, dtype=x.dtype)
        delta = torch.where(actions < -0.5, torch.full_like(delta, self.ain), delta)
        delta = torch.where(actions > 0.5, torch.full_like(delta, -self.aout), delta)
        C_next = C + delta
        steps_left_after = max(int(self.horizon - n - 1), 0)
        can_reach_c0 = C_next + steps_left_after * self.ain >= self.c0 - 1e-8
        guarded = base & can_reach_c0
        no_guarded = ~guarded.any(dim=1, keepdim=True)
        return torch.where(no_guarded, base, guarded)

    def terminal_viability_action_matrix_np(self, x: np.ndarray, n: int) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        actions = np.asarray(self.discrete_actions, dtype=np.float32).reshape(1, -1)
        c = x_arr[:, 1:2]
        delta = np.zeros((x_arr.shape[0], actions.shape[1]), dtype=np.float32)
        flat_actions = actions.reshape(-1)
        delta[:, flat_actions < -0.5] = self.ain
        delta[:, flat_actions > 0.5] = -self.aout
        c_next = c + delta
        base = (c_next >= self.C_min - 1e-8) & (c_next <= self.C_max + 1e-8)
        steps_left_after = max(int(self.horizon - n - 1), 0)
        can_reach_c0 = c_next + steps_left_after * self.ain >= self.c0 - 1e-8
        guarded = base & can_reach_c0
        no_guarded = ~guarded.any(axis=1, keepdims=True)
        return np.where(no_guarded, base, guarded)

    def project_action_np(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Repair off-grid Qknn actions so inventory constraints are never violated."""
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        out = np.asarray(a, dtype=np.float32).copy()
        if out.ndim == 1:
            out = out[:, None]
        c = x_arr[:, 1]
        aa = out[:, 0]
        # If injection/withdrawal would violate bounds, store instead.
        inj_bad = (aa < -0.5) & (c + self.ain > self.C_max + 1e-8)
        wdr_bad = (aa > 0.5) & (c - self.aout < self.C_min - 1e-8)
        aa[inj_bad | wdr_bad] = 0.0
        out[:, 0] = aa
        return out

    def terminal_cost(self, x: Tensor) -> Tensor:
        P = x[:, 0]
        C = x[:, 1]
        return self.mu_penalty * P * torch.clamp(self.c0 - C, min=0.0)


def make_gas_storage_problem(
    N: int = 30,
    ain: float = 0.06,
    aout: float = 0.25,
) -> GasStorageProblem:
    return GasStorageProblem(
        state_dim=2,
        action_dim=1,
        horizon=N,
        T=float(N),
        action_mode="discrete",
        discrete_actions=np.array([-1.0, 0.0, 1.0], dtype=np.float32),
        ain=ain,
        aout=aout,
        bin_rate=ain,
        bout_rate=aout,
        name=f"GasStorage-ain={ain}",
    )


# ----------------------------
# Microgrid management problem (paper Section 3.5)
# ----------------------------

@dataclass
class MicrogridProblem(StochasticControlProblem):
    """
    State: x = (C, M, R) ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â battery charge, generator mode (0/1), residual demand.
    Control: a in {0} ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã¢â‚¬Â¹ÃƒÂ¢Ã¢â€šÂ¬Ã‚Â ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Âª [A_min, A_max] ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€šÃ‚Â generator power (0 = off).
    We use a ClassifHybrid encoding: (p_off, alpha_on).
    For rollout purposes, the action tensor is 1D: a=0 means off, a>0 means on with that power.
    """
    R_bar: float = 0.1
    rho: float = 0.9
    sigma_R: float = 0.2
    C_max: float = 1.0
    A_min: float = 0.05
    A_max: float = 10.0
    kappa: float = 0.2
    Q_minus: float = 10.0
    Q_plus: float = 1000.0
    K_fuel: float = 2.0
    gamma_fuel: float = 2.0
    c0: float = 0.0
    r0: float = 0.1

    def __post_init__(self) -> None:
        self.noise_dim = 1
        self.c_max = self.C_max   # alias for notebook
        super().__post_init__()

    def sample_noise(self, batch_size: int, device: Optional[str] = None) -> Tensor:
        return self.sigma_R * torch.randn(batch_size, 1, device=device)

    def sample_training_states(self, n: int, batch_size: int, device: Optional[str] = None) -> Tensor:
        """Training law µ_n tailored to the microgrid figures and value estimate.

        The paper's Figures 12/13 show switching frontiers in the whole (C,R)
        plotting window.  Sampling only from the marginal AR(1) law of R_n
        under-trains the frontier region R≈C and the tails visible in those
        figures.  We therefore use a mixture:
          - 60% from the marginal law of R_n,
          - 20% uniform on a broad paper-style R window,
          - 20% concentrated near the switching frontier R≈C.
        """
        n_marg = int(0.60 * batch_size)
        n_uniform = int(0.20 * batch_size)
        n_diag = batch_size - n_marg - n_uniform

        mean_n = self.R_bar + (self.rho ** n) * (self.r0 - self.R_bar)
        if n == 0:
            std_n = 1e-4
        else:
            var_n = self.sigma_R ** 2 * (1.0 - self.rho ** (2 * n)) / max(1.0 - self.rho ** 2, 1e-8)
            std_n = math.sqrt(max(var_n, 1e-8))

        def _sample_cm(k: int) -> tuple[Tensor, Tensor]:
            C = self.C_max * torch.rand(k, 1, device=device)
            M = (torch.rand(k, 1, device=device) > 0.5).float()
            return C, M

        C1, M1 = _sample_cm(n_marg)
        R1 = mean_n + std_n * torch.randn(n_marg, 1, device=device)

        C2, M2 = _sample_cm(n_uniform)
        if self.C_max <= 1.0:
            r_low = max(-1.5, mean_n - 4.0 * max(std_n, 0.20))
            r_high = min(1.8, mean_n + 4.0 * max(std_n, 0.20))
        else:
            r_low = max(-2.0, mean_n - 4.0 * max(std_n, 0.20))
            r_high = min(4.0, mean_n + 4.0 * max(std_n, 0.20))
        R2 = r_low + (r_high - r_low) * torch.rand(n_uniform, 1, device=device)

        C3, M3 = _sample_cm(n_diag)
        R3 = C3 + 0.18 * torch.randn(n_diag, 1, device=device)
        R3 = torch.clamp(R3, r_low, r_high)

        X = torch.cat([
            torch.cat([C1, M1, R1], dim=1),
            torch.cat([C2, M2, R2], dim=1),
            torch.cat([C3, M3, R3], dim=1),
        ], dim=0)
        perm = torch.randperm(X.shape[0], device=device)
        return X[perm]

    def dynamics(self, x: Tensor, a: Tensor, eps: Tensor, n: int) -> Tensor:
        C = x[:, 0:1]
        M = x[:, 1:2]
        R = x[:, 2:3]
        alpha = a[:, 0:1]  # generated power (0 = off)
        # battery balance
        surplus = alpha - R
        I_charge = torch.clamp(surplus, min=0.0)
        I_charge = torch.min(I_charge, self.C_max - C)
        O_discharge = torch.clamp(-surplus, min=0.0)
        O_discharge = torch.min(O_discharge, C)
        C_new = torch.clamp(C + I_charge - O_discharge, 0.0, self.C_max)
        M_new = (alpha > 0).float()
        R_new = self.R_bar * (1.0 - self.rho) + self.rho * R + eps
        return torch.cat([C_new, M_new, R_new], dim=1)

    def running_cost(self, x: Tensor, a: Tensor, n: int) -> Tensor:
        C = x[:, 0]
        M = x[:, 1]
        R = x[:, 2]
        alpha = a[:, 0]
        # fuel cost
        fuel = self.K_fuel * (alpha ** self.gamma_fuel) * (alpha > 0).float()
        # switching cost
        M_new = (alpha > 0).float()
        switch = self.kappa * (M_new != M).float()
        # imbalance penalty
        surplus = alpha - R
        I_charge = torch.clamp(surplus, min=0.0)
        I_charge = torch.clamp(I_charge, max=self.C_max - C)
        O_discharge = torch.clamp(-surplus, min=0.0)
        O_discharge = torch.clamp(O_discharge, max=C)
        imbalance = R - alpha + I_charge - O_discharge  # S^alpha
        excess_penalty = self.Q_minus * torch.clamp(-imbalance, min=0.0)
        unmet_penalty = self.Q_plus * torch.clamp(imbalance, min=0.0)
        return fuel + switch + excess_penalty + unmet_penalty

    def admissible_action_mask_np(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """State-dependent admissibility mask for Qknn.

        The microgrid constraint is S(x,a) <= 0, equivalently a >= R - C.
        Passing this mask to Qknn matches the paper's implementation comment:
        Qknn searches directly in A_n(x), while the NN solver uses Q+ penalties.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        a_arr = np.asarray(a, dtype=np.float32)
        if a_arr.ndim == 1:
            a_arr = a_arr[:, None] if a_arr.shape[0] != x_arr.shape[0] else a_arr.reshape(-1, 1)

        c = x_arr[:, 0]
        r = x_arr[:, 2]
        alpha = a_arr[:, 0]
        return alpha >= (r - c - 1e-8)

    def project_action_np(self, x: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Project a policy output into A_n(x) = {0}∪[Amin,Amax] with a >= R-C.

        This is mainly for Qknn interpolation: the DP grid optimizes over
        admissible actions, but interpolating Q-values between grid points can
        otherwise choose a slightly infeasible off/low action and create a huge
        artificial Q+ penalty in Monte Carlo rollouts.
        """
        x_arr = np.asarray(x, dtype=np.float32)
        if x_arr.ndim == 1:
            x_arr = x_arr[None, :]
        out = np.asarray(a, dtype=np.float32).copy()
        if out.ndim == 1:
            out = out[:, None]
        c = x_arr[:, 0]
        r = x_arr[:, 2]
        required = r - c
        alpha = out[:, 0]
        off_ok = (alpha <= 1e-8) & (required <= 0.0)
        lower = np.maximum(self.A_min, required)
        lower = np.clip(lower, self.A_min, self.A_max)
        alpha = np.where(off_ok, 0.0, np.maximum(alpha, lower))
        alpha = np.clip(alpha, 0.0, self.A_max)
        out[:, 0] = alpha
        return out

    def terminal_cost(self, x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], device=x.device)


def make_microgrid_problem(N: int = 30, C_max: float = 1.0) -> MicrogridProblem:
    return MicrogridProblem(
        state_dim=3,
        action_dim=1,
        horizon=N,
        T=float(N),
        action_mode="continuous",
        action_bounds=(0.0, 10.0),
        C_max=C_max,
        name=f"Microgrid-N={N}",
    )


# ----------------------------
# ClassifHybrid solver (Algorithm 6, paper Section 3.5)
# ----------------------------

class ClassifHybridSolver(BaseNeuralSolver):
    """
    Mixed discrete/continuous control solver for the microgrid.
    Learns (p_off(x), alpha(x)) simultaneously:
      - p_off: probability of switching off (classification head)
      - alpha: continuous power level when on (regression head)
    """

    def solve(self, problem: StochasticControlProblem) -> Solution:
        N = problem.horizon
        dev = self.config.device
        hs = self.config.hidden_sizes

        # Build a shared trunk + two heads.
        def make_net() -> nn.Module:
            class ClassifHybridNet(nn.Module):
                def __init__(self, in_dim: int, hidden: tuple, A_min: float, A_max: float) -> None:
                    super().__init__()
                    self.A_min = A_min
                    self.A_max = A_max
                    layers: list = []
                    prev = in_dim
                    for h in hidden:
                        layers += [nn.Linear(prev, h), nn.ELU()]
                        prev = h
                    self.trunk = nn.Sequential(*layers)
                    self.p_off_head = nn.Linear(prev, 1)   # logit for p_off
                    self.alpha_head = nn.Linear(prev, 1)   # alpha when on

                def _state_dependent_on_lower_bound(self, x: Tensor) -> Tensor:
                    # Admissibility for the "on" branch: alpha >= R - C and
                    # alpha in [A_min, A_max].  If R-C > A_max the constraint is
                    # infeasible on the finite action set; the Q+ penalty still
                    # handles that rare tail state.
                    required = torch.relu(x[:, 2:3] - x[:, 0:1])
                    lower = torch.maximum(
                        torch.full_like(required, self.A_min),
                        required,
                    )
                    return torch.minimum(lower, torch.full_like(lower, self.A_max))

                def forward(self, x: Tensor):
                    feat = self.trunk(x)
                    p_off = torch.sigmoid(self.p_off_head(feat))
                    raw_alpha = self.alpha_head(feat)
                    lower = self._state_dependent_on_lower_bound(x)
                    alpha = lower + (self.A_max - lower) * torch.sigmoid(raw_alpha)
                    return p_off, alpha

                def action(self, x: Tensor) -> Tensor:
                    p_off, alpha = self.forward(x)
                    a_off = torch.zeros_like(alpha)
                    # The off action is admissible only when the battery can cover
                    # positive demand, i.e. R-C <= 0.  This removes pathological
                    # rollouts where the classifier outputs "off" in forbidden
                    # states after being trained with a finite penalty.
                    off_feasible = (x[:, 2:3] - x[:, 0:1]) <= 0.0
                    return torch.where((p_off > 0.5) & off_feasible, a_off, alpha)

            A_min = float(np.asarray(problem.action_bounds[0]))
            A_max = float(np.asarray(problem.action_bounds[1]))
            return ClassifHybridNet(problem.state_dim, tuple(hs), max(A_min, 0.01), A_max).to(dev)

        value_nets: List[Optional[nn.Module]] = [None] * N
        policy_nets: List[Optional[nn.Module]] = [None] * N
        policies: List[Callable[[Tensor], Tensor]] = [lambda x: x] * N
        value_functions: List[Callable[[Tensor], Tensor]] = [self._terminal_value(problem) for _ in range(N)]
        V_next = self._terminal_value(problem)
        next_value_net = None
        logs = SolverLogs(meta={"method": "ClassifHybrid"})

        for n in _make_tqdm(reversed(range(N)), desc="ClassifHybrid -> n", verbose=self.config.verbose, total=N):
            net = make_net()
            if self.config.transfer_warm_start and policy_nets[n + 1 if n + 1 < N else n] is not None:
                try:
                    net.load_state_dict(copy.deepcopy(policy_nets[n + 1].state_dict()))
                except Exception:
                    pass
            if next_value_net is not None:
                freeze_module(next_value_net)

            def policy_loss_closure(net=net, Vn1=V_next, step_n=n) -> Tensor:
                x = problem.sample_training_states(step_n, self.config.batch_size, dev)
                p_off, alpha = net(x)
                a_off = torch.zeros_like(alpha)
                # Same shock for both branches.  This is a common-random-number
                # estimate of Algorithm 6 and greatly reduces the variance of
                # the off/on comparison.
                eps = problem.sample_noise(self.config.batch_size, dev)
                x_off = problem.dynamics(x, a_off, eps, step_n)
                x_on = problem.dynamics(x, alpha, eps, step_n)
                cost_off = problem.running_cost(x, a_off, step_n) + Vn1(x_off)
                cost_on = problem.running_cost(x, alpha, step_n) + Vn1(x_on)
                return (p_off[:, 0] * cost_off + (1.0 - p_off[:, 0]) * cost_on).mean()

            policy_losses = self._train_loop(net, policy_loss_closure, desc=f"  n={n}")
            logs.policy_losses.insert(0, policy_losses)
            freeze_module(net)
            policy_nets[n] = net
            policies[n] = lambda x, net=net: net.action(x)

            value_net = self._new_value_net(problem)
            self._maybe_copy(value_net, next_value_net)

            def value_loss_closure(net=net, value_net=value_net, Vn1=V_next, step_n=n) -> Tensor:
                x = problem.sample_training_states(step_n, self.config.batch_size, dev)
                with torch.no_grad():
                    p_off, alpha = net(x)
                    a_off = torch.zeros_like(alpha)
                    eps = problem.sample_noise(self.config.batch_size, dev)
                    x_off = problem.dynamics(x, a_off, eps, step_n)
                    x_on = problem.dynamics(x, alpha, eps, step_n)
                    target_off = problem.running_cost(x, a_off, step_n) + Vn1(x_off)
                    target_on = problem.running_cost(x, alpha, step_n) + Vn1(x_on)
                pred = value_net(x)
                # Algorithm 6 regresses the value against the two branch targets
                # weighted by the learned classification probability; using the
                # hard argmax here was another source of unstable microgrid plots.
                return (
                    p_off[:, 0] * (target_off.detach() - pred).pow(2)
                    + (1.0 - p_off[:, 0]) * (target_on.detach() - pred).pow(2)
                ).mean()

            value_losses = self._train_loop(value_net, value_loss_closure, desc=f"  n={n}")
            logs.value_losses.insert(0, value_losses)
            freeze_module(value_net)
            value_nets[n] = value_net
            value_functions[n] = lambda x, net=value_net: net(x)
            V_next = value_functions[n]
            next_value_net = value_net

        return Solution(
            method="ClassifHybrid",
            policies=policies,
            value_functions=value_functions,
            policy_nets=policy_nets,
            value_nets=value_nets,
            logs=logs,
            problem_name=problem.name,
        )


# ----------------------------
# Linear-terminal semilinear benchmark (exact affine solution)
# ----------------------------

def make_linear_terminal_semilinear_problem(
    c: np.ndarray,
    N: int = 20,
    T: float = 1.0,
    action_bounds: tuple = (-5.0, 5.0),
) -> SemilinearPDEProblem:
    """
    g(x) = c . x  (linear terminal cost).
    Exact solution: v(t,x) = -exp(c.x + |c|^2 * (T-t))  ... (log-transform).
    Actually: v(t,x) = -log E[exp(-c.(x+sqrt2 W))] = c.x + |c|^2*(T-t).
    Optimal control: alpha*(t,x) = -c  (constant).
    """
    c_arr = np.asarray(c, dtype=np.float32).reshape(-1)
    d = c_arr.shape[0]

    def g(x: Tensor) -> Tensor:
        cv = to_tensor(c_arr, device=x.device)
        return (x * cv).sum(dim=1)

    return SemilinearPDEProblem(
        state_dim=d,
        action_dim=d,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=action_bounds,
        terminal_cost_fn=g,
        name=f"LinearTerminalSemilinear-d={d}",
    )


def exact_linear_terminal_solution(
    problem: SemilinearPDEProblem,
    c: Optional[np.ndarray] = None,
) -> Solution:
    """
    Exact solution for g(x) = c.x:
      v(t,x) = c.x + |c|^2*(T-t)
      alpha*(t,x) = -c  (constant, independent of x)
    """
    N = problem.horizon
    T = problem.T
    dt = problem.dt

    d = problem.state_dim
    if c is None:
        # Infer c from the terminal cost evaluated on the canonical basis.
        e = torch.eye(d)
        c_arr = problem.terminal_cost(e).detach().cpu().numpy().astype(np.float32)
    else:
        c_arr = np.asarray(c, dtype=np.float32).reshape(-1)

    norm_sq = float((c_arr ** 2).sum())

    def make_val(n: int) -> Callable[[Tensor], Tensor]:
        tau = (N - n) * dt  # time-to-go
        def val(x: Tensor) -> Tensor:
            c_dev = to_tensor(c_arr, device=x.device)
            return (x * c_dev).sum(dim=1) + norm_sq * tau
        return val

    def make_pol(n: int) -> Callable[[Tensor], Tensor]:
        def pol(x: Tensor) -> Tensor:
            c_dev = to_tensor(c_arr, device=x.device)
            return -c_dev.unsqueeze(0).expand(x.shape[0], -1)
        return pol

    policies = [make_pol(n) for n in range(N)]
    value_functions = [make_val(n) for n in range(N)]
    return Solution(
        method="Exact-LinearTerminal",
        policies=policies,
        value_functions=value_functions,
        policy_nets=[None] * N,
        value_nets=[None] * N,
        logs=SolverLogs(meta={"method": "Exact-LinearTerminal"}),
        problem_name=problem.name,
    )


# ----------------------------
# Quadratic-terminal semilinear benchmark (exact recursion)
# ----------------------------

def make_quadratic_terminal_semilinear_problem(
    alpha: float = 0.7,
    N: int = 20,
    T: float = 1.0,
    action_bounds: tuple = (-5.0, 5.0),
) -> SemilinearPDEProblem:
    """
    g(x) = alpha * x^2  (1D).
    Hopf-Cole: v(t,x) = -log E[exp(-alpha*(x+sqrt2 W_{T-t})^2)]
             = alpha*x^2/(1+4*alpha*(T-t)) - 0.5*log(1+4*alpha*(T-t)).
    """
    def g(x: Tensor) -> Tensor:
        return alpha * (x[:, 0] ** 2)

    return SemilinearPDEProblem(
        state_dim=1,
        action_dim=1,
        horizon=N,
        T=T,
        action_mode="continuous",
        action_bounds=action_bounds,
        terminal_cost_fn=g,
        name=f"QuadraticTerminalSemilinear-alpha={alpha}",
    )


def exact_quadratic_terminal_solution(
    problem: SemilinearPDEProblem,
    alpha: float = 0.7,
) -> Solution:
    """
    Exact solution for g(x) = alpha*x^2 (1D semilinear PDE):
      v(t,x) = alpha*x^2 / (1+4*alpha*(T-t)) - 0.5*log(1+4*alpha*(T-t))
      alpha*(t,x) = -Dx v(t,x) = -2*alpha*x / (1+4*alpha*(T-t))
    """
    N = problem.horizon
    T = problem.T
    dt = problem.dt

    def make_val(n: int) -> Callable[[Tensor], Tensor]:
        tau = (N - n) * dt
        denom = 1.0 + 4.0 * alpha * tau
        def val(x: Tensor) -> Tensor:
            return alpha * x[:, 0] ** 2 / denom - 0.5 * math.log(max(denom, 1e-12))
        return val

    def make_pol(n: int) -> Callable[[Tensor], Tensor]:
        tau = (N - n) * dt
        denom = 1.0 + 4.0 * alpha * tau
        def pol(x: Tensor) -> Tensor:
            return (-2.0 * alpha * x[:, 0] / denom).unsqueeze(1)
        return pol

    policies = [make_pol(n) for n in range(N)]
    value_functions = [make_val(n) for n in range(N)]
    return Solution(
        method="Exact-QuadraticTerminal",
        policies=policies,
        value_functions=value_functions,
        policy_nets=[None] * N,
        value_nets=[None] * N,
        logs=SolverLogs(meta={"method": "Exact-QuadraticTerminal"}),
        problem_name=problem.name,
    )

