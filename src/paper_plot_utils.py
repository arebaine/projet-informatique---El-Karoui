from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

import stochastic_control_core as scc


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, path: Path | str, dpi: int = 180) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def eval_value(sol, n: int, x_grid, device: Optional[str] = None) -> np.ndarray:
    device = device or scc.default_device()
    x = torch.tensor(np.asarray(x_grid, np.float32), device=device)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    out = sol.value(n, x)
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy().reshape(-1)
    return np.asarray(out).reshape(-1)


@torch.no_grad()
def eval_policy(sol, n: int, x_grid, device: Optional[str] = None) -> np.ndarray:
    device = device or scc.default_device()
    x = torch.tensor(np.asarray(x_grid, np.float32), device=device)
    if x.ndim == 1:
        x = x.unsqueeze(1)
    out = sol.policy(n, x)
    if isinstance(out, torch.Tensor):
        return out.detach().cpu().numpy()
    return np.asarray(out)


def zero_solution(problem: scc.StochasticControlProblem) -> scc.Solution:
    d = problem.action_dim
    return scc.Solution(
        method="Zero",
        policies=[lambda x, d=d: torch.zeros(x.shape[0], d, device=x.device) for _ in range(problem.horizon)],
        value_functions=[lambda x: torch.zeros(x.shape[0], device=x.device) for _ in range(problem.horizon)],
        policy_nets=[None] * problem.horizon,
        value_nets=[None] * problem.horizon,
        logs=scc.SolverLogs(),
        problem_name=problem.name,
    )


class ExactOptionPolicy:
    def __init__(self, exact_dict: Dict, problem: scc.OptionHedgingProblem):
        self.exact = exact_dict
        self.problem = problem
        self.method = "Opt"
        self.policies = [None] * problem.horizon
        self.value_functions = [None] * problem.horizon

    def policy(self, n: int, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        a = self.exact["policy"](n, x_np[:, 0], x_np[:, 1]).reshape(-1, 1)
        return torch.tensor(a, dtype=x.dtype, device=x.device)

    def value(self, n: int, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().cpu().numpy()
        v = self.exact["value"](n, x_np[:, 0], x_np[:, 1]).reshape(-1)
        return torch.tensor(v, dtype=x.dtype, device=x.device)


@torch.no_grad()
def rollout_on_fixed_returns(
    problem: scc.OptionHedgingProblem,
    solution,
    w0: float,
    returns_matrix: np.ndarray,
    device: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    device = device or scc.default_device()
    n_paths, N = returns_matrix.shape
    x = np.column_stack([
        np.full(n_paths, w0, dtype=np.float32),
        np.full(n_paths, problem.p0, dtype=np.float32),
    ])
    traj = [x.copy()]
    for n in range(N):
        xt = torch.tensor(x, dtype=torch.float32, device=device)
        a = solution.policy(n, xt).detach().cpu().numpy().reshape(-1, 1)
        r = returns_matrix[:, n:n+1].astype(np.float32)
        x = np.concatenate([x[:, :1] + a * r, x[:, 1:2] * (1.0 + r)], axis=1)
        traj.append(x.copy())
    payoff = np.maximum(x[:, 1] - problem.strike, 0.0)
    err = payoff - x[:, 0]
    cost = err ** 2
    return {
        "trajectory": traj,
        "wealth": np.stack([u[:, 0] for u in traj], axis=1),
        "price": np.stack([u[:, 1] for u in traj], axis=1),
        "terminal_error": err,
        "terminal_cost": cost,
    }


def shared_returns_matrix(problem: scc.OptionHedgingProblem, n_paths: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.asarray(problem.returns, dtype=float)
    probs = np.asarray(problem.return_probs, dtype=float)
    idx = rng.choice(len(returns), size=(n_paths, problem.horizon), p=probs)
    return returns[idx]


def mc_confidence(costs: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    costs = np.asarray(costs, dtype=float).reshape(-1)
    mean = float(costs.mean())
    std = float(costs.std(ddof=0))
    half = 1.96 * std / max(math.sqrt(len(costs)), 1e-12)
    return {"mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


# ---------- plotting helpers ----------

def plot_rel_error_curve(xvals: Sequence[float], yvals: Sequence[float], path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xvals, yvals, marker="o", linewidth=2)
    ax.set_xlabel("Size of training set")
    ax.set_ylabel("Relative error w.r.t. benchmark")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    savefig(fig, path)


def plot_overlay_first_component(rolls: Mapping[str, Dict], path: Path | str, title: str, max_paths: int = 5) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    styles = {"Hybrid-Now (opt)": "-", "α=0 (bench)": "-", "α=0 (naïf)": "-"}
    colors = {"Hybrid-Now (opt)": "tab:blue", "α=0 (bench)": "tab:red", "α=0 (naïf)": "tab:red"}
    for label, roll in rolls.items():
        arr = np.stack([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in roll["trajectory"]], axis=0)
        ts = np.arange(arr.shape[0]) / max(arr.shape[0] - 1, 1)
        for i in range(min(max_paths, arr.shape[1])):
            ax.plot(ts, arr[:, i, 0], linestyle=styles.get(label, "-"), color=colors.get(label, None), alpha=0.85, label=label if i == 0 else None)
    ax.set_xlabel("t")
    ax.set_ylabel("X_1")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, path)


def plot_lq_policy_curves(sol, x_grid: np.ndarray, path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in range(len(sol.policies)):
        a = eval_policy(sol, n, x_grid).reshape(-1)
        ax.plot(x_grid, a, linewidth=1.2, label=str(n))
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\hat\alpha^{opt}$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="optimal decision at time n=", ncol=2, fontsize=7, title_fontsize=8)
    savefig(fig, path)


def plot_lq_value_curves(sol, x_grid: np.ndarray, path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for n in range(len(sol.value_functions)):
        v = eval_value(sol, n, x_grid.reshape(-1, 1))
        ax.plot(x_grid, v, linewidth=1.2, label=str(n))
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\hat V$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(title="value function at time n=", ncol=2, fontsize=7, title_fontsize=8)
    savefig(fig, path)


def plot_lq_forward_components(roll_opt: Dict, roll_bench: Optional[Dict], path: Path | str, title: str) -> None:
    arr = np.stack([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in roll_opt["trajectory"]], axis=0)[:, 0, :]
    fig, ax = plt.subplots(figsize=(8, 4))
    for j in range(arr.shape[1]):
        ax.plot(np.arange(arr.shape[0]), arr[:, j], alpha=0.9, label=f"component {j}" if j < 10 else None)
    if roll_bench is not None:
        bench = np.stack([t.numpy() if hasattr(t, 'numpy') else np.asarray(t) for t in roll_bench["trajectory"]], axis=0)[:, 0, 0]
        ax.plot(np.arange(len(bench)), bench, color="tab:red", linewidth=2.0, label="bench")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7)
    savefig(fig, path)


def plot_hedging_value_curve(w_grid: np.ndarray, exact_v: np.ndarray, hn_v: np.ndarray, hlq_v: np.ndarray, path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(w_grid, hn_v, linewidth=1.8, label="Hybrid-Now")
    ax.plot(w_grid, hlq_v, linestyle="--", linewidth=1.8, label="Hybrid-LaterQ")
    ax.plot(w_grid, exact_v, color="tab:red", linewidth=2.2, label="Opt")
    ax.set_xlabel(r"$w_0$")
    ax.set_ylabel(r"$V(t=0,W_0=w_0)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, path)


def plot_hedging_paths(shared_paths: Mapping[str, np.ndarray], path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Opt": "tab:red", "Hybrid-Now": "tab:blue", "Hybrid-LaterQ": "tab:green"}
    for label, wealth in shared_paths.items():
        wealth = np.asarray(wealth)
        for i in range(wealth.shape[0]):
            ax.plot(np.arange(wealth.shape[1]), wealth[i], color=colors.get(label), alpha=0.85, label=label if i == 0 else None)
    ax.set_xlabel("n")
    ax.set_ylabel("Portfolio wealth")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, path)


def plot_gas_ain_curve(ain_vals: Sequence[float], qknn_vals: Sequence[float], bench_vals: Sequence[float], path: Path | str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ain_vals, qknn_vals, marker="o", label="Qknn")
    ax.plot(ain_vals, bench_vals, linewidth=2.0, label=r"$\alpha=0$")
    ax.set_xlabel(r"$a_{in}$")
    ax.set_ylabel(r"$V(0,P_0,C_0)$")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    savefig(fig, path)


_GAS_QKNN_CMAP = mcolors.ListedColormap(["tab:red", "black", "tab:blue"])
_GAS_OTHER_CMAP = mcolors.ListedColormap(["purple", "tab:blue", "gold"])


def _gas_action_to_index(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a).reshape(-1)
    idx = np.zeros_like(a, dtype=int)

    idx[a < -0.5] = 0                         # injection
    idx[(a >= -0.5) & (a <= 0.5)] = 1        # store
    idx[a > 0.5] = 2                         # withdraw

    return idx


def plot_gas_decision_panels(
    sol,
    n_list: Sequence[int],
    p_vals: np.ndarray,
    c_vals: np.ndarray,
    path: Path | str,
    title: str,
    qknn_style: bool = False,
    eval_grids: Optional[Sequence[np.ndarray]] = None,
    scatter: Optional[bool] = None,
    paper_bounds: Optional[Mapping[int, Tuple[float, float, float, float]]] = None,
    interpolation: str = "nearest",
) -> None:
    """Plot gas-storage decisions in the style of the paper.

    - Qknn (Fig. 9): scatter on the discrete/reachable grid, red/black/blue.
    - ClassifPI (Fig. 10): scatter cloud, purple/blue/yellow.
    - Hybrid-Now (Fig. 11): heatmap on the paper-like panel bounds.

    ``paper_bounds[n]`` is (p_low, p_high, c_low, c_high).  It is important not
    to force every panel onto the full rectangle [3,7]x[0,8], because the paper
    displays each date on the state window relevant for that date.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), squeeze=False)
    cmap = _GAS_QKNN_CMAP if qknn_style else _GAS_OTHER_CMAP
    labels = ["Injection", "Store", "Withdraw"]

    meta_grids = None
    if hasattr(sol, "logs") and isinstance(getattr(sol.logs, "meta", None), dict):
        meta_grids = sol.logs.meta.get("state_grids")
    if eval_grids is None and meta_grids is not None:
        eval_grids = meta_grids
    if scatter is None:
        scatter = eval_grids is not None or qknn_style

    for ax, n in zip(axes.ravel(), n_list):
        if paper_bounds is not None and n in paper_bounds:
            p_low, p_high, c_low, c_high = paper_bounds[n]
        else:
            p_low, p_high = float(np.min(p_vals)), float(np.max(p_vals))
            c_low, c_high = float(np.min(c_vals)), float(np.max(c_vals))

        if scatter and eval_grids is not None:
            grid = np.asarray(eval_grids[n], dtype=np.float32)
            # Restrict to the displayed panel, otherwise scatter plots have
            # out-of-window points that distort the axes and the visual density.
            in_panel = (
                (grid[:, 0] >= p_low) & (grid[:, 0] <= p_high) &
                (grid[:, 1] >= c_low) & (grid[:, 1] <= c_high)
            )
            if np.any(in_panel):
                grid = grid[in_panel]
            acts = eval_policy(sol, n, grid).reshape(-1)
            idx = _gas_action_to_index(acts)
            for k in range(3):
                m = idx == k
                if m.any():
                    ax.scatter(
                        grid[m, 0],
                        grid[m, 1],
                        s=7 if qknn_style else 6,
                        c=[cmap(k)],
                        marker=".",
                        linewidths=0,
                    )
            ax.set_xlim(p_low, p_high)
            ax.set_ylim(c_low, c_high)
        else:
            p_panel = np.linspace(p_low, p_high, len(p_vals))
            c_panel = np.linspace(c_low, c_high, len(c_vals))
            xx, yy = np.meshgrid(p_panel, c_panel)
            dense_grid = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
            acts = eval_policy(sol, n, dense_grid).reshape(-1)
            img = _gas_action_to_index(acts).reshape(xx.shape)
            ax.imshow(
                img,
                origin="lower",
                extent=[p_low, p_high, c_low, c_high],
                aspect="auto",
                cmap=cmap,
                vmin=-0.5,
                vmax=2.5,
                interpolation=interpolation,
            )

        ax.set_title(f"Decision at time {n}")
        ax.set_xlabel("P")
        ax.set_ylabel("C")
        ax.grid(True, alpha=0.15)

    handles = [plt.Line2D([0], [0], color=cmap(i), linewidth=8) for i in range(3)]
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title)
    savefig(fig, path)


def _micro_auto_norm(values: np.ndarray, vmin: float = 0.0, vmax: Optional[float] = None) -> mcolors.Normalize:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    vals = vals[vals > vmin + 1e-12]
    if vmax is None:
        if vals.size == 0:
            vmax_eff = 1.0
        else:
            vmax_eff = float(np.nanmax(vals))
            # keep colorbars readable; the paper uses local scales per panel
            vmax_eff = max(vmax_eff, 1e-6)
    else:
        vmax_eff = float(vmax)
    return mcolors.Normalize(vmin=float(vmin), vmax=vmax_eff)


def plot_micro_boundary_panels(
    sol,
    n_list: Sequence[int],
    m_values: Sequence[float],
    c_vals: np.ndarray,
    r_vals: np.ndarray,
    path: Path | str,
    title: str,
    threshold: float = 5e-2,
    r_bounds: Optional[Mapping[int, Tuple[float, float]]] = None,
    vmin: float = 0.0,
    vmax: Optional[float] = None,
) -> None:
    """Plot Experiment-5 Qknn maps like Figure 12 in the paper.

    Paper styling: red heatmap for the amount of generated power, a thin blue
    on/off frontier, white/light background where the generator is off, and a
    local colorbar for each panel.
    """
    cmap = plt.cm.Reds.copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(len(n_list), len(m_values), figsize=(10.5, 12.0), squeeze=False)

    for i, n in enumerate(n_list):
        if r_bounds is not None and int(n) in r_bounds:
            r_low, r_high = r_bounds[int(n)]
            r_panel = np.linspace(r_low, r_high, len(r_vals))
        else:
            r_panel = r_vals
            r_low, r_high = float(r_vals.min()), float(r_vals.max())
        cc, rr = np.meshgrid(c_vals, r_panel)

        for j, m_val in enumerate(m_values):
            ax = axes[i, j]
            grid = np.stack([cc.ravel(), np.full(cc.size, m_val), rr.ravel()], axis=1).astype(np.float32)
            acts = eval_policy(sol, int(n), grid).reshape(cc.shape)
            acts = np.maximum(acts, 0.0)
            on_mask = acts > threshold
            panel_norm = _micro_auto_norm(acts[on_mask], vmin=vmin, vmax=vmax)
            power = np.ma.masked_where(~on_mask, acts)

            im = ax.imshow(
                power,
                origin="lower",
                extent=[float(c_vals.min()), float(c_vals.max()), float(r_low), float(r_high)],
                aspect="auto",
                interpolation="nearest",
                cmap=cmap,
                norm=panel_norm,
            )
            try:
                cs = ax.contour(cc, rr, acts, levels=[threshold], colors=["#5a5cff"], linewidths=1.1)
                has_boundary = bool(cs.allsegs and cs.allsegs[0] and len(cs.allsegs[0][0]) > 1)
            except Exception:
                has_boundary = False
            if not has_boundary:
                if np.all(~on_mask):
                    ax.text(0.5, 0.5, "off everywhere", transform=ax.transAxes, ha="center", va="center", fontsize=8)
                elif np.all(on_mask):
                    ax.text(0.5, 0.5, "on everywhere", transform=ax.transAxes, ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"m={int(m_val)}")
            ax.set_xlabel("C")
            ax.set_ylabel("R")
            ax.set_xlim(float(c_vals.min()), float(c_vals.max()))
            ax.set_ylim(float(r_low), float(r_high))
            ax.grid(False)

        axes[i, 0].text(1.36, 1.08, f"Decision at time {int(n)}", transform=axes[i, 0].transAxes, ha="center", va="bottom", fontsize=11)

    fig.suptitle(title, y=0.995)
    savefig(fig, path)


def plot_micro_scatter_panels(
    sol,
    n_list: Sequence[int],
    m_values: Sequence[float],
    clouds: Mapping[Tuple[int, float], np.ndarray],
    path: Path | str,
    title: str,
    r_bounds: Optional[Mapping[int, Tuple[float, float]]] = None,
    threshold: float = 5e-2,
    vmax: Optional[float] = None,
) -> None:
    """ClassifHybrid scatter plot like Figure 13 in the paper.

    No artificial blue frontier or open-circle styling is drawn here: the paper
    shows red scatter points colored by generated power, with near-zero power
    points appearing white/light-red.
    """
    cmap = plt.cm.Reds.copy()
    fig, axes = plt.subplots(len(n_list), len(m_values), figsize=(10.5, 12.0), squeeze=False)

    for i, n in enumerate(n_list):
        row_clouds = [clouds[(int(n), float(m))] for m in m_values]
        c_max = max(float(cl[:, 0].max()) for cl in row_clouds) if row_clouds else 1.0
        if r_bounds is not None and int(n) in r_bounds:
            r_low, r_high = r_bounds[int(n)]
        else:
            stacked = np.vstack(row_clouds)
            r_low, r_high = float(stacked[:, 2].min()), float(stacked[:, 2].max())

        for j, m_val in enumerate(m_values):
            ax = axes[i, j]
            pts = np.asarray(clouds[(int(n), float(m_val))], dtype=np.float32)
            acts = eval_policy(sol, int(n), pts).reshape(-1)
            acts = np.maximum(acts, 0.0)
            norm = _micro_auto_norm(acts, vmin=0.0, vmax=vmax)
            sc = ax.scatter(
                pts[:, 0], pts[:, 2],
                s=9,
                c=acts,
                cmap=cmap,
                norm=norm,
                linewidths=0,
                alpha=0.95,
            )
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Decisions at time n={int(n)} for m={int(m_val)}")
            ax.set_xlabel("C")
            ax.set_ylabel("R")
            ax.set_xlim(0.0, c_max)
            ax.set_ylim(r_low, r_high)
            ax.grid(False)

    fig.suptitle(title, y=0.995)
    savefig(fig, path)


def plot_micro_trajectories(roll: Dict, path: Path | str, title: str, n_paths: int = 2) -> None:
    arr = np.stack([t.numpy() if hasattr(t, "numpy") else np.asarray(t) for t in roll["trajectory"]], axis=0)
    # arr shape: (T+1, n_paths, 3)
    fig, axes = plt.subplots(n_paths, 1, figsize=(10.5, 3.3 * n_paths), sharex=True)
    if n_paths == 1:
        axes = [axes]
    t = np.arange(arr.shape[0])
    for i in range(n_paths):
        axes[i].plot(t, arr[:, i, 0], linewidth=1.8, label="C")
        axes[i].plot(t, arr[:, i, 1], linewidth=1.8, label="M")
        axes[i].plot(t, arr[:, i, 2], linewidth=1.8, label="R")
        axes[i].set_title(f"Simulation {i + 1}")
        axes[i].grid(True, alpha=0.25)
        axes[i].legend(loc="upper right")
        ymin = min(float(arr[:, i, 2].min()) - 0.2, -0.2)
        ymax = max(float(arr[:, i, 0].max()) + 0.2, float(arr[:, i, 2].max()) + 0.2, 1.2)
        axes[i].set_ylim(ymin, ymax)
    axes[-1].set_xlabel("n")
    fig.suptitle(title)
    savefig(fig, path)
