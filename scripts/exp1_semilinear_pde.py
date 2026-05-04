from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import stochastic_control_core as scc
import stochastic_control_extensions as sce
from paper_plot_utils import plot_rel_error_curve, plot_overlay_first_component, ensure_dir, zero_solution


def make_cfg(hidden=(64, 64, 64), epochs=20, batch_size=512, n_batches=8, lr=5e-3, verbose=False):
    return scc.NeuralSolverConfig(
        hidden_sizes=hidden,
        epochs=epochs,
        n_batches_per_epoch=n_batches,
        batch_size=batch_size,
        learning_rate=lr,
        l2_reg=1e-4,
        verbose=verbose,
        seed=1234,
    )


def figure1_sweep(fast: bool, fig_dir: Path, progress: bool = True) -> pd.DataFrame:
    d = 10 if fast else 100
    budgets = [2, 4, 6, 8, 10, 12] if fast else [4, 8, 12, 16, 20, 24]
    problem = scc.make_test1_semilinear_problem(d=d, T=1.0, N=20)
    x0 = np.zeros(d, dtype=np.float32)
    bench = float(scc.semilinear_closed_form_mc(problem.terminal_cost, x0, t=0.0, T=1.0, n_mc=4000 if fast else 20000)[0])
    rows = []
    xvals, yvals = [], []
    for n_batches in scc.progress_iter(budgets, desc='Semilinear Fig.1 budgets', enabled=progress, total=len(budgets)):
        cfg = make_cfg(hidden=(64, 64, 64), epochs=12 if fast else 30, batch_size=256 if fast else 1024, n_batches=n_batches, verbose=progress)
        sol = scc.HybridNowSolver(cfg).solve(problem)
        stat = scc.policy_value_mc(problem, sol, x0=x0, n_paths=1000 if fast else 5000, seed=1234, verbose=progress)
        rel = abs(stat['mean'] - bench) / max(abs(bench), 1e-12)
        train_size = cfg.epochs * cfg.n_batches_per_epoch * cfg.batch_size
        rows.append({'d': d, 'train_size': train_size, 'rel_error': rel, 'estimate': stat['mean'], 'bench': bench})
        xvals.append(train_size)
        yvals.append(rel)
    plot_rel_error_curve(xvals, yvals, fig_dir / 'figure1_semilinear_relative_error.png', f'Figure 1 — Relative error w.r.t. size of training set (d={d})')
    return pd.DataFrame(rows)


def figure2_trajectories(fast: bool, fig_dir: Path, progress: bool = True) -> pd.DataFrame:
    problem = scc.make_test1_semilinear_problem(d=2, T=1.0, N=20)
    cfg = make_cfg(hidden=(32, 32), epochs=15 if fast else 35, batch_size=256 if fast else 1024, n_batches=8 if fast else 16, verbose=progress)
    sol = scc.HybridNowSolver(cfg).solve(problem)
    roll_opt = scc.rollout_policy(problem, sol, x0=np.zeros(2, dtype=np.float32), n_paths=5, seed=1234, verbose=progress, desc='rollout semilinear opt')
    roll_zero = scc.rollout_policy(problem, zero_solution(problem), x0=np.zeros(2, dtype=np.float32), n_paths=5, seed=1234, verbose=progress, desc='rollout semilinear alpha=0')
    plot_overlay_first_component({'Hybrid-Now (opt)': roll_opt, 'α=0 (bench)': roll_zero}, fig_dir / 'figure2_semilinear_trajectories.png', 'Figure 2 — Five forward simulations of the first component of X')
    return pd.DataFrame([{
        'cost_opt': roll_opt['mean'],
        'std_opt': roll_opt['std'],
        'cost_zero': roll_zero['mean'],
        'std_zero': roll_zero['std'],
    }])


def table1_gamma_sweep(fast: bool, out_dir: Path, progress: bool = True) -> pd.DataFrame:
    cfg_t2 = make_cfg(hidden=(10, 5, 5), epochs=20 if fast else 40, batch_size=256 if fast else 1024, n_batches=8 if fast else 16, verbose=progress)
    q_gh = sce.best_quantizer(K=11 if fast else 21, dim=1)
    gammas = [1.0, 0.5, 0.1, 0.0]
    N = 20 if fast else 40
    rows = []
    for gamma in scc.progress_iter(gammas, desc='Semilinear gamma sweep', enabled=progress, total=len(gammas)):
        prob = sce.make_test2_lipschitz_problem(gamma=gamma, N=N, N_slope=float(N), train_domain=(-1.5, 2.0))
        def g_orig(x):
            x1 = x[:, 0]
            out = torch.zeros_like(x1)
            m1 = (x1 >= 0.0) & (x1 <= 1.0)
            m2 = x1 >= 1.0
            if gamma == 0.0:
                out[m1] = -1.0
            else:
                out[m1] = -(torch.clamp(x1[m1], min=1e-12) ** gamma)
            out[m2] = -1.0
            return out
        bench_orig = float(scc.semilinear_closed_form_mc(g_orig, np.zeros((1, 1), dtype=np.float32), t=0.0, T=1.0, n_mc=10000 if fast else 30000)[0])
        hn = scc.HybridNowSolver(cfg_t2).solve(prob)
        hlq = scc.HybridLaterQSolver(cfg_t2, q_gh).solve(prob)
        grids = scc.make_state_grids_from_sampler(prob, n_points=100 if fast else 200, seed=1234)
        qknn = scc.QknnSolver(scc.QknnConfig(state_grids=grids, action_candidates=np.linspace(-3.0, 3.0, 31 if fast else 61).reshape(-1, 1).astype(np.float32), k_neighbors=2, verbose=progress), q_gh).solve(prob)
        yr = sce.YRSolver(sce.YRConfig(n_paths=10000 if fast else 50000, poly_degree=5, verbose=progress)).solve(prob)
        x00 = torch.zeros(1, 1)
        rows.append({
            'gamma': gamma,
            'Y&R': float(yr.value(0, x00).item()),
            'Hybrid-LaterQ': float(hlq.value(0, x00).item()),
            'Hybrid-Now': float(hn.value(0, x00).item()),
            'Qknn': float(qknn.value(0, x00).item()),
            'Bench_orig': bench_orig,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'exp1_table1_gamma.csv', index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(fast=True)
    parser.add_argument('--fast', action='store_true')
    parser.add_argument('--no-fast', dest='fast', action='store_false')
    parser.add_argument('--out', type=str, default=str(ROOT / 'results'))
    parser.add_argument('--progress', dest='progress', action='store_true', default=True, help='Show tqdm progress bars.')
    parser.add_argument('--no-progress', dest='progress', action='store_false', help='Disable tqdm progress bars.')
    args = parser.parse_args()

    scc.set_seed(1234)
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(Path(out_dir) / 'figures')

    t0 = perf_counter()
    df1 = figure1_sweep(args.fast, fig_dir, progress=args.progress)
    df1.to_csv(Path(out_dir) / 'exp1_figure1_curve.csv', index=False)
    df2 = figure2_trajectories(args.fast, fig_dir, progress=args.progress)
    df2.to_csv(Path(out_dir) / 'exp1_figure2_stats.csv', index=False)
    df3 = table1_gamma_sweep(args.fast, Path(out_dir), progress=args.progress)
    print(df3.to_string(index=False))
    print(f'Wrote Section 3.1 outputs to {out_dir} in {perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
