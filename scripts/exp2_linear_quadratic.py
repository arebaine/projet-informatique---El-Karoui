from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))

import stochastic_control_core as scc
from paper_plot_utils import ensure_dir, plot_lq_forward_components, plot_lq_policy_curves, plot_lq_value_curves, zero_solution


def cfg_for_dim(d: int, fast: bool, progress: bool = True) -> scc.NeuralSolverConfig:
    hidden = (d + 20, d + 10) if d <= 20 else (64, 64)
    return scc.NeuralSolverConfig(
        hidden_sizes=hidden,
        epochs=15 if fast else 50,
        n_batches_per_epoch=8 if fast else 20,
        batch_size=256 if fast else 1024,
        learning_rate=5e-3 if fast else 1e-3,
        l2_reg=1e-4 if fast else 1e-2,
        verbose=progress,
        seed=1234,
    )


def generate_lq_figures(fast: bool, fig_dir: Path, out_dir: Path, progress: bool = True) -> pd.DataFrame:
    rows = []

    # Figures 3 and 4: d=1 using Hybrid-Now, as in the paper.
    prob1 = scc.make_lq_problem(d=1, N=20)
    cfg1 = cfg_for_dim(1, fast, progress=progress)
    hn1 = scc.HybridNowSolver(cfg1).solve(prob1)
    x_grid = np.linspace(-3, 4, 121).astype(np.float32)
    plot_lq_policy_curves(hn1, x_grid, fig_dir / 'figure3_lq_policy.png', 'Figure 3 — Optimal decision estimated by Hybrid-Now')
    plot_lq_value_curves(hn1, x_grid, fig_dir / 'figure4_lq_value.png', 'Figure 4 — Value function estimated by Hybrid-Now')

    # Figure 5: d=10 forward simulation under Hybrid-Now with bench alpha=0.
    prob10 = scc.make_lq_problem(d=10, N=20)
    cfg10 = cfg_for_dim(10, fast, progress=progress)
    hn10 = scc.HybridNowSolver(cfg10).solve(prob10)
    roll_opt = scc.rollout_policy(prob10, hn10, x0=np.ones(10, dtype=np.float32), n_paths=1, seed=1234, verbose=progress, desc='rollout LQ opt')
    roll_bench = scc.rollout_policy(prob10, zero_solution(prob10), x0=np.ones(10, dtype=np.float32), n_paths=1, seed=1234, verbose=progress, desc='rollout LQ alpha=0')
    plot_lq_forward_components(roll_opt, roll_bench, fig_dir / 'figure5_lq_forward.png', 'Figure 5 — Forward simulation of X driven optimally using Hybrid estimates')

    # Tables 2 and 3 style outputs.
    configs = [(10, 1.0)] if fast else [(10, 1.0), (100, 0.1), (100, 0.5)]
    for d, x0_scale in scc.progress_iter(configs, desc='LQ table cases', enabled=progress, total=len(configs)):
        prob = scc.make_lq_problem(d=d, N=20)
        cfg = cfg_for_dim(d, fast, progress=progress)
        exact = scc.solve_discrete_lq_exact(prob)
        hn = scc.HybridNowSolver(cfg).solve(prob)
        nnpi = None
        if d <= 10:
            nnpi = scc.NNContPISolver(cfg).solve(prob)
        x0 = x0_scale * np.ones(d, dtype=np.float32)
        n_paths = 1000 if fast else 10000
        v_hn = scc.policy_value_mc(prob, hn, x0=x0, n_paths=n_paths, seed=1234, verbose=progress)['mean']
        v_ric = float(x0 @ exact['K'][0] @ x0)
        row = {'d': d, 'x0_scale': x0_scale, 'Hybrid-Now': v_hn, 'Riccati': v_ric}
        if nnpi is not None:
            row['NNContPI'] = scc.policy_value_mc(prob, nnpi, x0=x0, n_paths=n_paths, seed=1234, verbose=progress)['mean']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'exp2_tables.csv', index=False)
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
    df = generate_lq_figures(args.fast, fig_dir, Path(out_dir), progress=args.progress)
    print(df.to_string(index=False))
    print(f'Wrote Section 3.2 outputs to {out_dir} in {perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
