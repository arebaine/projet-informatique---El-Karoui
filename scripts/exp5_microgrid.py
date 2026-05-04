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
import stochastic_control_extensions as sce
from paper_plot_utils import ensure_dir, plot_micro_boundary_panels, plot_micro_scatter_panels, plot_micro_trajectories


def _micro_action_candidates(problem: scc.MicrogridProblem, fast: bool, smoke: bool = False) -> np.ndarray:
    """Candidate set for Qknn.

    The admissible set is {0} U [A_min, A_max].  The grid is denser near the
    economically relevant power levels and lighter in the far tail up to Amax.
    """
    return np.unique(
        np.concatenate(
            [
                np.array([0.0], dtype=np.float32),
                np.linspace(problem.A_min, 2.0, 15 if smoke else (55 if fast else 81), dtype=np.float32),
                np.linspace(2.25, problem.A_max, 3 if smoke else (13 if fast else 18), dtype=np.float32),
            ]
        )
    ).reshape(-1, 1).astype(np.float32)


def _micro_r_bounds(problem: scc.MicrogridProblem, n_list) -> dict[int, tuple[float, float]]:
    """Paper-like R windows for Figures 12/13."""
    bounds: dict[int, tuple[float, float]] = {}
    for n in n_list:
        if int(n) == 1:
            bounds[int(n)] = (-0.6, 1.0)
        elif int(n) == 10:
            bounds[int(n)] = (-1.0, 1.8)
        else:
            bounds[int(n)] = (-1.5, 1.5)
    return bounds


def _micro_classif_clouds(
    problem: scc.MicrogridProblem,
    n_list,
    m_values,
    n_points: int = 3500,
    seed: int = 1234,
) -> dict[tuple[int, float], np.ndarray]:
    """Scatter clouds used by Figure 13.

    C is uniform on [0,Cmax], while R follows the marginal AR(1) law at date n
    and is clipped to the figure window.  This avoids a misleading full
    rectangular heatmap for the neural classifier.
    """
    rng = np.random.default_rng(seed)
    bounds = _micro_r_bounds(problem, n_list)
    clouds: dict[tuple[int, float], np.ndarray] = {}
    for n in n_list:
        mean = problem.R_bar + (problem.rho ** n) * (problem.r0 - problem.R_bar)
        if n == 0:
            std = 1e-4
        else:
            var = problem.sigma_R ** 2 * (1.0 - problem.rho ** (2 * n)) / max(1.0 - problem.rho ** 2, 1e-8)
            std = float(np.sqrt(max(var, 1e-12)))
        r_low, r_high = bounds[int(n)]
        for m in m_values:
            C = rng.uniform(0.0, problem.C_max, size=n_points).astype(np.float32)
            R = rng.normal(mean, std, size=n_points).astype(np.float32)
            R = np.clip(R, r_low, r_high).astype(np.float32)
            M = np.full(n_points, float(m), dtype=np.float32)
            clouds[(int(n), float(m))] = np.column_stack([C, M, R]).astype(np.float32)
    return clouds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(fast=True)
    parser.add_argument('--fast', action='store_true', help='Reduced but meaningful run.')
    parser.add_argument('--no-fast', dest='fast', action='store_false', help='Paper-scale grids/training.')
    parser.add_argument('--smoke', action='store_true', help='Very small execution-only test; do not use its neural numbers.')
    parser.add_argument('--out', type=str, default=str(ROOT / 'results'))
    parser.add_argument('--progress', dest='progress', action='store_true', default=True, help='Show tqdm progress bars.')
    parser.add_argument('--no-progress', dest='progress', action='store_false', help='Disable tqdm progress bars.')
    parser.add_argument('--fast-epochs', type=int, default=5, help='Number of epochs used by --fast ClassifHybrid.')
    parser.add_argument('--fast-batches', type=int, default=45, help='Mini-batches per epoch used by --fast ClassifHybrid.')
    parser.add_argument('--fast-batch-size', type=int, default=768, help='Mini-batch size used by --fast ClassifHybrid.')
    parser.add_argument(
        '--qknn-only',
        action='store_true',
        help='Quick diagnostic: skip the neural ClassifHybrid solver.',
    )
    parser.add_argument('--fig-points', type=int, default=3500, help='Points per panel for the ClassifHybrid Figure 13 scatter plot.')
    parser.add_argument('--fast-long-qknn-C', type=int, default=31, help='C-grid size for fast Figure 14 Qknn with N=200,Cmax=4.')
    parser.add_argument('--fast-long-qknn-R', type=int, default=31, help='R-grid size for fast Figure 14 Qknn with N=200,Cmax=4.')
    parser.add_argument('--fast-long-quantizer', type=int, default=11, help='Noise quantizer size for fast Figure 14 Qknn.')
    args = parser.parse_args()

    run_mode = 'smoke' if args.smoke else ('fast' if args.fast else 'paper')

    scc.set_seed(1234)
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(Path(out_dir) / 'figures')
    t0 = perf_counter()

    # Figures 12 and 13 with N = 30, Cmax = 1.
    N_small = 30
    prob = scc.make_microgrid_problem(N=N_small, C_max=1.0)

    if run_mode == 'smoke':
        cfg = scc.NeuralSolverConfig(hidden_sizes=(20, 20), epochs=1, n_batches_per_epoch=1, batch_size=128, learning_rate=3e-3, l2_reg=1e-4, verbose=args.progress, seed=1234, device='cpu')
        print('WARNING: --smoke is only an execution test. Neural estimates/figures are not numerically meaningful.')
    elif run_mode == 'fast':
        # Fast mode is tuned to recover Figures 12--14 more faithfully while
        # remaining lighter than the paper-scale run.
        cfg = scc.NeuralSolverConfig(hidden_sizes=(32, 32), epochs=args.fast_epochs, n_batches_per_epoch=args.fast_batches, batch_size=args.fast_batch_size, learning_rate=8.0e-4, l2_reg=1e-4, verbose=args.progress, seed=1234, device='cpu')
        approx = args.fast_epochs * args.fast_batches * args.fast_batch_size
        print(f'NOTE: --fast ClassifHybrid uses about {approx:,} resampled states/date and is tuned for Figures 12-14.')
    else:
        cfg = scc.NeuralSolverConfig(hidden_sizes=(32, 32), epochs=8, n_batches_per_epoch=60, batch_size=1024, learning_rate=8e-4, l2_reg=1e-4, verbose=args.progress, seed=1234, device='cpu')
    ch = None if args.qknn_only else scc.ClassifHybridSolver(cfg).solve(prob)

    grids = sce.paper_style_grids_for_microgrid(
        prob,
        n_C=7 if args.smoke else (71 if args.fast else 81),
        n_R=7 if args.smoke else (71 if args.fast else 81),
    )
    q_base = sce.best_quantizer(K=5 if args.smoke else (31 if args.fast else 21), dim=1)
    q_micro = sce.scale_quantizer(q_base, prob.sigma_R)
    qknn = scc.QknnSolver(
        scc.QknnConfig(
            state_grids=grids,
            action_candidates=_micro_action_candidates(prob, args.fast, args.smoke),
            k_neighbors=1,
            verbose=args.progress,
        ),
        q_micro,
    ).solve(prob)

    n_list = [1, 10, 28]
    m_values = [0.0, 1.0]
    c_vals = np.linspace(0.0, prob.C_max, 120)
    r_vals = np.linspace(-1.6, 1.8, 120)  # resolution template; window is per n
    r_bounds = _micro_r_bounds(prob, n_list)
    plot_micro_boundary_panels(
        qknn,
        n_list,
        m_values,
        c_vals,
        r_vals,
        fig_dir / 'figure12_microgrid_qknn.png',
        'Figure 12 — Estimated optimal decisions using Qknn',
        r_bounds=r_bounds,
    )
    if ch is not None:
        classif_clouds = _micro_classif_clouds(prob, n_list, m_values, n_points=args.fig_points, seed=1234)
        plot_micro_scatter_panels(
            ch,
            n_list,
            m_values,
            classif_clouds,
            fig_dir / 'figure13_microgrid_classifhybrid.png',
            'Figure 13 — Estimated optimal decisions using ClassifHybrid',
            r_bounds=r_bounds,
        )

    # Figure 14 must keep the paper's horizon/capacity.  In --fast we use a
    # coarser Qknn grid/quantizer, but still plot N=200 and Cmax=4.
    N_long = 10 if args.smoke else 200
    prob_long = scc.make_microgrid_problem(N=N_long, C_max=1.0 if args.smoke else 4.0)
    grids_long = sce.paper_style_grids_for_microgrid(
        prob_long,
        n_C=7 if args.smoke else (args.fast_long_qknn_C if args.fast else 51),
        n_R=7 if args.smoke else (args.fast_long_qknn_R if args.fast else 51),
    )
    q_base_long = sce.best_quantizer(K=5 if args.smoke else (args.fast_long_quantizer if args.fast else 21), dim=1)
    q_micro_long = sce.scale_quantizer(q_base_long, prob_long.sigma_R)
    qknn_long = scc.QknnSolver(
        scc.QknnConfig(
            state_grids=grids_long,
            action_candidates=_micro_action_candidates(prob_long, args.fast, args.smoke),
            k_neighbors=1,
            verbose=args.progress,
        ),
        q_micro_long,
    ).solve(prob_long)
    x0 = np.array([prob_long.c0, 0.0, prob_long.r0], dtype=np.float32)
    roll = scc.rollout_policy(prob_long, qknn_long, x0=x0, n_paths=2, seed=1234, verbose=args.progress, desc='rollout microgrid trajectories')
    plot_micro_trajectories(
        roll,
        fig_dir / 'figure14_microgrid_trajectories.png',
        f'Figure 14 — Two simulations of (C, M, R) optimally controlled using Qknn (N={N_long})',
        n_paths=2,
    )

    stats = []
    n_mc = 1000 if args.smoke else 10000
    if ch is not None:
        vals = scc.policy_value_mc(
            prob,
            ch,
            x0=np.array([prob.c0, 0.0, prob.r0], dtype=np.float32),
            n_paths=n_mc,
            seed=1234,
            device='cpu',
            verbose=args.progress,
        )
        stats.append({'method': 'ClassifHybrid', 'mean': vals['mean'], 'std': vals['std']})
    vals = scc.policy_value_mc(
        prob,
        qknn,
        x0=np.array([prob.c0, 0.0, prob.r0], dtype=np.float32),
        n_paths=n_mc,
        seed=1234,
        device='cpu',
        verbose=args.progress,
    )
    stats.append({'method': 'Qknn', 'mean': vals['mean'], 'std': vals['std']})
    df = pd.DataFrame(stats)
    df.to_csv(Path(out_dir) / 'exp5_microgrid_stats.csv', index=False)
    print(df.to_string(index=False))
    print(f'Wrote Section 3.5 outputs to {out_dir} in {perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
