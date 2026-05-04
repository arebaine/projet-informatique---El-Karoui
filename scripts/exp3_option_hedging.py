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
from paper_plot_utils import (
    ExactOptionPolicy,
    ensure_dir,
    eval_value,
    plot_hedging_paths,
    plot_hedging_value_curve,
    rollout_on_fixed_returns,
    shared_returns_matrix,
)


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

    # Problem and exact benchmark
    prob = scc.make_option_hedging_problem(
        N=6,
        p0=100.0,
        strike=100.0,
        r_plus=0.05,
        r_minus=-0.05,
        pi_plus=0.6,
        pi_minus=0.3,
    )

    exact = scc.solve_option_hedging_exact(
        N=6,
        p0=100.0,
        strike=100.0,
        r_plus=0.05,
        r_minus=-0.05,
        pi_plus=0.6,
        pi_minus=0.3,
    )

    exact_sol = ExactOptionPolicy(exact, prob)

    q_trin = scc.Quantizer(
        points=np.array(prob.returns, dtype=np.float32).reshape(-1, 1),
        weights=np.array(prob.return_probs, dtype=np.float32),
        dim=1,
    )

    cfg = scc.NeuralSolverConfig(
    hidden_sizes=(32, 32) if args.fast else (64, 64),
    epochs=80 if args.fast else 200,
    n_batches_per_epoch=20 if args.fast else 50,
    batch_size=1024 if args.fast else 4096,
    learning_rate=1e-3,
    l2_reg=1e-6,
    grad_clip=1.0,
    verbose=args.progress,
    seed=1234,)

    # Train algorithms
    hn = None
    hlq = None
    for label in scc.progress_iter(['Hybrid-Now', 'Hybrid-LaterQ'], desc='Hedging neural solvers', enabled=args.progress, total=2):
        if label == 'Hybrid-Now':
            hn = scc.HybridNowSolver(cfg).solve(prob)
        else:
            hlq = scc.HybridLaterQSolver(cfg, q_trin).solve(prob)
    assert hn is not None and hlq is not None

    # ------------------------------------------------------------
    # Figure 6: value function at time 0 as a function of w0
    # ------------------------------------------------------------
    w_grid = np.linspace(0.0, 10.0, 121).astype(np.float32)
    p_grid = np.full_like(w_grid, prob.p0, dtype=np.float32)
    sg = np.stack([w_grid, p_grid], axis=1).astype(np.float32)

    exact_v = exact['value'](0, w_grid, p_grid)
    hn_v = eval_value(hn, 0, sg)
    hlq_v = eval_value(hlq, 0, sg)

    plot_hedging_value_curve(
        w_grid,
        exact_v,
        hn_v,
        hlq_v,
        fig_dir / 'figure6_hedging_value_curve.png',
        'Figure 6 — Estimates of the value function at time 0 w.r.t. $w_0$',
    )

    # ------------------------------------------------------------
    # Figure 7: three shared scenarios at w0 = 100
    # This is only for reproducing the paper-style path figure.
    # It is not the hedging-price experiment.
    # ------------------------------------------------------------
    returns_matrix = shared_returns_matrix(prob, n_paths=3, seed=1234)

    wealth_opt = rollout_on_fixed_returns(
        prob,
        exact_sol,
        w0=100.0,
        returns_matrix=returns_matrix,
    )['wealth']

    wealth_hn = rollout_on_fixed_returns(
        prob,
        hn,
        w0=100.0,
        returns_matrix=returns_matrix,
    )['wealth']

    wealth_hlq = rollout_on_fixed_returns(
        prob,
        hlq,
        w0=100.0,
        returns_matrix=returns_matrix,
    )['wealth']

    plot_hedging_paths(
        {
            'Opt': wealth_opt,
            'Hybrid-Now': wealth_hn,
            'Hybrid-LaterQ': wealth_hlq,
        },
        fig_dir / 'figure7_hedging_paths.png',
        'Figure 7 — Three simulations of the agent’s wealth w.r.t. time n',
    )

    # ------------------------------------------------------------
    # Correct hedging-price estimation
    # ------------------------------------------------------------
    # Important:
    # The old version copied exact['hedging_price'] for all methods.
    # This was wrong: Hybrid-Now and Hybrid-LaterQ must have their own
    # estimated price, obtained by minimizing their estimated V_0(w, p0).
    # ------------------------------------------------------------
    def estimate_price_from_value(label: str, sol, n_grid: int = 1001):
        w_test = np.linspace(0.0, 10.0, n_grid).astype(np.float32)
        p_test = np.full_like(w_test, prob.p0, dtype=np.float32)
        x_test = np.stack([w_test, p_test], axis=1).astype(np.float32)

        if label == 'Opt':
            values = exact['value'](0, w_test, p_test)
        else:
            values = eval_value(sol, 0, x_test)

        values = np.asarray(values).reshape(-1)
        idx = int(np.argmin(values))

        return float(w_test[idx]), float(values[idx])

    rows = []

    n_mc = 5000 if args.fast else 20000
    returns_eval = shared_returns_matrix(prob, n_paths=n_mc, seed=4321)

    methods = [
        ('Opt', exact_sol),
        ('Hybrid-Now', hn),
        ('Hybrid-LaterQ', hlq),
    ]
    for label, sol in scc.progress_iter(methods, desc='Hedging evaluation', enabled=args.progress, total=len(methods)):
        price_est, min_value_est = estimate_price_from_value(label, sol)

        out = rollout_on_fixed_returns(
            prob,
            sol,
            w0=price_est,
            returns_matrix=returns_eval,
        )

        rows.append(
            {
                'method': label,
                'hedging_price': price_est,
                'estimated_V0_min': min_value_est,
                'mean_loss': float(out['terminal_cost'].mean()),
                'std_loss': float(out['terminal_cost'].std()),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / 'exp3_hedging_stats.csv', index=False)

    print(df.to_string(index=False))
    print(f'Wrote Section 3.3 outputs to {out_dir} in {perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()