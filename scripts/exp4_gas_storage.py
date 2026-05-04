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
from paper_plot_utils import ensure_dir, plot_gas_ain_curve, plot_gas_decision_panels, zero_solution

def gas_panel_bounds(prob, n_list):
    bounds = {}

    for n in n_list:
        c_low = max(prob.C_min, prob.c0 - n * prob.aout)
        c_high = min(prob.C_max, prob.c0 + n * prob.ain)

        mean_p = prob.p_bar + (prob.beta ** n) * (prob.p0 - prob.p_bar)
        var_p = 0.0 if n == 0 else prob.sigma_p**2 * (1.0 - prob.beta ** (2 * n)) / (1.0 - prob.beta**2)
        std_p = np.sqrt(max(var_p, 1e-12))

        p_low = max(3.0, mean_p - 3.5 * std_p)
        p_high = min(7.0, mean_p + 3.5 * std_p)

        if p_high - p_low < 1.5:
            mid = 0.5 * (p_low + p_high)
            p_low = max(3.0, mid - 0.75)
            p_high = min(7.0, mid + 0.75)

        bounds[n] = (p_low, p_high, c_low, c_high)

    return bounds


def gas_scatter_clouds(prob, n_list, n_points=2500, seed=1234):
    rng = np.random.default_rng(seed)
    clouds = [None] * (prob.horizon + 1)

    bounds = gas_panel_bounds(prob, n_list)

    for n in n_list:
        p_low, p_high, c_low, c_high = bounds[n]

        mean_p = prob.p_bar + (prob.beta ** n) * (prob.p0 - prob.p_bar)
        var_p = 0.0 if n == 0 else prob.sigma_p**2 * (1.0 - prob.beta ** (2 * n)) / (1.0 - prob.beta**2)
        std_p = np.sqrt(max(var_p, 1e-12))

        P = rng.normal(mean_p, 2.0 * std_p, size=n_points)
        P = np.clip(P, p_low, p_high)

        C = rng.uniform(c_low, c_high, size=n_points)

        clouds[n] = np.column_stack([P, C]).astype(np.float32)

    return clouds

def _neural_configs(
    mode: str,
    fast_epochs: int = 6,
    fast_batches: int = 50,
    fast_batch_size: int = 768,
    hybrid_fast_epochs: int = 5,
    hybrid_fast_batches: int = 52,
    hybrid_fast_batch_size: int = 768,
    hybrid_train_noise: int = 5,
    hybrid_eval_noise: int = 7,
    progress: bool = True,
):
    classif_common = dict(
        hidden_sizes=(20, 20),
        activation='elu',
        l2_reg=1e-4,
        transfer_warm_start=True,
        verbose=progress,
        device='cpu',
        grad_clip=2.0,
    )
    # Hybrid-Now keeps the residual-value baseline, but --fast must remain
    # usable on Colab. A 24+24 residual net is a compromise; --no-fast below
    # still uses the larger 32+32+16 net.
    hybrid_common = dict(
        hidden_sizes=(24, 24),
        activation='elu',
        l2_reg=5e-5,
        transfer_warm_start=True,
        verbose=progress,
        device='cpu',
        grad_clip=1.0,
    )

    if mode == 'smoke':
        return (
            scc.NeuralSolverConfig(**classif_common, epochs=1, n_batches_per_epoch=2, batch_size=256, learning_rate=3e-3, seed=1234),
            scc.NeuralSolverConfig(**hybrid_common, epochs=1, n_batches_per_epoch=2, batch_size=256, learning_rate=1e-3, seed=4321, hybrid_train_noise=3, hybrid_eval_noise=5),
        )

    if mode == 'fast':
        return (
            scc.NeuralSolverConfig(**classif_common, epochs=fast_epochs, n_batches_per_epoch=fast_batches, batch_size=fast_batch_size, learning_rate=8e-4, seed=1234),
            scc.NeuralSolverConfig(**hybrid_common, epochs=hybrid_fast_epochs, n_batches_per_epoch=hybrid_fast_batches, batch_size=hybrid_fast_batch_size, learning_rate=3e-4, seed=4321, hybrid_train_noise=hybrid_train_noise, hybrid_eval_noise=hybrid_eval_noise),
        )

    # --no-fast : plus proche papier, avec Hybrid-Now renforcé.
    return (
        scc.NeuralSolverConfig(**classif_common, epochs=10, n_batches_per_epoch=120, batch_size=512, learning_rate=7e-4, seed=1234),
        scc.NeuralSolverConfig(hidden_sizes=(32, 32, 16), activation='elu', l2_reg=5e-5, transfer_warm_start=True, verbose=progress, device='cpu', grad_clip=1.0, epochs=14, n_batches_per_epoch=140, batch_size=768, learning_rate=2e-4, seed=4321, hybrid_train_noise=9, hybrid_eval_noise=15),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.set_defaults(fast=True)
    parser.add_argument('--fast', action='store_true', help='Reduced but meaningful run. Slower than a smoke test.')
    parser.add_argument('--no-fast', dest='fast', action='store_false', help='Paper-scale neural batches: about M=60,000 states per time step.')
    parser.add_argument('--smoke', action='store_true', help='Very small execution-only test; do not use its neural numbers.')
    parser.add_argument('--with-later', action='store_true', help='Also run the finite-action Hybrid-LaterQ diagnostic.')
    parser.add_argument('--qknn-only', action='store_true', help='Quick diagnostic: run only Qknn and the passive benchmark.')
    parser.add_argument('--progress', dest='progress', action='store_true', default=True, help='Show tqdm progress bars.')
    parser.add_argument('--no-progress', dest='progress', action='store_false', help='Disable tqdm progress bars.')
    parser.add_argument(
        '--figure-ain',
        type=float,
        default=0.20,
        help='ain used for Figures 9-11. The paper-like panel windows match ain=0.20; the table still reports all ain values.',
    )
    parser.add_argument('--out', type=str, default=str(ROOT / 'results'))
    parser.add_argument('--fast-epochs', type=int, default=6, help='Number of epochs used by --fast ClassifPI.')
    parser.add_argument('--fast-batches', type=int, default=50, help='Mini-batches per epoch used by --fast ClassifPI.')
    parser.add_argument('--fast-batch-size', type=int, default=768, help='Mini-batch size used by --fast ClassifPI.')
    parser.add_argument('--hybrid-fast-epochs', type=int, default=5, help='Number of epochs used by --fast Hybrid-Now.')
    parser.add_argument('--hybrid-fast-batches', type=int, default=52, help='Mini-batches per epoch used by --fast Hybrid-Now.')
    parser.add_argument('--hybrid-fast-batch-size', type=int, default=768, help='Mini-batch size used by --fast Hybrid-Now.')
    parser.add_argument('--hybrid-train-noise', type=int, default=5, help='Gauss-Hermite points used in Hybrid-Now training during --fast.')
    parser.add_argument('--hybrid-eval-noise', type=int, default=7, help='Gauss-Hermite points used in Hybrid-Now greedy policy/evaluation during --fast.')
    args = parser.parse_args()

    run_mode = 'smoke' if args.smoke else ('fast' if args.fast else 'paper')

    scc.set_seed(1234)
    out_dir = ensure_dir(args.out)
    fig_dir = ensure_dir(Path(out_dir) / 'figures')
    t0 = perf_counter()

    ain_list =  [0.06, 0.10, 0.20] if args.smoke else [0.06, 0.10, 0.20, 0.30, 0.40]
    figure_ain = float(args.figure_ain)

    cfg_classif, cfg_hybrid = _neural_configs(
        run_mode,
        fast_epochs=args.fast_epochs,
        fast_batches=args.fast_batches,
        fast_batch_size=args.fast_batch_size,
        hybrid_fast_epochs=args.hybrid_fast_epochs,
        hybrid_fast_batches=args.hybrid_fast_batches,
        hybrid_fast_batch_size=args.hybrid_fast_batch_size,
        hybrid_train_noise=args.hybrid_train_noise,
        hybrid_eval_noise=args.hybrid_eval_noise,
        progress=args.progress,
    )

    # 21 points is what the paper reports for the 1D normal quantizer.
    q_base = sce.best_quantizer(K=21, dim=1)

    rows = []
    qknn_vals, zero_vals = [], []
    chosen_bundle = None

    if run_mode == 'smoke' and not args.qknn_only:
        print('WARNING: --smoke is only an execution test. Neural estimates/figures are not numerically meaningful.')
    elif run_mode == 'fast' and not args.qknn_only:
        approx_classif = args.fast_epochs * args.fast_batches * args.fast_batch_size
        approx_hybrid = args.hybrid_fast_epochs * args.hybrid_fast_batches * args.hybrid_fast_batch_size
        print(
            f'NOTE: --fast ClassifPI uses about {approx_classif:,} states/date; '
            f'Hybrid-Now uses about {approx_hybrid:,} states/date (GH train/eval={args.hybrid_train_noise}/{args.hybrid_eval_noise}) with the stabilised residual baseline. '
            'Use --no-fast for longer paper-scale checks, or increase --hybrid-fast-* for a stronger Hybrid-Now run.'
        )

    for ain in scc.progress_iter(ain_list, desc='Gas ain values', enabled=args.progress, total=len(ain_list)):
        prob = scc.make_gas_storage_problem(N=30, ain=ain)

        grids = sce.paper_style_grids_for_gas_storage(
            prob,
            n_P=21 if args.smoke else (61 if args.fast else 81),
            n_C=31 if args.smoke else (81 if args.fast else 101),
            p0=4.0,
            reachable_inventory=True,
        )
        q_gas = sce.scale_quantizer(q_base, prob.sigma_p)

        qknn = scc.QknnSolver(
            scc.QknnConfig(
                state_grids=grids,
                action_candidates=np.array([[-1.0], [0.0], [1.0]], dtype=np.float32),
                k_neighbors=2,
                verbose=args.progress,
                device='cpu',
            ),
            q_gas,
        ).solve(prob)

        classif = None if args.qknn_only else scc.ClassifPISolver(cfg_classif).solve(prob)
        hnow = None if args.qknn_only else scc.DiscreteHybridNowSolver(cfg_hybrid).solve(prob)
        hlq = None
        if args.with_later and not args.qknn_only:
            hlq = scc.DiscreteHybridLaterQSolver(cfg_hybrid, q_gas).solve(prob)

        x0 = np.array([4.0, 4.0], dtype=np.float32)
        n_paths = 1000 if args.smoke else (15000 if args.fast else 50000)
        v_qknn = scc.rollout_policy(prob, qknn, x0=x0, n_paths=n_paths, seed=1234, device='cpu', verbose=args.progress, desc=f'rollout Qknn ain={ain:.2f}')['mean']
        v_class = np.nan if classif is None else scc.rollout_policy(prob, classif, x0=x0, n_paths=n_paths, seed=1234, device='cpu', verbose=args.progress, desc=f'rollout ClassifPI ain={ain:.2f}')['mean']
        v_hn = np.nan if hnow is None else scc.rollout_policy(prob, hnow, x0=x0, n_paths=n_paths, seed=1234, device='cpu', verbose=args.progress, desc=f'rollout Hybrid-Now ain={ain:.2f}')['mean']
        v_zero = scc.rollout_policy(prob, zero_solution(prob), x0=x0, n_paths=n_paths, seed=1234, device='cpu', verbose=args.progress, desc=f'rollout alpha=0 ain={ain:.2f}')['mean']

        row = {
            'ain': ain,
            'Hybrid-Now': -v_hn,
            'ClassifPI': -v_class,
            'Qknn': -v_qknn,
            'alpha=0': -v_zero,
        }
        if hlq is not None:
            row['Hybrid-LaterQ'] = -scc.rollout_policy(prob, hlq, x0=x0, n_paths=n_paths, seed=1234, device='cpu', verbose=args.progress, desc=f'rollout Hybrid-LaterQ ain={ain:.2f}')['mean']
        rows.append(row)
        qknn_vals.append(-v_qknn)
        zero_vals.append(-v_zero)

        if chosen_bundle is None or abs(ain - figure_ain) < abs(chosen_bundle['prob'].ain - figure_ain):
            chosen_bundle = {'prob': prob, 'Qknn': qknn, 'ClassifPI': classif, 'Hybrid-Now': hnow, 'grids': grids}

    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / 'exp4_table4.csv', index=False)
    plot_gas_ain_curve(
        ain_list,
        qknn_vals,
        zero_vals,
        fig_dir / 'figure8_gas_ain_curve.png',
        'Figure 8 — Estimate of the value function at time 0 w.r.t. $a_{in}$',
    )

    p_vals = np.linspace(3.0, 7.0, 220)
    c_vals = np.linspace(0.0, 8.0, 220)
    n_list = [5, 10, 15, 20, 25, 29]
    eval_grids = chosen_bundle['grids']
    plot_bounds = gas_panel_bounds(chosen_bundle["prob"], n_list)
    classif_clouds = gas_scatter_clouds(chosen_bundle["prob"], n_list, n_points=5000)
    
    plot_gas_decision_panels(
        chosen_bundle['Qknn'], n_list, p_vals, c_vals,
        fig_dir / 'figure9_gas_qknn.png',
        'Figure 9 — Estimated optimal decisions using Qknn',
        qknn_style=True,
        eval_grids=eval_grids,
        scatter=True,
        paper_bounds=plot_bounds,
    )
    if chosen_bundle['ClassifPI'] is not None:
        plot_gas_decision_panels(
            chosen_bundle["ClassifPI"],
            n_list,
            p_vals,
            c_vals,
            fig_dir / "figure10_gas_classifpi.png",
            "Figure 10 — Estimated optimal decisions using ClassifPI",
            qknn_style=False,
            eval_grids=classif_clouds,
            scatter=True,
            paper_bounds=plot_bounds,
        )
    if chosen_bundle['Hybrid-Now'] is not None:
        plot_gas_decision_panels(
            chosen_bundle["Hybrid-Now"],
            n_list,
            p_vals,
            c_vals,
            fig_dir / "figure11_gas_hybridnow.png",
            "Figure 11 — Estimated optimal decisions using Hybrid-Now",
            qknn_style=False,
            eval_grids=None,
            scatter=False,
            paper_bounds=plot_bounds,
            interpolation="bilinear",
        )

    print(df.to_string(index=False))
    print(f'Wrote Section 3.4 outputs to {out_dir} in {perf_counter()-t0:.1f}s')


if __name__ == '__main__':
    main()
