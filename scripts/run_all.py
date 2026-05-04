from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import stochastic_control_core as scc
import exp1_semilinear_pde as e1
import exp2_linear_quadratic as e2
import exp3_option_hedging as e3
import exp4_gas_storage as e4
import exp5_microgrid as e5


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
    overall_t0 = perf_counter()

    sections = [('Section 3.1 — Semilinear PDE', e1), ('Section 3.2 — LQ', e2), ('Section 3.3 — Hedging', e3), ('Section 3.4 — Gas Storage', e4), ('Section 3.5 — Microgrid', e5)]
    for name, module in scc.progress_iter(sections, desc='All paper experiments', enabled=args.progress, total=len(sections)):
        print('\n' + '#' * 72)
        print('# ' + name)
        print('#' * 72)
        t0 = perf_counter()
        sys.argv = [module.__name__, '--fast' if args.fast else '--no-fast', '--out', args.out]
        if not args.progress:
            sys.argv.append('--no-progress')
        module.main()
        print(f'--> done in {perf_counter() - t0:.1f}s')

    print('\n' + '=' * 72)
    print(f'All experiments finished in {perf_counter() - overall_t0:.1f}s')
    print(f'Results written to {args.out}')


if __name__ == '__main__':
    main()
