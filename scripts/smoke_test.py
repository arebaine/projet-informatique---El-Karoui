from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

import stochastic_control_core as scc
import stochastic_control_extensions as sce
import exp1_semilinear_pde as e1
import exp2_linear_quadratic as e2
import exp3_option_hedging as e3
import exp4_gas_storage as e4
import exp5_microgrid as e5


def main() -> None:
    # Import sanity.
    mods = [scc, sce, e1, e2, e3, e4, e5]
    print('Imported modules:', ', '.join(m.__name__ for m in mods))

    # Regression test for the gamma=1 overflow bug.
    g1 = sce.lipschitz_terminal_cost_test2(gamma=1.0, N_slope=40.0)
    x = torch.tensor([[-1.0], [0.0], [0.5], [1.0], [2.0]], dtype=torch.float32)
    y = g1(x).detach().cpu().numpy().tolist()
    print('gamma=1 terminal values:', y)
    flat = [v[0] if isinstance(v, list) else v for v in y]
    assert flat == [0.0, 0.0, -0.5, -1.0, -1.0], f'unexpected gamma=1 profile: {flat}'

    # Basic object construction from each experiment family.
    p1 = scc.make_test1_semilinear_problem(d=2, T=1.0, N=4)
    p2 = scc.make_lq_problem(d=1, N=4)
    p3 = scc.make_option_hedging_problem(N=3)
    p4 = scc.make_gas_storage_problem(N=4)
    p5 = scc.make_microgrid_problem(N=4)
    print('Constructed problems:', p1.name, p2.name, p3.name, p4.name, p5.name)

    print('Smoke test passed.')


if __name__ == '__main__':
    main()
