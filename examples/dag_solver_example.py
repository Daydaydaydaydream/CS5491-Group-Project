#!/usr/bin/env python
"""
dag_solver_example.py -- bare-bones reference for the Auto-BTP assignment.

It is the example block of Section 3 of the draft (see the figure there), plus
a brute-force search for the best placement, `solve()`.

Scoring is not written here.  `evaluate_dag()`, `depth_of()` and `budget_of()`
of `dag_builder.py` score the placement; the DAG of a placement is this block
with one inserted `Bootstrap1` node on every bootstrapped edge (`get_inserted`),
and the latency table of the block is passed in.

Three places to edit, everything else is reusable:

    latency_*   the estimated latency of one module (a plain number),
    LATENCY     which module type (second column of NODES) uses which function,
    NODES/EDGES the block itself.

Run:  python dag_solver_example.py                 # L_init = L_max = 14 and 10
      python dag_solver_example.py --l-max 14      # one instance only
      python dag_solver_example.py --l-init 14 --l-max 10
"""

import argparse
import itertools
import math

import dag_builder as db

# =========================================================================== #
#  1. estimated latency of ONE module (a plain number), given the level of its input
#     (a Bootstrapped edge is charged once per ciphertext, in `dag_builder.py`)
# =========================================================================== #
def latency_bootstrap(l_max):
    """cost of refreshing one ciphertext back to level l_max"""
    return 3.41 * math.exp(0.18 * l_max) + 4.81


def latency_linear_transform(level):
    """ConvBN: linear in the input level"""
    return 0.144 * level


def latency_mul(level):
    return 0.0


def latency_add(level):
    return 0.0


def latency_polyeval(level):
    return 0.0


def latency_identity(level):
    return 0.0


#   module type -> latency function
LATENCY = {
    "LinearTransform": latency_linear_transform,
    "Mul":             latency_mul,
    "Add":             latency_add,
    "PolyEval":        latency_polyeval,
    "Identity":        latency_identity,
}


def latency_table(l_max):
    """`LATENCY` plus the `Bootstrap1` of this instance.

    `dag_builder.evaluate_dag` calls every latency function with the *input
    level* of its node.  A `Bootstrap1` here is refreshed to `l_max` whatever
    level it reads, so its cost is a function of `l_max` and not of the level it
    is called with: the closure pins `l_max`.  A cost that did depend on the
    input level would be written `lambda l: ...` and need no closure.
    """
    return dict(LATENCY, Bootstrap1=lambda l, lm=l_max: latency_bootstrap(lm))


# =========================================================================== #
#  2. the block: (module, type, level cost, desc), in topological order
# =========================================================================== #
NODES = [
    ("X_in",    "Identity",        0, ""),
    ("id1",     "Identity",        0, ""),
    ("convbn1", "LinearTransform", 1, ""),
    ("affine1", "Mul",             1, ""),
    ("relu1",   "PolyEval",       10, ""),
    ("convbn2", "LinearTransform", 1, ""),
    ("add",     "Add",             0, ""),
    ("affine2", "Mul",             1, ""),
    ("relu2",   "PolyEval",       10, ""),
    ("out",     "Identity",        0, ""),
]

#   (from, to, #cipher);  ("id1", "add") is the skip connection
EDGES = [
    ("X_in",    "id1",     1),
    ("id1",     "convbn1", 1),
    ("convbn1", "affine1", 1),
    ("affine1", "relu1",   1),
    ("relu1",   "convbn2", 1),
    ("convbn2", "add",     1),
    ("id1",     "add",     1),
    ("add",     "affine2", 1),
    ("affine2", "relu2",   1),
    ("relu2",   "out",     1),
]

EDGE = [(u, v) for (u, v, _) in EDGES]                        # edge -> (from, to)


# =========================================================================== #
#  3. the placed DAG and the search
# =========================================================================== #
def get_inserted(booted):
    """The block with a `Bootstrap1` inserted on every edge of `booted`.

    `booted` = set of edges (from, to).  Returns (NODES, EDGES) of the inserted
    DAG, in the shape `dag_builder.py` reads.  Deterministic: the same `booted`
    always gives the same DAG.
    """
    nodes = list(NODES)
    edges = []
    for (u, v, ciphers) in EDGES:
        if (u, v) in booted:
            boot = f"Boot1_{u}_{v}"
            nodes.append((boot, "Bootstrap1", 0, f"Boot1({u} -> {v})"))
            edges += [(u, boot, ciphers), (boot, v, ciphers)]
        else:
            edges.append((u, v, ciphers))
    return nodes, edges


def solve(l_init, l_max):
    """Return the best placement and every placement that ties with it.

    A placement is a set of edges carrying a `Bootstrap1`; its latency, its
    levels and its depth and budget come from `dag_builder.py`.

    Returns (latency, boots) of the cheapest placement, and the list of the
    placements that tie with it, or (None, []) if nothing is feasible.
    """
    latency_of = latency_table(l_max)

    best, ties = None, []
    for r in range(len(EDGE) + 1):
        for combo in itertools.combinations(EDGE, r):
            booted = set(combo)
            nodes, edges = get_inserted(booted)
            res = db.evaluate_dag(nodes, edges, l_init, l_max, latency_of)
            if res is None:                           # underflow -> infeasible
                continue
            sol = (res["latency"], sorted(booted))
            if best is None or sol[0] < best[0] - 1e-9:
                best, ties = sol, [sol]
            elif abs(sol[0] - best[0]) < 1e-9:
                ties.append(sol)
    return best, ties


def report(latency, boots, l_init, l_max):
    """the numbers of one placement"""
    nodes, edges = get_inserted(set(boots))
    res = db.evaluate_dag(nodes, edges, l_init, l_max, latency_table(l_max))
    l_in, l_out = res["l_in"], res["l_out"]

    n_b1, _, budget = db.budget_of(nodes, edges, l_init, l_max)

    lines = [f"L_init = {l_init},  L_max = {l_max},  "
             f"t_boot = {latency_bootstrap(l_max):.4f}",
             "",
             f"minimum estimated latency = {latency:.4f}",
             f"#B (number of Bootstraps) = {n_b1}"]
    lines += [f"  Bootstrap on {u} -> {v}" for (u, v) in boots] or ["  (none)"]
    lines += ["",
              f"  {'module':<9} {'d_v':>4} {'l_in':>5} {'l_out':>6} {'latency':>12}",
              "  " + "-" * 41]
    for (name, op, cost, _) in NODES:
        lines.append(f"  {name:<9} {cost:>4} {l_in[name]:>5} "
                     f"{l_out[name]:>6} {LATENCY[op](l_in[name]):>12.4f}")
    depth = db.depth_of(nodes, edges)
    lines += ["",
              f"  depth = {depth}   budget = {budget}   "
              f"utilization = {depth / budget:.6f}"]
    return "\n".join(lines)


# =========================================================================== #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l-init", type=int, default=None,
                    help="default: L_init = L_max")
    ap.add_argument("--l-max", type=int, default=None,
                    help="if omitted, both 14 and 10 are solved")
    args = ap.parse_args()

    if args.l_max is None and args.l_init is None:
        instances = [(14, 14), (10, 10)]
    else:
        l_max = args.l_max if args.l_max is not None else args.l_init
        l_init = args.l_init if args.l_init is not None else l_max
        instances = [(l_init, l_max)]

    for (l_init, l_max) in instances:
        best, ties = solve(l_init, l_max)
        print("=" * 76)
        if best is None:
            print(f"L_init = {l_init}, L_max = {l_max}: no feasible placement")
        else:
            latency, boots = best
            print(report(latency, boots, l_init, l_max))
            print(f"\n  optimal placements: {len(ties)}")
            for sol in ties:
                print("    {" + ", ".join(f"{u} -> {v}" for (u, v) in sol[1]) + "}")
        print()


if __name__ == "__main__":
    main()
