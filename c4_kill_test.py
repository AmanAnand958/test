"""
Quantum C4 kill test.

Question (per the revised novelty assessment): take a tiny circuit with
hidden-inverse choices, formulate its exact coherent-error objective, and
check whether existing extraction methods -- a Quasar-style additive/local
e-graph extractor, and a SmoothE-style differentiable relaxation over
arbitrary costs -- can already optimize it without modification.

Circuit model
-------------
A circuit has k "hidden-inverse" choice points. At choice point i we pick
x_i in {0, 1}: implement gate i as G or as its hidden inverse G^dagger.
Both give the same logical unitary; they differ in which physical pulse
sequence is used, hence in coherent-error behavior (Zhang et al. 2022).

Coherent-error objective (the thing C4 claims is special)
-----------------------------------------------------------
Per hidden-inverse theory, whether G or G^dagger is preferred at site i can
depend on *other gates in the vicinity* (Ferracin et al. and the 2024
cross-layer compiler paper both confirm this -- optimal choice is context
dependent, not globally fixed). We encode that directly as a cost with:

  - a local (additive) term per choice:      sum_i a_i * x_i
  - pairwise interaction terms for site pairs that are physically
    "nearby" in the pulse schedule (interference can cancel or compound):
                                              sum_{(i,j) in NEARBY} b_ij * x_i * x_j
  - pairwise interaction terms for site pairs that are physically FAR
    apart in the circuit (the disputed "non-local" claim):
                                              sum_{(i,j) in FAR} c_ij * x_i * x_j

True objective = local + nearby-interaction + far-interaction.
Exact optimum found by brute force over all 2^k assignments (k is small).

Two baselines, standing in for existing tools:

1. "Quasar-style" additive/local extraction -- bottom-up e-graph
   extraction assumes a cost that decomposes per e-class/e-node
   (i.e. it can represent the LOCAL term exactly, and, if given the
   NEARBY-interaction terms explicitly as part of a node's cost, can
   still greedily resolve them since they only couple adjacent choices
   in the DAG). It CANNOT natively represent the FAR-interaction terms,
   because bottom-up extraction cost is computed per subterm independent
   of the assignment made at some other unrelated subterm; the greedy
   proxy here picks each x_i by its local term plus only its NEARBY
   interactions, ignoring FAR ones entirely.

2. "SmoothE-style" differentiable relaxation -- p_i in [0, 1], objective
   evaluated on the exact analytic cost (which SmoothE can do, since it
   supports arbitrary differentiable f over e-node selection probabilities,
   including the FAR-interaction terms), optimized by gradient descent,
   then rounded.

If SmoothE-style already recovers the true optimum on instances with
strong FAR-interaction terms, C4 has no opening: "just use SmoothE with
this objective" kills it, exactly as flagged. If SmoothE-style regularly
lands in the wrong local optimum specifically when FAR-interaction terms
dominate (rugged, non-convex landscape from long-range coupling), while an
exact joint/structure-aware search does not, that is evidence the claimed
gap survives and C4 has a testable technical hook.
"""

import itertools
import numpy as np

RNG_SEED = 0
K = 8  # hidden-inverse choice points; 2^8 = 256, brute-forceable
N_TRIALS = 30
N_RESTARTS = 20
GD_STEPS = 300
LR = 0.3


def make_instance(seed, far_strength):
    rng = np.random.RandomState(seed)
    a = rng.uniform(-1, 1, size=K)  # local terms

    # "nearby" pairs: adjacent choice points in circuit order (i, i+1)
    nearby_pairs = [(i, i + 1) for i in range(K - 1)]
    b = {p: rng.uniform(-0.5, 0.5) for p in nearby_pairs}

    # "far" pairs: choice points separated by >= K//2 positions
    far_pairs = [
        (i, j) for i in range(K) for j in range(i + 1, K) if (j - i) >= K // 2
    ]
    c = {p: rng.uniform(-far_strength, far_strength) for p in far_pairs}

    return a, nearby_pairs, b, far_pairs, c


def exact_cost(x, a, nearby_pairs, b, far_pairs, c):
    total = np.dot(a, x)
    for (i, j), bij in b.items():
        total += bij * x[i] * x[j]
    for (i, j), cij in c.items():
        total += cij * x[i] * x[j]
    return total


def brute_force_optimum(a, nearby_pairs, b, far_pairs, c):
    best_cost = None
    best_x = None
    for bits in itertools.product([0, 1], repeat=K):
        x = np.array(bits, dtype=float)
        cost = exact_cost(x, a, nearby_pairs, b, far_pairs, c)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_x = x
    return best_x, best_cost


def quasar_style_extraction(a, nearby_pairs, b, far_pairs, c):
    """Local/additive extraction: resolve each x_i greedily using only its
    local term and NEARBY interactions (what a bottom-up e-graph cost
    function can represent), completely blind to FAR terms."""
    x = np.zeros(K)
    # simple left-to-right greedy pass: at each site, having fixed earlier
    # choices, pick x_i minimizing local + already-known nearby coupling
    for i in range(K):
        cost0 = a[i] * 0.0
        cost1 = a[i] * 1.0
        for (p, q), bij in b.items():
            if q == i and p < i:
                cost0 += bij * x[p] * 0.0
                cost1 += bij * x[p] * 1.0
        x[i] = 0.0 if cost0 <= cost1 else 1.0
    cost = exact_cost(x, a, nearby_pairs, b, far_pairs, c)
    return x, cost


def smoothe_style_extraction(a, nearby_pairs, b, far_pairs, c, seed):
    """Differentiable relaxation over the FULL exact cost (including far
    terms), optimized with gradient descent + sigmoid parametrization,
    multiple random restarts, then rounded to nearest binary point."""
    rng = np.random.RandomState(seed)

    def relaxed_cost(theta):
        p = 1.0 / (1.0 + np.exp(-theta))  # sigmoid relaxation, p in (0,1)
        total = np.dot(a, p)
        for (i, j), bij in b.items():
            total += bij * p[i] * p[j]
        for (i, j), cij in c.items():
            total += cij * p[i] * p[j]
        return total, p

    def grad(theta, eps=1e-4):
        g = np.zeros(K)
        base, _ = relaxed_cost(theta)
        for i in range(K):
            theta2 = theta.copy()
            theta2[i] += eps
            f2, _ = relaxed_cost(theta2)
            g[i] = (f2 - base) / eps
        return g

    best_x, best_cost = None, None
    for r in range(N_RESTARTS):
        theta = rng.uniform(-2, 2, size=K)
        for _ in range(GD_STEPS):
            g = grad(theta)
            theta -= LR * g
        _, p = relaxed_cost(theta)
        x = (p >= 0.5).astype(float)
        cost = exact_cost(x, a, nearby_pairs, b, far_pairs, c)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_x = x

    return best_x, best_cost


def run(far_strength):
    gaps_quasar = []
    gaps_smoothe = []
    exact_matches_quasar = 0
    exact_matches_smoothe = 0

    for t in range(N_TRIALS):
        a, nearby_pairs, b, far_pairs, c = make_instance(RNG_SEED + t, far_strength)
        _, opt_cost = brute_force_optimum(a, nearby_pairs, b, far_pairs, c)

        _, q_cost = quasar_style_extraction(a, nearby_pairs, b, far_pairs, c)
        _, s_cost = smoothe_style_extraction(a, nearby_pairs, b, far_pairs, c, seed=RNG_SEED + t)

        gaps_quasar.append(q_cost - opt_cost)
        gaps_smoothe.append(s_cost - opt_cost)
        if abs(q_cost - opt_cost) < 1e-6:
            exact_matches_quasar += 1
        if abs(s_cost - opt_cost) < 1e-6:
            exact_matches_smoothe += 1

    return dict(
        far_strength=far_strength,
        avg_gap_quasar=np.mean(gaps_quasar),
        avg_gap_smoothe=np.mean(gaps_smoothe),
        exact_rate_quasar=exact_matches_quasar / N_TRIALS,
        exact_rate_smoothe=exact_matches_smoothe / N_TRIALS,
    )


def main():
    print(f"K={K} hidden-inverse choice points, {N_TRIALS} random instances per setting\n")
    print(f"{'far_strength':>12} {'Quasar exact-rate':>18} {'Quasar avg gap':>15} "
          f"{'SmoothE exact-rate':>19} {'SmoothE avg gap':>16}")

    settings = [0.0, 0.25, 0.5, 1.0, 2.0]
    rows = []
    for fs in settings:
        r = run(fs)
        rows.append(r)
        print(f"{r['far_strength']:>12.2f} {r['exact_rate_quasar']*100:>16.1f}% "
              f"{r['avg_gap_quasar']:>15.4f} {r['exact_rate_smoothe']*100:>18.1f}% "
              f"{r['avg_gap_smoothe']:>16.4f}")

    print()
    high_far = [r for r in rows if r["far_strength"] >= 1.0]
    smoothe_solves_it = all(r["exact_rate_smoothe"] >= 0.9 for r in high_far)
    quasar_solves_it = all(r["exact_rate_quasar"] >= 0.9 for r in high_far)

    print("Kill-test verdict:")
    if quasar_solves_it:
        print("  Quasar-style (local/additive) extraction already solves this -> weak claim, kill.")
    else:
        print("  Quasar-style (local/additive) extraction FAILS as far-interaction grows (expected: "
              "it structurally cannot see far terms).")

    if smoothe_solves_it:
        print("  SmoothE-style (differentiable relaxation over exact cost) ALREADY SOLVES this "
              "objective even with strong far-interaction terms -> C4's core claim does not survive: "
              "'just use SmoothE' is a valid reviewer objection. KILL C4 in its current form.")
    else:
        print("  SmoothE-style relaxation DEGRADES as far-interaction strength grows (gets stuck in "
              "local optima from long-range coupling) -> the claimed gap has a testable technical "
              "basis. C4 survives this kill test, pending a structure-aware extractor prototype.")


if __name__ == "__main__":
    main()
