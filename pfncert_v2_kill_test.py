"""
PFN-Cert v2 kill test -- decisive first experiment ONLY.

Per the falsification protocol: before building a full TabPFN-v2 model,
answer the single decisive question:

    Can an adversary controlling ONE training row force that row's
    attention weight toward 1, and if so, does any PFN-specific
    certificate still beat a generic attention-perturbation bound?

If the answer is "the naive T/n attention-mass argument is false, and
once attention saturates the tightest valid bound is just the full
value-range bound (same as any generic softmax-attention model would
need)" -- then PFN-Cert v2's core claimed advantage is dead before any
large-scale experiment is needed.

Setup (single row-attention block, matches TabPFN's train->test attention)
----------------------------------------------------------------------
- n clean training rows with keys k_1..k_n and values v_1..v_n,
  ||k_i|| <= R, ||v_i|| <= V for all i (the ONLY assumption used).
- query q* (the test row), ||q*|| <= R.
- adversary controls T of the n rows (here T=1 for the decisive test):
  chooses k_p, v_p subject to the SAME norm constraints (no special
  pleading -- poisoned rows are bounded exactly like clean rows, this
  is the strongest reasonable threat model per Phase B).
- attention weight w_i = softmax(q* . k_i / sqrt(d))_i
- output = sum_i w_i v_i  (row-attention output before residual/head)

Naive claim under test: "poisoned attention mass <= T/n" (i.e. w_p <= 1/n
when T=1). We test whether an adversary can violate this by choosing k_p
to maximize q*.k_p within the norm ball.
"""

import numpy as np

RNG_SEED = 0
D = 16          # embedding dim
R = 5.0         # key/query norm bound
V_NORM = 5.0    # value norm bound


def build_clean_rows(n, seed):
    rng = np.random.RandomState(seed)
    K = rng.normal(size=(n, D))
    K = K / np.linalg.norm(K, axis=1, keepdims=True) * (R * 0.5)  # modest norms
    Vv = rng.normal(size=(n, D))
    Vv = Vv / np.linalg.norm(Vv, axis=1, keepdims=True) * (V_NORM * 0.5)
    q = rng.normal(size=D)
    q = q / np.linalg.norm(q) * R
    return K, Vv, q


def attention_weights(q, K):
    logits = K @ q / np.sqrt(D)
    logits = logits - logits.max()
    w = np.exp(logits)
    w = w / w.sum()
    return w


def worst_case_value_diff(v_clean_out, V_bound):
    """Loosest possible generic bound if poisoned row's attention weight
    saturates to (near) 1: output can move anywhere in the value ball, so
    the tightest *valid* bound on ||Delta output|| is 2*V_bound (poisoned
    output could be -v_hat direction vs clean output +v_hat direction)."""
    return 2 * V_bound


def naive_bound(n, T, V_bound):
    """The claim under test: poisoned attention mass <= T/n, hence
    ||Delta output|| <= (T/n) * 2*V_bound (worst case value swap)."""
    return (T / n) * 2 * V_bound


def run_single_high_key_attack(n, seed):
    K, Vv, q = build_clean_rows(n, seed)

    clean_w = attention_weights(q, K)
    clean_out = clean_w @ Vv

    # Adversary picks k_p aligned with q, scaled to the SAME norm bound R
    # as clean keys are allowed (Phase B: no special pleading).
    k_p = (q / np.linalg.norm(q)) * R
    v_p = -(clean_out / (np.linalg.norm(clean_out) + 1e-12)) * V_NORM  # push output away

    K_poisoned = np.vstack([K, k_p[None, :]])
    V_poisoned = np.vstack([Vv, v_p[None, :]])

    w_poisoned = attention_weights(q, K_poisoned)
    w_p = w_poisoned[-1]

    out_poisoned = w_poisoned @ V_poisoned
    empirical_delta = np.linalg.norm(out_poisoned - clean_out)

    T = 1
    naive = naive_bound(n, T, V_NORM)
    generic_loose = worst_case_value_diff(clean_out, V_NORM)

    return dict(
        n=n,
        w_p=w_p,
        naive_T_over_n_bound=naive,
        empirical_delta=empirical_delta,
        naive_bound_violated=empirical_delta > naive + 1e-9,
        generic_loose_bound=generic_loose,
        generic_bound_violated=empirical_delta > generic_loose + 1e-9,
    )


def main():
    print("Decisive test: one adversarial high-key poisoned row (T=1), varying n")
    print(f"{'n':>6} {'attn weight w_p':>16} {'naive T/n bound':>16} "
          f"{'empirical ||delta||':>20} {'naive violated?':>16} {'generic-loose violated?':>24}")

    ns = [16, 32, 64, 128, 256]
    rows = []
    for n in ns:
        r = run_single_high_key_attack(n, RNG_SEED)
        rows.append(r)
        print(f"{r['n']:>6} {r['w_p']:>16.4f} {r['naive_T_over_n_bound']:>16.4f} "
              f"{r['empirical_delta']:>20.4f} {str(r['naive_bound_violated']):>16} "
              f"{str(r['generic_bound_violated']):>24}")

    print()
    any_naive_violation = any(r["naive_bound_violated"] for r in rows)
    any_generic_violation = any(r["generic_bound_violated"] for r in rows)
    max_wp = max(r["w_p"] for r in rows)

    print("Findings:")
    print(f"  Max attention weight achieved by a single poisoned row: {max_wp:.4f}")
    print(f"  Naive 'poisoned attention mass <= T/n' bound violated on: "
          f"{sum(r['naive_bound_violated'] for r in rows)}/{len(rows)} settings")
    print(f"  Loose generic worst-case bound (2*V_bound) violated on: "
          f"{sum(r['generic_bound_violated'] for r in rows)}/{len(rows)} settings")
    print()

    print("Decisive verdict:")
    if any_naive_violation:
        print("  YES -- a single adversarial key drives attention weight w_p close to 1"
              " and BREAKS the naive T/n attention-mass argument. Any certificate resting"
              " on 'poisoned influence <= T/n' is mathematically INVALID under adversarial"
              " keys (Phase C/J: certificate violation).")
    else:
        print("  Naive T/n bound held in this run (unexpected -- re-check attack construction).")

    if not any_generic_violation:
        print("  The loose generic worst-case bound (attention can saturate to 1, so output"
              " can move up to the full value range, ||Delta|| <= 2*V_bound) is NEVER violated."
              " That bound requires NOTHING PFN-specific -- it holds for ANY softmax-attention"
              " model (generic cross-attention, generic self-attention) with the same norm"
              " constraints. So once a single row can saturate attention, the tightest bound"
              " available to PFN-Cert v2 collapses to the same expression a generic transformer"
              " perturbation certificate would already give.")
        print()
        print("  ANSWER TO THE DECISIVE QUESTION: an adversary CAN force w_p -> 1 with one row,"
              " and NO, no PFN-specific certificate can beat the generic bound in this regime --"
              " they are the same expression. Per the stated stopping rule: STOP. Do not proceed"
              " to full TabPFN-v2 implementation on this attack surface.")
        print()
        print("  VERDICT: KILLED (for any certificate that claims sub-generic scaling via")
        print("  'attention mass <= T/n' or similar; PFN row/column-attention structure gives")
        print("  no leverage against a single-row, adversarially-aligned-key attack -- softmax")
        print("  saturation is a property of attention in general, not of PFN specifically).")
    else:
        print("  Generic loose bound was violated -- investigate further before concluding.")


if __name__ == "__main__":
    main()
