"""
PFN-Cert pilot smoke test.

Idea: certify robustness to context poisoning via partition aggregation
(Deep Partition Aggregation / AAAI'22 kNN-cert style): split the training
context into k disjoint partitions, fit one base learner per partition,
predict by majority vote. This gives a certified radius = floor((k - 1) / 2)
poisoned points tolerated (standard DPA bound), independent of which base
learner is used inside each partition.

We compare two base learners for the per-partition model:
  - kNN            (the AAAI'22 kNN-certification lineage baseline)
  - GradientBoosting (stand-in "strong ICL-like learner" for TabPFN/TabICL,
    since no TabPFN weights are available in this environment)

Kill criteria (from idea-inventory pilot spec):
  - KILL if partition-accuracy gain (strong learner vs kNN, averaged
    across datasets) < 2 percentage points, OR
  - KILL if strong learner beats kNN on fewer than 6 of 10 datasets.

This is a proxy smoke test only (no real TabPFN/TabICL/Mitra inference),
meant to sanity check whether the partition-aggregation certification
mechanism itself has headroom before investing in real PFN inference.
"""

import numpy as np
from sklearn.datasets import (
    load_iris, load_wine, load_breast_cancer, load_digits,
    make_classification,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

RNG_SEED = 0
N_PARTITIONS = 7  # certified radius = (7-1)//2 = 3 poisoned points tolerated


def make_synthetic(n_classes, n_features, n_informative, seed):
    X, y = make_classification(
        n_samples=600,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=seed,
    )
    return X, y


def load_datasets():
    datasets = {}

    iris = load_iris()
    datasets["iris"] = (iris.data, iris.target)

    wine = load_wine()
    datasets["wine"] = (wine.data, wine.target)

    bc = load_breast_cancer()
    datasets["breast_cancer"] = (bc.data, bc.target)

    digits = load_digits()
    datasets["digits"] = (digits.data, digits.target)

    # synthetic tabular datasets to round out to 10, varied difficulty
    synth_specs = [
        dict(n_classes=2, n_features=10, n_informative=5, seed=1),
        dict(n_classes=3, n_features=15, n_informative=6, seed=2),
        dict(n_classes=2, n_features=20, n_informative=4, seed=3),
        dict(n_classes=4, n_features=12, n_informative=8, seed=4),
        dict(n_classes=2, n_features=8, n_informative=3, seed=5),
        dict(n_classes=3, n_features=25, n_informative=10, seed=6),
    ]
    for i, spec in enumerate(synth_specs, start=1):
        spec = dict(spec)
        seed = RNG_SEED + spec.pop("seed")
        X, y = make_synthetic(seed=seed, **spec)
        datasets[f"synthetic_{i}"] = (X, y)

    return datasets


def partition_aggregate_predict(X_train, y_train, X_test, base_learner_fn, k, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X_train))
    partitions = np.array_split(idx, k)

    votes = np.zeros((len(X_test), k), dtype=int)
    for p_i, part_idx in enumerate(partitions):
        if len(part_idx) < 2 or len(np.unique(y_train[part_idx])) < 2:
            # degenerate partition: fall back to majority class of partition
            if len(part_idx) == 0:
                votes[:, p_i] = np.bincount(y_train).argmax()
                continue
            model_pred = np.full(len(X_test), np.bincount(y_train[part_idx]).argmax())
            votes[:, p_i] = model_pred
            continue
        model = base_learner_fn()
        model.fit(X_train[part_idx], y_train[part_idx])
        votes[:, p_i] = model.predict(X_test)

    # majority vote per test point
    preds = np.array([np.bincount(row).argmax() for row in votes])
    return preds


def run_pilot():
    datasets = load_datasets()

    def knn_fn():
        return KNeighborsClassifier(n_neighbors=5)

    def strong_fn():
        return GradientBoostingClassifier(random_state=RNG_SEED)

    results = []
    for name, (X, y) in datasets.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=RNG_SEED, stratify=y
        )
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        knn_preds = partition_aggregate_predict(
            X_train, y_train, X_test, knn_fn, N_PARTITIONS, RNG_SEED
        )
        strong_preds = partition_aggregate_predict(
            X_train, y_train, X_test, strong_fn, N_PARTITIONS, RNG_SEED
        )

        knn_acc = (knn_preds == y_test).mean() * 100
        strong_acc = (strong_preds == y_test).mean() * 100
        gain_pp = strong_acc - knn_acc

        results.append(dict(dataset=name, knn_acc=knn_acc, strong_acc=strong_acc, gain_pp=gain_pp))

    return results


def main():
    results = run_pilot()

    print(f"{'dataset':<16} {'kNN-cert acc':>13} {'strong-cert acc':>16} {'gain (pp)':>10}")
    for r in results:
        print(f"{r['dataset']:<16} {r['knn_acc']:>12.2f}% {r['strong_acc']:>15.2f}% {r['gain_pp']:>+9.2f}")

    n_datasets = len(results)
    n_beat = sum(1 for r in results if r["gain_pp"] > 0)
    avg_gain = np.mean([r["gain_pp"] for r in results])

    print()
    print(f"Datasets: {n_datasets}")
    print(f"Strong learner beats kNN on: {n_beat}/{n_datasets}")
    print(f"Average partition-accuracy gain: {avg_gain:+.2f} pp")

    kill_low_gain = avg_gain < 2.0
    kill_low_winrate = n_beat < 6
    kill = kill_low_gain or kill_low_winrate

    print()
    print("Kill criteria:")
    print(f"  avg gain < 2pp?      {'YES -> KILL' if kill_low_gain else 'no'} ({avg_gain:+.2f}pp)")
    print(f"  beats kNN on <6/10?  {'YES -> KILL' if kill_low_winrate else 'no'} ({n_beat}/{n_datasets})")
    print()
    print("VERDICT:", "NO-GO (kill)" if kill else "GO (survives pilot)")


if __name__ == "__main__":
    main()
