"""Five-fold target-disjoint boosted-tree membership scoring."""

import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits


def score(X, y, groups, seed=42, folds=5):
    if len(np.unique(groups)) < folds:
        raise ValueError(f"Need at least {folds} distinct target groups")
    if not np.isfinite(X).all() or set(np.unique(y)) != {0, 1}:
        raise ValueError("Attack needs finite features and both membership labels")
    prediction = np.empty(len(y), float)
    fold_auc = []
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for train, test in splitter.split(X, y, groups):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Target leakage between folds")
            model = HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=31,
                learning_rate=0.1,
                min_samples_leaf=10,
                random_state=seed,
            )
            model.fit(X[train], y[train])
            prediction[test] = model.predict_proba(X[test])[:, 1]
            fold_auc.append(float(roc_auc_score(y[test], prediction[test])))
    auc = float(roc_auc_score(y, prediction))
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    estimates = []
    positions = {g: np.flatnonzero(groups == g) for g in unique}
    for _ in range(1000):
        rows = np.concatenate(
            [positions[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        if len(np.unique(y[rows])) == 2:
            estimates.append(roc_auc_score(y[rows], prediction[rows]))
    return dict(
        auc=auc,
        fold_auc=fold_auc,
        target_bootstrap_ci95=list(map(float, np.quantile(estimates, [0.025, 0.975]))),
        targets=len(unique),
        samples=len(y),
    ), prediction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    raw = np.load(args.features, allow_pickle=False)
    result = dict(protocol=json.loads(str(raw["protocol"])), methods={})
    for method in np.unique(raw["methods"]):
        take = raw["methods"] == method
        metrics, _ = score(
            raw["X"][take], raw["y"][take], raw["groups"][take], args.seed
        )
        result["methods"][str(method)] = metrics
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
