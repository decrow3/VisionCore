from __future__ import annotations

from typing import Any

import numpy as np

from .summaries import DEFAULT_SUMMARY_METRICS


def classify_stimulus_from_windows(
    rows: list[dict[str, Any]],
    *,
    features: tuple[str, ...] = DEFAULT_SUMMARY_METRICS,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if len(rows) < 20:
        return []
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return []

    x_rows: list[list[float]] = []
    y: list[str] = []
    groups: list[str] = []
    phases: list[str] = []
    for row in rows:
        vals = []
        ok = True
        for feat in features:
            try:
                val = float(row.get(feat, np.nan))
            except (TypeError, ValueError):
                val = np.nan
            if not np.isfinite(val):
                ok = False
                break
            vals.append(val)
        if ok:
            x_rows.append(vals)
            y.append(str(row["stimulus"]))
            groups.append(str(row["session"]))
            phases.append(str(row.get("phase", "")))
    if len(x_rows) < 20 or len(set(y)) < 2:
        return []

    x = np.asarray(x_rows, dtype=np.float64)
    y_arr = np.asarray(y)
    group_arr = np.asarray(groups)
    phase_arr = np.asarray(phases)
    out: list[dict[str, Any]] = []

    def evaluate(mask: np.ndarray, label: str) -> None:
        xx = x[mask]
        yy = y_arr[mask]
        gg = group_arr[mask]
        if xx.shape[0] < 20 or len(set(yy)) < 2:
            return
        splits = []
        if len(set(gg)) >= 2 and all(np.unique(yy[gg == g]).size >= 1 for g in np.unique(gg)):
            for train, test in LeaveOneGroupOut().split(xx, yy, gg):
                train_classes = set(yy[train])
                test_classes = set(yy[test])
                if len(train_classes) >= 2 and len(test_classes) >= 1 and test_classes.issubset(train_classes):
                    splits.append((train, test, "leave_one_session_out"))
        if not splits:
            counts = np.array([np.sum(yy == cls) for cls in np.unique(yy)])
            n_splits = int(min(5, np.min(counts)))
            if n_splits < 2:
                return
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
            splits = [(train, test, "stratified_kfold") for train, test in cv.split(xx, yy)]

        preds = np.empty_like(yy, dtype=object)
        used = np.zeros(yy.shape[0], dtype=bool)
        split_kind = splits[0][2]
        for train, test, _kind in splits:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=int(seed)),
            )
            model.fit(xx[train], yy[train])
            preds[test] = model.predict(xx[test])
            used[test] = True
        if not np.any(used):
            return
        out.append({
            "phase": label,
            "cv": split_kind,
            "n_windows": int(np.count_nonzero(used)),
            "n_sessions": int(len(set(gg))),
            "n_classes": int(len(set(yy))),
            "accuracy": float(accuracy_score(yy[used], preds[used])),
            "balanced_accuracy": float(balanced_accuracy_score(yy[used], preds[used])),
            "classes": ",".join(sorted(set(yy))),
            "features": ",".join(features),
        })

    evaluate(np.ones(y_arr.shape[0], dtype=bool), "all")
    for phase in sorted(set(phase_arr)):
        evaluate(phase_arr == phase, phase)
    return out
