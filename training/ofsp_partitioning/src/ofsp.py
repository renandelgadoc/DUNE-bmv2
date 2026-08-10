"""
Optimal Feature Set Partitioning (OFSP) + Multi-View Ensemble Learning (MEL)
==============================================================================

Implements Algorithm 7 ("Optimal Feature Set Partition Method") and the
performance-weighted ensembling described in Section 3.10, both from:

Kumar, A., Yadav, J. (2023). A review of feature set partitioning methods
for multi-view ensemble learning. Information Fusion, 100, 101959.

OFSP was originally proposed in:
Kumar, V., Minz, S. (2016). Multi-view ensemble learning: an optimal feature
set partitioning for high-dimensional data classification. Knowledge and
Information Systems, 49, 1-59.

Notes on implementation choices
--------------------------------
The paper's pseudocode does not specify HOW the k seed features are chosen
or what determines feature processing order -- it just says "select some
features". This implementation shuffles the feature order once (seeded by
random_state) for reproducibility; you can swap in e.g. an information-gain
ranking if you want a more deterministic seeding strategy.

The base classifier defaults to RandomForestClassifier (rather than the
paper's RBF-SVM) so that the views OFSP builds are optimized for the same
classifier family used downstream in the project -- the wrapper-style
"does this feature help?" decisions in Algorithm 7 are classifier-specific,
so the evaluation classifier and the deployed classifier should match.

View evaluation uses the out-of-bag (OOB) score only -- no k-fold
cross-validation fallback. This assumes a bagging-style classifier
(RandomForest, ExtraTrees, ...); plugging in a non-bagging classifier
(e.g. SVM) will fail since it has no oob_score_ attribute.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Fixed number of views (k). Set this once here, or pass k= explicitly to
# ofsp_partition() if you want to override it per call.
N_VIEWS = 5


# ---------------------------------------------------------------------------
# Algorithm 7: Optimal Feature Set Partitioning
# ---------------------------------------------------------------------------

def _view_accuracy(X, y, feature_idx, clf, sample_weight=None, objective='accuracy'):
    """
    Out-of-bag score of `clf` trained on a feature subset.

    Assumes a bagging-style estimator (RandomForest, ExtraTrees, ...).
    Each tree trains on its bootstrap sample (~63% of rows); the ~37% left
    out are scored for free in the same fit -- one fit, no k-fold retraining.

    `objective` selects what the greedy partitioner maximizes:
      'accuracy' -> plain (unweighted) OOB accuracy = model.oob_score_ (default,
                    original behaviour; dominated by the majority classes).
      'flow_acc' -> OOB accuracy weighted by `sample_weight` (the flow-size
                    weight), i.e. each flow counts equally -- consistent with
                    how cluster_analysis trains and how calculate_score scores.
      'macro_f1' -> unweighted macro-F1 over the OOB predictions, giving every
                    class an equal vote so minority-class features get credit.
    """
    if not feature_idx:
        return 0.0

    model = clone(clf)
    model.set_params(oob_score=True, bootstrap=True)
    model.fit(X[:, feature_idx], y, sample_weight=sample_weight)
    if objective == 'accuracy':
        return float(model.oob_score_)

    # 'flow_acc' / 'macro_f1' need per-sample OOB predictions. Rows never left
    # out of any bootstrap have NaN in oob_decision_function_ -- drop them.
    oob_df = model.oob_decision_function_
    valid = ~np.isnan(oob_df).any(axis=1)
    if not valid.any():
        return 0.0
    y_pred = model.classes_[np.argmax(oob_df[valid], axis=1)]
    y_true = y[valid]
    if objective == 'flow_acc':
        w = None if sample_weight is None else np.asarray(sample_weight)[valid]
        return float(accuracy_score(y_true, y_pred, sample_weight=w))
    if objective == 'macro_f1':
        # Flow-weighted macro-F1: each class counts equally (macro), and within
        # each class every sample is weighted by the flow-size weight -- keeps the
        # objective consistent with flow_acc / cluster_analysis / calculate_score.
        w = None if sample_weight is None else np.asarray(sample_weight)[valid]
        return float(f1_score(y_true, y_pred, average='macro', sample_weight=w))
    raise ValueError(f"unknown objective: {objective!r}")


def ofsp_partition(X, y, k=N_VIEWS, clf=None, random_state=0, sample_weight=None,
                   objective='accuracy'):
    """
    Greedily build `k` views from X's feature set, following Algorithm 7.

    Each of the k views is seeded with one feature. Remaining features are
    then tested one at a time: a feature is permanently added to whichever
    view it improves the most; if it improves no view, it is discarded
    (this is the algorithm's built-in dimensionality reduction step).

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    k : int
        Number of views to construct. Defaults to the fixed N_VIEWS set at
        the top of this file -- pass k= explicitly to override per call.
    clf : sklearn estimator, optional
        Bagging-style base classifier used to score each view via its
        out-of-bag score. Defaults to a RandomForestClassifier.
    random_state : int
        Seeds the shuffle of feature processing order.

    Returns
    -------
    views : list[list[int]]            -- column indices per view
    view_accuracies : list[float]      -- final OOB accuracy per view
    discarded : list[int]              -- features that helped no view
    """
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state) if clf is None else clf
    n_features = X.shape[1]
    feature_order = np.random.RandomState(random_state).permutation(n_features)

    seed, remaining = list(feature_order[:k]), list(feature_order[k:])
    views = [[int(f)] for f in seed]
    print(f"  [OFSP] Seeding {k} views...")
    accuracies = [_view_accuracy(X, y, v, clf, sample_weight, objective) for v in views]
    for i, (v, a) in enumerate(zip(views, accuracies)):
        print(f"    View {i}: seed feature {v[0]}, OOB={a:.4f}")
    discarded = []

    total_remaining = len(remaining)
    for idx, f_new in enumerate(remaining):
        f_new = int(f_new)
        print(f"  [OFSP] Feature {idx+1}/{total_remaining} (col {f_new}): testing {k} views...", end='', flush=True)
        trial_acc = [_view_accuracy(X, y, v + [f_new], clf, sample_weight, objective) for v in views]
        gains = [trial_acc[i] - accuracies[i] for i in range(k)]
        best = int(np.argmax(gains))

        if gains[best] > 0:
            views[best].append(f_new)
            accuracies[best] = trial_acc[best]
            print(f" -> added to view {best} (gain={gains[best]:+.4f}, OOB={accuracies[best]:.4f})")
        else:
            discarded.append(f_new)
            print(f" -> discarded (no gain)")

    return views, accuracies, discarded


def ofsp_select_k(X, y, k_range=range(2, 11), clf=None, random_state=0):
    """
    Optional utility: run OFSP for every k in k_range and keep the partition
    with the best (performance-weighted) score -- matching the paper's
    protocol of building 2-views through 10-views and recording the best
    result. Not used by default now that k is fixed via N_VIEWS / ofsp_partition.
    """
    best = None
    for k in k_range:
        views, accs, discarded = ofsp_partition(X, y, k, clf, random_state)
        weights = np.array(accs) / sum(accs) if sum(accs) > 0 else np.ones(k) / k
        score = float(np.dot(weights, accs))  # proxy to rank candidate k's
        record = dict(k=k, views=views, view_accuracies=accs,
                      discarded=discarded, weights=weights, score=score)
        if best is None or score > best["score"]:
            best = record
    return best


# ---------------------------------------------------------------------------
# Section 3.10: performance-weighted ensembling of view predictions
# ---------------------------------------------------------------------------

def weighted_vote(predictions, weights):
    """
    Combine per-view predictions for a batch of test samples using
    performance-weighted majority voting:

        y(x_t) = argmax_{y'}  sum_i  W_i * [y_i(x_t) == y']

    Parameters
    ----------
    predictions : ndarray, shape (n_views, n_samples)
    weights : ndarray, shape (n_views,)  -- e.g. each view's training accuracy

    Returns
    -------
    ndarray, shape (n_samples,) -- ensembled prediction per sample
    """
    predictions = np.asarray(predictions)
    n_views, n_samples = predictions.shape
    final = np.empty(n_samples, dtype=predictions.dtype)

    for j in range(n_samples):
        votes: dict = {}
        for i in range(n_views):
            label = predictions[i, j]
            votes[label] = votes.get(label, 0.0) + weights[i]
        final[j] = max(votes, key=votes.get)

    return final


# ---------------------------------------------------------------------------
# End-to-end demo: OFSP views -> per-view classifiers -> weighted ensemble
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = make_classification(
        n_samples=300, n_features=40, n_informative=12,
        n_redundant=8, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=200, random_state=42)

    # 1. Build OFSP views for the fixed number of views set by N_VIEWS
    views, view_accuracies, discarded = ofsp_partition(
        X_train, y_train, k=N_VIEWS, clf=clf
    )
    print(f"k = {N_VIEWS} views (fixed)")
    for i, (view, acc) in enumerate(zip(views, view_accuracies)):
        print(f"  View {i + 1}: {len(view)} features, OOB accuracy = {acc:.3f}")
    print(f"  Discarded {len(discarded)} features")

    # 2. Train one classifier per view on the full training set
    view_models = [clone(clf).fit(X_train[:, v], y_train) for v in views]

    # 3. Predict on the test set from each view, then ensemble
    test_preds = np.array([m.predict(X_test[:, v])
                            for m, v in zip(view_models, views)])
    final_pred = weighted_vote(test_preds, view_accuracies)

    print(f"\nEnsembled (OFSP/MEL) test accuracy: {accuracy_score(y_test, final_pred):.3f}")
    single_view_acc = accuracy_score(
        y_test, clone(clf).fit(X_train, y_train).predict(X_test)
    )
    print(f"Single-view (all features) test accuracy: {single_view_acc:.3f}")