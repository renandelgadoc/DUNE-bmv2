"""
Optimal Feature Set Partitioning (OFSP) + Multi-View Ensemble Learning (MEL)
==============================================================================

Implements Algorithm 7 ("Optimal Feature Set Partition Method") from the review
below; the views are combined by majority vote, one vote per view, as in the
switches (the review's Section 3.10 describes a performance-weighted vote instead):

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

import logging
import time

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# Fixed number of views (k). Set this once here, or pass k= explicitly to
# ofsp_partition() if you want to override it per call.
N_VIEWS = 5

# Log of the partitioning; run_ofsp_partitioning.py sets its format (with date/time).
logger = logging.getLogger('ofsp')


# ---------------------------------------------------------------------------
# Algorithm 7: Optimal Feature Set Partitioning
# ---------------------------------------------------------------------------

def _oob_evaluate(X, y, feature_idx, clf, sample_weight=None, objective='macro_f1', predicted_classes=None):
    """
    Train `clf` with only the features of one view and return how good it is, plus
    the class it predicts for every training row out-of-bag (OOB).

    OOB: each tree is trained on a random draw of the rows (bootstrap) and never sees
    about 37% of them. The OOB prediction of a row is the vote of only the trees that
    did not see it, so it works as a validation, with no separate validation set and
    no extra training. Assumes a bagging-style estimator (RandomForest, ExtraTrees).

    y holds the class of each row as a number, 0 to n_classes-1 (see ofsp_partition).

    `objective` is what the partitioner maximizes:
      'accuracy' -> fraction of rows predicted right (model.oob_score_); dominated by
                    the majority classes.
      'flow_acc' -> the same, each row weighted by `sample_weight` (its flow weight),
                    so each flow counts equally, as cluster_analysis trains and
                    calculate_score scores.
      'macro_f1' -> F1 of each class (rows weighted by `sample_weight`), averaged over
                    the classes, so a small class counts as much as a big one.

    If `predicted_classes` is given, nothing is trained: that prediction (e.g. the
    majority vote of all views, see majority_vote) is scored the same way. This is how
    the ensemble is scored with the same metric as a single view.

    Returns (score, predicted_classes):
      score   -> the `objective` value
      predicted_classes[i] -> predicted class of row i, or -1 if every tree saw row i (no OOB vote)
    """
    if predicted_classes is None:
        if not feature_idx:
            return 0.0, np.full(len(y), -1, dtype=np.int64)
        model = clone(clf)
        model.set_params(oob_score=True, bootstrap=True)
        model.fit(X[:, feature_idx], y, sample_weight=sample_weight)
        # oob_decision_function_: for each row, the share of OOB tree votes per class;
        # all NaN for a row that every tree saw.
        oob_df = model.oob_decision_function_
        has_oob = ~np.isnan(oob_df).any(axis=1)
        predicted_classes = np.full(len(y), -1, dtype=np.int64)
        # The predicted class of a row is the class with the largest share of votes.
        predicted_classes[has_oob] = model.classes_[np.argmax(oob_df[has_oob], axis=1)]
        if objective == 'accuracy':
            return float(model.oob_score_), predicted_classes

    # Score only the rows that have a prediction (predicted_classes = -1 means no vote).
    valid = predicted_classes >= 0
    if not valid.any():
        return 0.0, predicted_classes
    y_true, y_pred = y[valid], predicted_classes[valid]
    w = None if sample_weight is None else np.asarray(sample_weight)[valid]
    if objective == 'accuracy':
        return float(accuracy_score(y_true, y_pred)), predicted_classes
    if objective == 'flow_acc':
        return float(accuracy_score(y_true, y_pred, sample_weight=w)), predicted_classes
    if objective == 'macro_f1':
        return float(f1_score(y_true, y_pred, average='macro', sample_weight=w)), predicted_classes
    raise ValueError(f"unknown objective: {objective!r}")


def ofsp_partition(X, y, k=N_VIEWS, clf=None, random_state=0, sample_weight=None,
                   objective='macro_f1', scope='ensemble', tie_label='normal', feature_names=None):
    """
    Greedily build `k` views from X's feature set, following Algorithm 7.

    Each of the k views is seeded with one feature. Remaining features are then tested
    one at a time: the feature is tried in each view (that view retrained with it) and
    is permanently added to the view where it gives the largest gain; if it gives no
    gain in any view, it is discarded (the algorithm's built-in dimensionality
    reduction step).

    `scope` sets what "gain" means:
      'ensemble' -> gain of the ENSEMBLE: majority vote of all k views (majority_vote),
                    one vote each, ties to `tie_label`, as the vote in the switches
                    (dune_inference.p4.j2). The ensemble is scored with the same
                    `objective` as a view.
      'view'     -> gain of the view alone (the former rule).

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
    objective : 'accuracy', 'flow_acc' or 'macro_f1' (see _oob_evaluate)
    scope : 'ensemble' or 'view' (see above)
    tie_label : class that wins ties of the ensemble vote
    feature_names : list[str], optional -- name of each column, used only in the log

    Returns
    -------
    views : list[list[int]]            -- column indices per view
    view_accuracies : list[float]      -- final OOB score of each view alone
    discarded : list[int]              -- features that gave no gain
    """
    if scope not in ('ensemble', 'view'):
        raise ValueError(f"unknown scope: {scope!r}")
    clf = RandomForestClassifier(n_estimators=200, random_state=random_state) if clf is None else clf
    n_features = X.shape[1]
    feature_order = np.random.RandomState(random_state).permutation(n_features)

    # The classes as numbers: `classes` is the sorted list of class names and y_idx[i]
    # is the position of row i's class in it (what _oob_evaluate and majority_vote use).
    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)
    tie_class = int(np.searchsorted(classes, tie_label)) if tie_label in classes else None

    def name(col):
        return feature_names[col] if feature_names is not None else f"col {col}"

    seed, remaining = list(feature_order[:k]), list(feature_order[k:])
    views = [[int(f)] for f in seed]
    logger.info(f"OFSP: {k} views, scope={scope}, objective={objective}, "
                f"{X.shape[0]:,} rows, {n_features} features, classes={list(classes)}")
    # One (score, OOB prediction) pair per view.
    accuracies, view_predictions = [], []
    for i, v in enumerate(views):
        t0 = time.time()
        score, predicted_classes = _oob_evaluate(X, y_idx, v, clf, sample_weight, objective)
        accuracies.append(score)
        view_predictions.append(predicted_classes)
        logger.info(f"seed view {i}: [{name(v[0])}] -> view OOB={score:.4f} ({time.time() - t0:.0f} s)")
    if scope == 'ensemble':
        ensemble, _ = _oob_evaluate(None, y_idx, None, None, sample_weight, objective,
                                     predicted_classes=majority_vote(view_predictions, n_classes, tie_class))
        logger.info(f"seed ensemble OOB={ensemble:.4f}")
    discarded = []

    total_remaining = len(remaining)
    for idx, f_new in enumerate(remaining):
        f_new = int(f_new)
        logger.info(f"feature {idx+1}/{total_remaining} '{name(f_new)}': testing it in each of the {k} views")
        trials, gains = [], []
        for i, v in enumerate(views):
            t0 = time.time()
            # The feature tried in view i: that view retrained with it.
            trial_score, trial_predictions = _oob_evaluate(X, y_idx, v + [f_new], clf, sample_weight, objective)
            trials.append((trial_score, trial_predictions))
            if scope == 'ensemble':
                # Ensemble score with view i replaced by its trial; the other views unchanged.
                vote = majority_vote(view_predictions[:i] + [trial_predictions] + view_predictions[i + 1:],
                                     n_classes, tie_class)
                trial_ensemble, _ = _oob_evaluate(None, y_idx, None, None, sample_weight, objective, predicted_classes=vote)
                gains.append(trial_ensemble - ensemble)
                logger.info(f"  view {i} + '{name(f_new)}': view OOB {accuracies[i]:.4f} -> {trial_score:.4f}, "
                            f"ensemble OOB {ensemble:.4f} -> {trial_ensemble:.4f} (gain {gains[-1]:+.4f}) "
                            f"({time.time() - t0:.0f} s)")
            else:
                gains.append(trial_score - accuracies[i])
                logger.info(f"  view {i} + '{name(f_new)}': view OOB {accuracies[i]:.4f} -> {trial_score:.4f} "
                            f"(gain {gains[-1]:+.4f}) ({time.time() - t0:.0f} s)")
        best = int(np.argmax(gains))

        if gains[best] > 0:
            views[best].append(f_new)
            accuracies[best], view_predictions[best] = trials[best]
            if scope == 'ensemble':
                ensemble += gains[best]
            logger.info(f"  -> '{name(f_new)}' added to view {best} (gain {gains[best]:+.4f}); "
                        f"view {best} now: {[name(c) for c in views[best]]}")
        else:
            discarded.append(f_new)
            logger.info(f"  -> '{name(f_new)}' discarded (no gain in any view)")

    for i, v in enumerate(views):
        logger.info(f"final view {i}: view OOB={accuracies[i]:.4f}, features={[name(c) for c in v]}")
    if scope == 'ensemble':
        logger.info(f"final ensemble OOB={ensemble:.4f}")
    if discarded:
        logger.info(f"discarded: {[name(c) for c in discarded]}")
    return views, accuracies, discarded


# ---------------------------------------------------------------------------
# Majority vote of the view predictions (one vote per view, as in the switches)
# ---------------------------------------------------------------------------

def majority_vote(predictions, n_classes=None, tie_class=None):
    """
    Combine per-view predictions for a batch of samples by majority vote: each view
    gives one vote to the class it predicted, and the class with the most votes wins,
    as the switches do (dune_inference.p4.j2). Done for all samples at once (no loop
    over samples), since OFSP votes on every training row many times.

    Parameters
    ----------
    predictions : ndarray, shape (n_views, n_samples)
        Predicted class of each sample by each view, as a number 0..n_classes-1;
        -1 means the view does not vote on that sample (e.g. no OOB prediction).
    n_classes : int, optional -- number of classes (default: largest class + 1)
    tie_class : int, optional
        Class that wins exact ties (e.g. 'normal', as in the P4 vote). Without it,
        a tie goes to the smallest class number.

    Returns
    -------
    ndarray, shape (n_samples,) -- ensembled prediction per sample (-1 if no view voted)
    """
    predictions = np.asarray(predictions)
    n_views, n_samples = predictions.shape
    n_classes = int(predictions.max()) + 1 if n_classes is None else n_classes

    # votes[j, c]: number of views that voted for class c on sample j.
    votes = np.zeros((n_samples, n_classes))
    for i in range(n_views):
        voting = np.nonzero(predictions[i] >= 0)[0]
        votes[voting, predictions[i][voting]] += 1

    has_vote = votes.sum(axis=1) > 0
    if tie_class is not None:
        # A tiny extra vote for the tie class: it only changes the result of exact ties.
        votes[has_vote, tie_class] += 1e-9
    final = np.argmax(votes, axis=1)
    final[~has_vote] = -1
    return final


# ---------------------------------------------------------------------------
# End-to-end demo: OFSP views -> per-view classifiers -> majority-vote ensemble
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
    final_pred = majority_vote(test_preds)

    print(f"\nEnsembled (OFSP/MEL) test accuracy: {accuracy_score(y_test, final_pred):.3f}")
    single_view_acc = accuracy_score(
        y_test, clone(clf).fit(X_train, y_train).predict(X_test)
    )
    print(f"Single-view (all features) test accuracy: {single_view_acc:.3f}")