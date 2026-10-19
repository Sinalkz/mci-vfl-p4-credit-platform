"""Cross-check the exact macro-F1 used by the circuit against scikit-learn."""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from pathlib import Path

from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s2_model import K, macro_f1_exact  # noqa: E402


def test_matches_sklearn_including_empty_classes():
    rng = random.Random(3)
    empties = 0
    for trial in range(2000):
        n = rng.randint(1, 60)
        # Restrict the alphabet sometimes so that some classes are empty.
        alphabet = rng.sample(range(K), rng.randint(1, K))
        y = [rng.choice(alphabet) for _ in range(n)]
        p = [rng.choice(alphabet) if rng.random() < 0.8 else rng.randrange(K) for _ in range(n)]
        res = macro_f1_exact(y, p)
        empties += bool(res["emptyClasses"])
        num, den = map(int, res["macroF1"].split("/"))
        ours = Fraction(num, den)
        theirs = f1_score(y, p, average="macro")  # labels=None: classes present in y or p
        assert abs(float(ours) - theirs) < 1e-12, (y, p)
        assert res["F"] == (ours * 10_000).numerator // (ours * 10_000).denominator
    assert empties > 100  # the empty-class rule was actually exercised


def test_confusion_identity():
    y = [0, 0, 1, 2, 2, 2, 4]
    p = [0, 1, 1, 2, 0, 2, 4]
    r = macro_f1_exact(y, p)
    for c in range(K):
        assert 2 * r["TP"][c] + r["FP"][c] + r["FN"][c] == y.count(c) + p.count(c)
    assert r["emptyClasses"] == [3]
