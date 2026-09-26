"""评测指标单测。"""
from visionforge.eval.metrics import average_precision, recall_at_k


def test_ap_perfect():
    assert average_precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0


def test_ap_empty_relevant():
    assert average_precision(["a", "b"], set()) == 0.0


def test_ap_partial():
    # relevant={a,b}; ranking [a,x,b] => AP = (1/1 + 2/3)/2 = 0.8333
    ap = average_precision(["a", "x", "b"], {"a", "b"})
    assert abs(ap - (1.0 + 2.0 / 3.0) / 2.0) < 1e-9


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, 2) == 1.0
    assert recall_at_k(["a", "x", "c"], {"a", "b"}, 2) == 0.5
