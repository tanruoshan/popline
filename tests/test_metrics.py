import pytest

from popline.metrics import cohen_kappa, ndcg_at_k, reading_effort, recall_at_k


def test_ndcg_perfect_and_worst():
    assert ndcg_at_k([2, 1, 0, 0], 3) == 1.0
    assert ndcg_at_k([0, 0, 1, 2], 2) == 0.0
    assert ndcg_at_k([0, 0, 0], 3) is None


def test_recall_and_effort():
    rel = [False, True, False, True, True]
    assert recall_at_k(rel, 2) == pytest.approx(1 / 3)
    assert reading_effort(rel, 0.8) == 5          # ceil(0.8 * 3) = 3 relevant needed
    assert reading_effort(rel, 0.5) == 4
    assert reading_effort([False, False], 0.8) is None


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_article_level_recall_and_effort():
    from popline.metrics import article_effort, article_recall_at_k, articles_found
    order = ["p1", "p2", "p3", "p4", "p5"]
    art = {"p1": None, "p2": "A", "p3": "A", "p4": "B", "p5": "C"}
    curve = articles_found(order, art, {"A", "B"})
    assert curve == [0, 1, 1, 2, 2]
    assert article_recall_at_k(curve, 2, 2) == 0.5
    assert article_effort(curve, 2, 0.8) == 4
    assert article_effort(articles_found(["p1"], art, {"A"}), 1, 0.8) is None
