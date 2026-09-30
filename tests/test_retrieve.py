from popline.pdftext import Block, Passage
from popline.retrieve import bm25_expanded, expand_exact, expand_fuzzy, keyword_b0, rrf


def P(pid, text):
    return Passage(pid, "X:0", [Block("X:0", 0, (0, 0, 1, 1), text)])


PS = [P("a", "Das Wetter war schön."), P("b", "Die Panzuropa Bewegung tagt."),
      P("c", "Die Paneuropa Bewegung und Paneuropa."), P("d", "Europäische Verständigung ist nötig.")]


def test_exact_expansion_takes_german_endings_not_ocr_errors():
    m = expand_exact(["europäisch", "paneuropa"], {"europäischen", "panzuropa", "paneuropas"})
    assert set(m) == {"europäischen", "paneuropas"}


def test_fuzzy_expansion_takes_ocr_variants():
    m = expand_fuzzy(["paneuropa", "kalergi"], {"panzuropa", "kafergi", "wetter", "pan"}, min_sim=0.8)
    assert m == {"panzuropa": "paneuropa", "kafergi": "kalergi"}


def test_b0_ranks_by_hits_and_ties_keep_reading_order():
    r = keyword_b0(PS, "paneurop")
    assert r.order == ["c", "a", "b", "d"]


def test_s1_finds_what_b1_misses():
    b1 = bm25_expanded(PS, ["paneuropa"], "B1")
    s1 = bm25_expanded(PS, ["paneuropa"], "S1", fuzzy_sim=0.8)
    assert b1.scores["b"] == 0 and s1.scores["b"] > 0
    assert "panzuropa~paneuropa" in s1.why["b"]


def test_rrf_combines():
    b0 = keyword_b0(PS, "paneurop")
    s1 = bm25_expanded(PS, ["paneuropa", "europäisch"], "S1", fuzzy_sim=0.8)
    f = rrf([b0, s1], 60, "S3", [p.pid for p in PS])
    assert f.order[0] == "c" and f.order[-1] == "a"
