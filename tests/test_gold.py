import pymupdf

from popline.gold import article_groups, articles, label_passages, merge, parse_note, read_marks, unread_pages
from popline.pdftext import pdf_passages


def test_parse_note_v3_tags():
    n = parse_note("A3 IDEA CONTRA implicit borderline maybe geography")
    assert (n.article, n.grade, n.stance) == ("A3", 2, "CONTRA")
    assert n.flags == {"implicit", "borderline"} and n.comment == "maybe geography"
    n = parse_note("a9 idea pro cont:t1-04/a3 against:France speech")
    assert n.cont == "T1-04:A3" and n.against == ["France"] and n.flags == {"speech"}
    n = parse_note("A1 IDEA UNSURE against:France against:Paneuropa sceptical")
    assert n.against == ["France", "Paneuropa"] and n.comment == "sceptical"
    assert parse_note("A? IDEA PRO welcomes").article == "A?"
    assert parse_note("A2 MENTION reprint").grade == 1
    assert parse_note("A12 IDEA NEUTRAL").article == "A12"
    assert parse_note("") is None and parse_note("important article") is None
    assert parse_note("A1 IDEAL") is None


def _rect(page, box, note):
    a = page.add_rect_annot(pymupdf.Rect(*box))
    a.set_info(content=note)
    a.update()


def test_labels_nested_part_none_and_problems(two_column_pdf, tmp_path):
    doc = pymupdf.open(two_column_pdf)
    page = doc[0]
    _rect(page, (30, 50, 290, 410), "A1 IDEA CONTRA implicit")        # the whole left article
    _rect(page, (35, 55, 285, 150), "A1 IDEA CONTRA implicit")        # its IDEA part (inner)
    _rect(page, (0, 0, 10, 10), "important")                          # unreadable note
    marked = tmp_path / "marked.pdf"
    doc.save(marked)
    marks = read_marks(marked, "T9-01")
    assert any("without a readable note" in p for p in marks.problems)
    assert [r.part for r in marks.rects if r.note] == [False, True]
    ps = pdf_passages(two_column_pdf, "T9-01", min_words=5, max_words=250)
    labels = label_passages(ps, marks, min_share=0.5)
    left = [p for p in ps if "Paneuropabewegung" in p.text]
    right = [p for p in ps if "Wetter" in p.text]
    assert all(labels[p.pid]["article"] == "T9-01:A1" and labels[p.pid]["grade"] == 2 for p in left)
    assert all(labels[p.pid] == {"article": None, "grade": 0, "in_part": False} for p in right)
    art = articles(marks)["T9-01:A1"]
    assert art["stance"] == "CONTRA" and "implicit" in art["flags"]


def test_none_sticky_note_and_unread_pages(two_column_pdf, tmp_path):
    doc = pymupdf.open(two_column_pdf)
    doc.new_page()                                                    # page 1 stays unread
    doc[0].add_text_annot((20, 20), "NONE")
    marked = tmp_path / "m.pdf"
    doc.save(marked)
    marks = read_marks(marked, "T9-01")
    assert marks.none_pages == {"T9-01:0"}
    assert unread_pages(["T9-01:0", "T9-01:1"], marks) == ["T9-01:1"]


def test_cont_joins_articles_across_files(two_column_pdf, tmp_path):
    a, b = pymupdf.open(two_column_pdf), pymupdf.open(two_column_pdf)
    _rect(a[0], (30, 50, 290, 410), "A3 IDEA PRO")
    _rect(b[0], (30, 50, 290, 410), "A1 IDEA PRO cont:T9-01/A3")
    a.save(tmp_path / "a.pdf")
    b.save(tmp_path / "b.pdf")
    marks = merge(read_marks(tmp_path / "a.pdf", "T9-01"), read_marks(tmp_path / "b.pdf", "T9-02"))
    g = article_groups(marks)
    assert g["T9-02:A1"] == g["T9-01:A3"]
    assert len(articles(marks)) == 1


def test_kappa_page_offset(two_column_pdf, tmp_path):
    doc = pymupdf.open(two_column_pdf)
    _rect(doc[0], (30, 50, 290, 410), "A1 MENTION")
    one_page = tmp_path / "K-01.pdf"
    doc.save(one_page)
    marks = read_marks(one_page, "T1-05", page_offset=1)             # Simon's page = page 2 of Bea's file
    assert marks.rects[0].page_key == "T1-05:1" and marks.rects[0].note.grade == 1


def test_reused_number_and_freetext_none(two_column_pdf, tmp_path):
    doc = pymupdf.open(two_column_pdf)
    _rect(doc[0], (30, 50, 290, 410), "A1 IDEA PRO")
    _rect(doc[0], (300, 50, 570, 410), "A1 MENTION region only")        # same number, other article
    _rect(doc[0], (90, 600, 92, 700), "")                                # accidental line
    p2 = doc.new_page()
    a = p2.add_freetext_annot(pymupdf.Rect(20, 20, 120, 60), "NONE")
    a.update()
    doc.save(tmp_path / "m.pdf")
    marks = read_marks(tmp_path / "m.pdf", "T9-01")
    assert sorted(r.key for r in marks.rects if r.note) == ["A1", "A1#2"]
    assert len(articles(marks)) == 2
    assert marks.none_pages == {"T9-01:1"}
    assert any("used for 2 different" in p for p in marks.problems)
    assert any("tiny" in p for p in marks.problems)


def test_small_article_attaches_to_its_passage(two_column_pdf, tmp_path):
    doc = pymupdf.open(two_column_pdf)
    _rect(doc[0], (35, 55, 285, 80), "A1 MENTION Briand named")          # a two-line item
    doc.save(tmp_path / "m.pdf")
    ps = pdf_passages(two_column_pdf, "T9-01", min_words=5, max_words=250)
    labels = label_passages(ps, read_marks(tmp_path / "m.pdf", "T9-01"), min_share=0.5)
    assert [v["grade"] for v in labels.values() if v["article"]] == [1]
