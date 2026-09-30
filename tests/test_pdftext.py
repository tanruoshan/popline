from popline.pdftext import normalize, pdf_passages, tokens


def test_normalize_glyphs_and_hyphens():
    assert normalize("Verhand-\nlungen über „Paneuropa“ und Groſſes") == 'Verhandlungen über "Paneuropa" und Grosses'


def test_normalize_keeps_ocr_errors():
    # we never correct OCR: 'de3' stays 'de3'
    assert normalize("de3 Europäers") == "de3 Europäers"


def test_tokens_lowercase_with_umlauts():
    assert tokens("Europäische Föderation!") == ["europäische", "föderation"]


def test_two_columns_stay_apart(two_column_pdf):
    ps = pdf_passages(two_column_pdf, "T9-01", min_words=5, max_words=250)
    texts = [p.text for p in ps]
    assert any("Paneuropabewegung" in t for t in texts)
    assert not any("Paneuropabewegung" in t and "Wetter" in t for t in texts)
    assert len({p.pid for p in ps}) == len(ps)
    assert all(p.pid.startswith("T9-01:0:") for p in ps)
