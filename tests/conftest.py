import pymupdf
import pytest


@pytest.fixture
def two_column_pdf(tmp_path):
    """A small page with two columns: left about Paneuropa, right about the weather."""
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    left = ("Die Paneuropabewegung des Grafen Coudenhove-Kalergi fordert den Zusammen-\n"
            "schluss der europäischen Staaten. " * 3)
    right = "Das Wetter in Hermannstadt war gestern kalt und regnerisch. " * 4
    page.insert_textbox(pymupdf.Rect(40, 60, 280, 400), left, fontsize=10)
    page.insert_textbox(pymupdf.Rect(320, 60, 560, 400), right, fontsize=10)
    path = tmp_path / "T9-01_test.pdf"
    doc.save(path)
    return path
