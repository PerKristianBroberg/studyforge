from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from studyforge.loaders import load
from studyforge.paths import classify


@pytest.mark.eval
def test_ocr_reads_a_scanned_norwegian_page(tmp_path):
    """Real EasyOCR run on an image-only PDF (downloads OCR models on first use)."""
    page = Image.new("RGB", (1240, 400), "white")
    ImageDraw.Draw(page).text((60, 150), "Hypotesetesting og p-verdi", fill="black",
                              font=ImageFont.load_default(size=64))
    f = tmp_path / "UiB/STAT110/scan.pdf"
    f.parent.mkdir(parents=True)
    page.save(f, "PDF", resolution=150)
    [doc] = load(f, classify(tmp_path, f))
    assert "hypotesetesting" in doc.text.lower()
