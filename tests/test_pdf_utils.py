"""Tests for PDF processing utilities."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest
from PIL import Image

from docscan.pdf_utils import (
    pdf_to_images,
    get_pdf_page_count,
    is_valid_pdf,
)


@pytest.fixture
def mock_pdf_document():
    """Create a mock PyMuPDF document."""
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2  # 2 pages

    # Create mock pages
    mock_page1 = MagicMock()
    mock_page2 = MagicMock()

    # Create mock pixmaps
    mock_pix1 = MagicMock()
    mock_pix2 = MagicMock()

    # Create a simple 1x1 PNG image bytes
    img = Image.new('RGB', (1, 1))
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_data = img_bytes.getvalue()

    mock_pix1.tobytes.return_value = img_data
    mock_pix2.tobytes.return_value = img_data

    mock_page1.get_pixmap.return_value = mock_pix1
    mock_page2.get_pixmap.return_value = mock_pix2

    mock_doc.__getitem__.side_effect = [mock_page1, mock_page2]

    return mock_doc


def test_pdf_to_images_success(mock_pdf_document, tmp_path):
    """Test successful PDF to images conversion."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_pdf_document
    mock_fitz.Matrix.return_value = MagicMock()

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        images = pdf_to_images(pdf_path)

        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)
        mock_pdf_document.close.assert_called_once()


def test_pdf_to_images_custom_dpi(mock_pdf_document, tmp_path):
    """Test PDF conversion with custom DPI."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_pdf_document
    mock_fitz.Matrix.return_value = MagicMock()

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        images = pdf_to_images(pdf_path, dpi=300)

        assert len(images) == 2
        # Check that Matrix was called with zoom factor (300/72 = 4.166...)
        mock_fitz.Matrix.assert_called()


def test_pdf_to_images_import_error(tmp_path):
    """Test PDF conversion when PyMuPDF not installed."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    with patch('builtins.__import__', side_effect=ImportError("No module named 'fitz'")):
        with pytest.raises(ImportError, match="PyMuPDF"):
            pdf_to_images(pdf_path)


def test_pdf_to_images_invalid_pdf(tmp_path):
    """Test PDF conversion with invalid PDF file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_fitz = MagicMock()
    mock_fitz.open.side_effect = Exception("Invalid PDF")

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        with pytest.raises(ValueError, match="Failed to convert"):
            pdf_to_images(pdf_path)


def test_get_pdf_page_count_success(tmp_path):
    """Test getting PDF page count successfully."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 5

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        count = get_pdf_page_count(pdf_path)

        assert count == 5
        mock_doc.close.assert_called_once()


def test_get_pdf_page_count_error(tmp_path):
    """Test getting page count with error."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_fitz = MagicMock()
    mock_fitz.open.side_effect = Exception("Cannot open")

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        count = get_pdf_page_count(pdf_path)

        assert count == 0


def test_is_valid_pdf_success(tmp_path):
    """Test valid PDF check with valid file."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.touch()

    mock_doc = MagicMock()

    mock_fitz = MagicMock()
    mock_fitz.open.return_value = mock_doc

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        result = is_valid_pdf(pdf_path)

        assert result is True
        mock_doc.close.assert_called_once()


def test_is_valid_pdf_nonexistent_file(tmp_path):
    """Test valid PDF check with nonexistent file."""
    pdf_path = tmp_path / "nonexistent.pdf"

    result = is_valid_pdf(pdf_path)

    assert result is False


def test_is_valid_pdf_wrong_extension(tmp_path):
    """Test valid PDF check with wrong file extension."""
    pdf_path = tmp_path / "test.txt"
    pdf_path.touch()

    result = is_valid_pdf(pdf_path)

    assert result is False


def test_is_valid_pdf_corrupted(tmp_path):
    """Test valid PDF check with corrupted PDF."""
    pdf_path = tmp_path / "corrupted.pdf"
    pdf_path.touch()

    mock_fitz = MagicMock()
    mock_fitz.open.side_effect = Exception("Corrupted")

    with patch.dict('sys.modules', {'fitz': mock_fitz}):
        result = is_valid_pdf(pdf_path)

        assert result is False


def test_is_valid_pdf_directory(tmp_path):
    """Test valid PDF check with directory instead of file."""
    pdf_dir = tmp_path / "test.pdf"
    pdf_dir.mkdir()

    result = is_valid_pdf(pdf_dir)

    assert result is False
