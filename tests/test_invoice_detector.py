"""Tests for invoice detection and extraction."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest
from PIL import Image

from docscan.invoice_detector import (
    InvoiceDetector,
    generate_invoice_filename,
)


@pytest.fixture
def mock_model():
    """Create mock VLM model."""
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MagicMock()


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "vlm_config": {
            "max_tokens": 200,
            "temperature": 0.1,
        }
    }


@pytest.fixture
def invoice_detector(mock_model, mock_tokenizer, config):
    """Create InvoiceDetector instance."""
    return InvoiceDetector(mock_model, mock_tokenizer, config)


@pytest.fixture
def mock_image():
    """Create mock PIL image."""
    return Image.new('RGB', (100, 100))


def test_invoice_detector_init(mock_model, mock_tokenizer, config):
    """Test InvoiceDetector initialization."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config)

    assert detector.model == mock_model
    assert detector.tokenizer == mock_tokenizer
    assert detector.config == config
    assert detector.vlm_config == config["vlm_config"]


def test_invoice_detector_init_no_vlm_config(mock_model, mock_tokenizer):
    """Test initialization without vlm_config."""
    config = {}
    detector = InvoiceDetector(mock_model, mock_tokenizer, config)

    assert detector.vlm_config == {}


def test_analyze_document_invalid_pdf(invoice_detector, tmp_path):
    """Test analyze_document with invalid PDF."""
    pdf_path = tmp_path / "invalid.pdf"
    pdf_path.touch()

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=False):
        is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is False
    assert data is None


def test_analyze_document_no_pages(invoice_detector, tmp_path):
    """Test analyze_document when PDF has no pages."""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.touch()

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=True):
        with patch('docscan.invoice_detector.pdf_to_images', return_value=[]):
            is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is False
    assert data is None


def test_analyze_document_not_invoice(invoice_detector, tmp_path, mock_image):
    """Test analyze_document when document is not an invoice."""
    pdf_path = tmp_path / "document.pdf"
    pdf_path.touch()

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=True):
        with patch('docscan.invoice_detector.pdf_to_images', return_value=[mock_image]):
            with patch.object(invoice_detector, '_detect_invoice', return_value=False):
                is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is False
    assert data is None


def test_analyze_document_success(invoice_detector, tmp_path, mock_image):
    """Test successful invoice analysis."""
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp"
    }

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=True):
        with patch('docscan.invoice_detector.pdf_to_images', return_value=[mock_image]):
            with patch.object(invoice_detector, '_detect_invoice', return_value=True):
                with patch.object(invoice_detector, '_extract_invoice_data', return_value=invoice_data):
                    is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is True
    assert data is not None
    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME_Corp"
    assert data["original_filename"] == "invoice.pdf"


def test_analyze_document_exception(invoice_detector, tmp_path):
    """Test analyze_document with exception during processing."""
    pdf_path = tmp_path / "error.pdf"
    pdf_path.touch()

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=True):
        with patch('docscan.invoice_detector.pdf_to_images', side_effect=Exception("PDF error")):
            is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is False
    assert data is None


def test_detect_invoice_yes(invoice_detector, mock_image):
    """Test invoice detection with YES response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="YES"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_ja(invoice_detector, mock_image):
    """Test invoice detection with JA (German) response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="JA"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_no(invoice_detector, mock_image):
    """Test invoice detection with NO response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="NO"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is False


def test_detect_invoice_case_insensitive(invoice_detector, mock_image):
    """Test invoice detection is case insensitive."""
    with patch.object(invoice_detector, '_query_vlm', return_value="yes"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_with_extra_text(invoice_detector, mock_image):
    """Test invoice detection with extra text in response."""
    response = "After analyzing the document, the answer is YES, this is an invoice."
    with patch.object(invoice_detector, '_query_vlm', return_value=response):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_exception(invoice_detector, mock_image):
    """Test invoice detection with exception."""
    with patch.object(invoice_detector, '_query_vlm', side_effect=Exception("VLM error")):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is False


def test_extract_invoice_data_success(invoice_detector, mock_image):
    """Test successful invoice data extraction."""
    response = "DATE: 2024-01-15\nPARTY: ACME Corporation"

    with patch.object(invoice_detector, '_query_vlm', return_value=response):
        with patch.object(invoice_detector, '_parse_extraction_response') as mock_parse:
            mock_parse.return_value = {
                "date": "2024-01-15",
                "invoicing_party": "ACME_Corporation"
            }

            data = invoice_detector._extract_invoice_data(mock_image)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME_Corporation"


def test_extract_invoice_data_exception(invoice_detector, mock_image):
    """Test invoice data extraction with exception."""
    with patch.object(invoice_detector, '_query_vlm', side_effect=Exception("Extraction error")):
        data = invoice_detector._extract_invoice_data(mock_image)

    assert data["date"] == "0000-00-00"
    assert data["invoicing_party"] == "Unknown"


def test_query_vlm_success(invoice_detector, mock_image):
    """Test successful VLM query."""
    mock_generate = MagicMock(return_value="Test response")
    mock_vlm = MagicMock()
    mock_vlm.generate = mock_generate
    mock_vlm.utils = MagicMock()

    with patch.dict('sys.modules', {'mlx_vlm': mock_vlm, 'mlx_vlm.utils': mock_vlm.utils}):
        response = invoice_detector._query_vlm(mock_image, "Test prompt")

    assert response == "Test response"
    mock_generate.assert_called_once()


def test_query_vlm_uses_config(invoice_detector, mock_image):
    """Test VLM query uses configuration parameters."""
    mock_generate = MagicMock(return_value="Response")
    mock_vlm = MagicMock()
    mock_vlm.generate = mock_generate
    mock_vlm.utils = MagicMock()

    with patch.dict('sys.modules', {'mlx_vlm': mock_vlm, 'mlx_vlm.utils': mock_vlm.utils}):
        invoice_detector._query_vlm(mock_image, "Prompt")

    # Check that generate was called with correct parameters
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs['max_tokens'] == 200
    assert call_kwargs['temp'] == 0.1
    assert call_kwargs['verbose'] is False


def test_query_vlm_import_error(invoice_detector, mock_image):
    """Test VLM query with import error fallback."""
    # Remove mlx_vlm from sys.modules to trigger ImportError
    original_modules = sys.modules.copy()
    if 'mlx_vlm' in sys.modules:
        del sys.modules['mlx_vlm']
    if 'mlx_vlm.utils' in sys.modules:
        del sys.modules['mlx_vlm.utils']

    try:
        with patch.dict('sys.modules', {'mlx_vlm': None}):
            with patch('builtins.__import__', side_effect=ImportError("No mlx_vlm")):
                response = invoice_detector._query_vlm(mock_image, "Test prompt")

        # Should return placeholder
        assert "YES" in response
        assert "DATE" in response
        assert "PARTY" in response
    finally:
        # Restore original modules
        sys.modules.update(original_modules)


def test_query_vlm_exception(invoice_detector, mock_image):
    """Test VLM query with exception."""
    mock_generate = MagicMock(side_effect=Exception("VLM failure"))
    mock_vlm = MagicMock()
    mock_vlm.generate = mock_generate
    mock_vlm.utils = MagicMock()

    with patch.dict('sys.modules', {'mlx_vlm': mock_vlm, 'mlx_vlm.utils': mock_vlm.utils}):
        with pytest.raises(Exception, match="VLM failure"):
            invoice_detector._query_vlm(mock_image, "Test prompt")


def test_parse_extraction_response_full_format(invoice_detector):
    """Test parsing response with full DATE/PARTY format."""
    response = "DATE: 2024-01-15\nPARTY: ACME Corp"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME_Corp"


def test_parse_extraction_response_case_insensitive(invoice_detector):
    """Test parsing is case insensitive."""
    response = "date: 2024-01-15\nparty: ACME Corp"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME_Corp"


def test_parse_extraction_response_date_only(invoice_detector):
    """Test parsing response with date in text."""
    response = "The invoice date is 2024-01-15"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "Unknown"


def test_parse_extraction_response_invalid_date(invoice_detector):
    """Test parsing with invalid date format."""
    response = "DATE: 2024-13-45\nPARTY: ACME Corp"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "0000-00-00"  # Should fall back to placeholder
    assert data["invoicing_party"] == "ACME_Corp"


def test_parse_extraction_response_no_data(invoice_detector):
    """Test parsing response with no recognizable data."""
    response = "I could not find the information"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "0000-00-00"
    assert data["invoicing_party"] == "Unknown"


def test_parse_extraction_response_sanitizes_party(invoice_detector):
    """Test that party name is sanitized."""
    response = "DATE: 2024-01-15\nPARTY: ACME Corp Inc."

    with patch.object(invoice_detector, '_sanitize_filename_part') as mock_sanitize:
        mock_sanitize.return_value = "ACME_Corp_Inc"

        data = invoice_detector._parse_extraction_response(response)

    mock_sanitize.assert_called_once()
    assert data["invoicing_party"] == "ACME_Corp_Inc"


def test_sanitize_filename_part_removes_invalid_chars(invoice_detector):
    """Test sanitization removes invalid filename characters."""
    text = 'Test<>:"/\\|?*Company'

    result = invoice_detector._sanitize_filename_part(text)

    assert result == "TestCompany"


def test_sanitize_filename_part_replaces_spaces(invoice_detector):
    """Test sanitization replaces spaces with underscores."""
    text = "ACME Corp Inc"

    result = invoice_detector._sanitize_filename_part(text)

    assert result == "ACME_Corp_Inc"


def test_sanitize_filename_part_removes_multiple_underscores(invoice_detector):
    """Test sanitization collapses multiple underscores."""
    text = "ACME___Corp___Inc"

    result = invoice_detector._sanitize_filename_part(text)

    assert result == "ACME_Corp_Inc"


def test_sanitize_filename_part_strips_underscores(invoice_detector):
    """Test sanitization removes leading/trailing underscores."""
    text = "___ACME Corp___"

    result = invoice_detector._sanitize_filename_part(text)

    assert result == "ACME_Corp"


def test_sanitize_filename_part_limits_length(invoice_detector):
    """Test sanitization limits filename length to 50 chars."""
    text = "A" * 100

    result = invoice_detector._sanitize_filename_part(text)

    assert len(result) == 50


def test_sanitize_filename_part_empty_returns_unknown(invoice_detector):
    """Test sanitization of empty string returns 'Unknown'."""
    result = invoice_detector._sanitize_filename_part("")

    assert result == "Unknown"


def test_sanitize_filename_part_only_invalid_chars(invoice_detector):
    """Test sanitization with only invalid characters."""
    text = "<>:\"/\\|?*"

    result = invoice_detector._sanitize_filename_part(text)

    assert result == "Unknown"


def test_generate_invoice_filename_full_data():
    """Test filename generation with complete data."""
    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp"
    }

    filename = generate_invoice_filename(invoice_data)

    assert filename == "2024-01-15_Rechnung_ACME_Corp.pdf"


def test_generate_invoice_filename_missing_date():
    """Test filename generation with missing date."""
    invoice_data = {
        "invoicing_party": "ACME_Corp"
    }

    filename = generate_invoice_filename(invoice_data)

    assert filename == "0000-00-00_Rechnung_ACME_Corp.pdf"


def test_generate_invoice_filename_missing_party():
    """Test filename generation with missing party."""
    invoice_data = {
        "date": "2024-01-15"
    }

    filename = generate_invoice_filename(invoice_data)

    assert filename == "2024-01-15_Rechnung_Unknown.pdf"


def test_generate_invoice_filename_empty_data():
    """Test filename generation with empty data."""
    invoice_data = {}

    filename = generate_invoice_filename(invoice_data)

    assert filename == "0000-00-00_Rechnung_Unknown.pdf"


def test_generate_invoice_filename_preserves_extension():
    """Test filename always ends with .pdf."""
    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "Test"
    }

    filename = generate_invoice_filename(invoice_data)

    assert filename.endswith(".pdf")
