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
    assert detector.processor == mock_tokenizer
    assert detector.config == config
    assert detector.vlm_config == config["vlm_config"]
    assert detector.use_text_llm is False


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
        with patch('docscan.invoice_detector.pdf_to_images', side_effect=RuntimeError("PDF error")):
            is_invoice, data = invoice_detector.analyze_document(pdf_path)

    assert is_invoice is False
    assert data is None


def test_detect_invoice_yes(invoice_detector, mock_image):
    """Test invoice detection with INVOICE response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="INVOICE"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_rechnung(invoice_detector, mock_image):
    """Test invoice detection with RECHNUNG (German) response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="RECHNUNG"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_no(invoice_detector, mock_image):
    """Test invoice detection with NO response."""
    with patch.object(invoice_detector, '_query_vlm', return_value="NO"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is False


def test_detect_invoice_case_insensitive(invoice_detector, mock_image):
    """Test invoice detection is case insensitive."""
    with patch.object(invoice_detector, '_query_vlm', return_value="invoice"):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_with_extra_text(invoice_detector, mock_image):
    """Test invoice detection with extra text in response."""
    response = "After analyzing the document, this is an INVOICE."
    with patch.object(invoice_detector, '_query_vlm', return_value=response):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is True


def test_detect_invoice_exception(invoice_detector, mock_image):
    """Test invoice detection with exception."""
    with patch.object(invoice_detector, '_query_vlm', side_effect=RuntimeError("VLM error")):
        result = invoice_detector._detect_invoice(mock_image)

    assert result is False


def test_extract_invoice_data_success(invoice_detector, mock_image):
    """Test successful invoice data extraction via OCR."""
    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "Rechnungsdatum: 15.01.2024\nChiropraktik White\nTotal: 100.00"

    with patch.dict('sys.modules', {'pytesseract': mock_pytesseract}):
        with patch.object(invoice_detector, '_parse_ocr_text') as mock_parse:
            mock_parse.return_value = {
                "date": "2024-01-15",
                "invoicing_party": "Chiropraktik_White"
            }

            data = invoice_detector._extract_invoice_data(mock_image)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "Chiropraktik_White"
    assert data["extraction_method"] == "OCR (pytesseract)"


def test_extract_invoice_data_ocr_fallback_to_vlm(invoice_detector, mock_image):
    """Test invoice data extraction falls back to VLM when OCR not available."""
    # Simulate pytesseract not being installed by raising ImportError
    with patch.dict('sys.modules', {'pytesseract': None}):
        with patch.object(invoice_detector, '_extract_invoice_data_vlm') as mock_vlm_extract:
            mock_vlm_extract.return_value = {
                "date": "2024-01-15",
                "invoicing_party": "ACME_Corp"
            }

            data = invoice_detector._extract_invoice_data(mock_image)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME_Corp"
    assert data["extraction_method"] == "VLM"


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
    mock_result = MagicMock()
    mock_result.text = "Response"
    mock_generate = MagicMock(return_value=mock_result)
    mock_vlm = MagicMock()
    mock_vlm.generate = mock_generate
    mock_vlm.utils = MagicMock()

    with patch.dict('sys.modules', {'mlx_vlm': mock_vlm, 'mlx_vlm.utils': mock_vlm.utils}):
        invoice_detector._query_vlm(mock_image, "Prompt")

    # Check that generate was called with correct parameters
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs['max_tokens'] == 200
    assert abs(call_kwargs['temperature'] - 0.1) < 0.001
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
    assert data["invoicing_party"] == "ACME Corp"  # Display name with spaces
    assert data["invoicing_party_filename"] == "ACME_Corp"  # Filename with underscores


def test_parse_extraction_response_case_insensitive(invoice_detector):
    """Test parsing is case insensitive."""
    response = "date: 2024-01-15\nparty: ACME Corp"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "2024-01-15"
    assert data["invoicing_party"] == "ACME Corp"  # Display name with spaces
    assert data["invoicing_party_filename"] == "ACME_Corp"  # Filename with underscores


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
    assert data["invoicing_party"] == "ACME Corp"  # Display name with spaces
    assert data["invoicing_party_filename"] == "ACME_Corp"  # Filename with underscores


def test_parse_extraction_response_no_data(invoice_detector):
    """Test parsing response with no recognizable data."""
    response = "I could not find the information"

    data = invoice_detector._parse_extraction_response(response)

    assert data["date"] == "0000-00-00"
    assert data["invoicing_party"] == "Unknown"


def test_parse_extraction_response_sanitizes_party(invoice_detector):
    """Test that party name is sanitized for filename and cleaned for display."""
    response = "DATE: 2024-01-15\nPARTY: ACME Corp Inc."

    data = invoice_detector._parse_extraction_response(response)

    # Display name should have spaces, filename should have underscores
    assert data["invoicing_party"] == "ACME Corp Inc."  # Display version
    assert data["invoicing_party_filename"] == "ACME_Corp_Inc."  # Filename version


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


# ============================================================================
# Text LLM Mode Tests
# ============================================================================

def test_invoice_detector_init_text_llm_mode(mock_model, mock_tokenizer, config):
    """Test InvoiceDetector initialization in text LLM mode."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    assert detector.model == mock_model
    assert detector.processor == mock_tokenizer
    assert detector.config == config
    assert detector.use_text_llm is True


def test_analyze_document_text_llm_mode(mock_model, mock_tokenizer, config, tmp_path, mock_image):
    """Test analyze_document in text LLM mode."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)
    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME Corp",
        "detection_method": "Text LLM",
        "extraction_method": "OCR + Text LLM"
    }

    with patch('docscan.invoice_detector.is_valid_pdf', return_value=True):
        with patch('docscan.invoice_detector.pdf_to_images', return_value=[mock_image]):
            with patch.object(detector, '_analyze_with_text_llm', return_value=(True, invoice_data)):
                is_invoice, data = detector.analyze_document(pdf_path)

    assert is_invoice is True
    assert data["date"] == "2024-01-15"
    assert data["detection_method"] == "Text LLM"


def test_analyze_with_text_llm_success(mock_model, mock_tokenizer, config, mock_image):
    """Test successful text LLM analysis."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    mock_pytesseract = MagicMock()
    mock_pytesseract.image_to_string.return_value = "Rechnungsdatum: 15.01.2024\nChiropraktik White"

    with patch.dict('sys.modules', {'pytesseract': mock_pytesseract}):
        with patch.object(detector, '_query_text_llm_for_invoice') as mock_query:
            mock_query.return_value = (True, {
                "date": "2024-01-15",
                "invoicing_party": "Chiropraktik White"
            })

            is_invoice, data = detector._analyze_with_text_llm(mock_image)

    assert is_invoice is True
    assert data["date"] == "2024-01-15"
    assert data["detection_method"] == "Text LLM"
    assert data["extraction_method"] == "OCR + Text LLM"


def test_analyze_with_text_llm_pytesseract_unavailable(mock_model, mock_tokenizer, config, mock_image):
    """Test text LLM analysis when pytesseract is unavailable."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    with patch.dict('sys.modules', {'pytesseract': None}):
        is_invoice, data = detector._analyze_with_text_llm(mock_image)

    assert is_invoice is False
    assert data is None


def test_query_text_llm_for_invoice_success(mock_model, mock_tokenizer, config):
    """Test successful text LLM query for invoice detection."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    llm_response = """INVOICE: YES
DATE: 2024-01-15
ISSUER: Chiropraktik White"""

    with patch.object(detector, '_query_text_llm', return_value=llm_response):
        is_invoice, data = detector._query_text_llm_for_invoice("Sample OCR text")

    assert is_invoice is True
    assert data["date"] == "2024-01-15"
    assert "Chiropraktik" in data["invoicing_party"] or "White" in data["invoicing_party"]


def test_query_text_llm_for_invoice_not_invoice(mock_model, mock_tokenizer, config):
    """Test text LLM query when document is not an invoice."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    llm_response = """INVOICE: NO
DATE: N/A
ISSUER: N/A"""

    with patch.object(detector, '_query_text_llm', return_value=llm_response):
        is_invoice, data = detector._query_text_llm_for_invoice("Sample OCR text")

    assert is_invoice is False


def test_query_text_llm(mock_model, mock_tokenizer, config):
    """Test text LLM query execution."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    # Mock mlx_lm modules
    mock_generate = MagicMock(return_value="Test response from LLM")
    mock_make_sampler = MagicMock(return_value=lambda: None)
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = mock_generate
    mock_sample_utils = MagicMock()
    mock_sample_utils.make_sampler = mock_make_sampler

    with patch.dict('sys.modules', {'mlx_lm': mock_mlx_lm, 'mlx_lm.sample_utils': mock_sample_utils}):
        with patch.object(detector, '_build_chat_prompt', return_value="formatted prompt"):
            response = detector._query_text_llm("system", "user")

    assert response == "Test response from LLM"
    mock_generate.assert_called_once()


def test_build_chat_prompt_with_chat_template(mock_model, mock_tokenizer, config):
    """Test chat prompt building with tokenizer that has chat template."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    mock_tok = MagicMock()
    mock_tok.apply_chat_template = MagicMock(return_value="<chat>formatted</chat>")

    prompt = detector._build_chat_prompt(mock_tok, "System message", "User message")

    assert prompt == "<chat>formatted</chat>"
    mock_tok.apply_chat_template.assert_called_once()


def test_build_chat_prompt_fallback(mock_model, mock_tokenizer, config):
    """Test chat prompt building fallback when no chat template."""
    detector = InvoiceDetector(mock_model, mock_tokenizer, config, use_text_llm=True)

    mock_tok = MagicMock()
    mock_tok.apply_chat_template = MagicMock(side_effect=RuntimeError("No template"))

    prompt = detector._build_chat_prompt(mock_tok, "System message", "User message")

    assert "System message" in prompt
    assert "User message" in prompt


# ============================================================================
# OCR Text Parsing Tests
# ============================================================================

def test_parse_ocr_text(invoice_detector):
    """Test OCR text parsing."""
    ocr_text = """Chiropraktik White
Rechnungsdatum: 15.01.2024
Total: 100.00 CHF"""

    data = invoice_detector._parse_ocr_text(ocr_text)

    assert "date" in data
    assert "invoicing_party" in data
    assert "invoicing_party_filename" in data


def test_extract_date_from_text_rechnungsdatum(invoice_detector):
    """Test date extraction with Rechnungsdatum pattern."""
    text = "Rechnungsdatum: 15.01.2024"

    date = invoice_detector._extract_date_from_text(text)

    assert date == "2024-01-15"


def test_extract_date_from_text_invoice_date(invoice_detector):
    """Test date extraction with Invoice Date pattern."""
    text = "Invoice Date: 01/15/2024"

    date = invoice_detector._extract_date_from_text(text)

    assert "2024" in date


def test_extract_date_from_text_no_match(invoice_detector):
    """Test date extraction with no date found."""
    text = "No date here"

    date = invoice_detector._extract_date_from_text(text)

    assert date == "0000-00-00"


def test_extract_company_from_text_practice_name(invoice_detector):
    """Test company extraction finds practice names."""
    text = """Chiropraktik White
Address line
Phone: 123456"""

    display, filename = invoice_detector._extract_company_from_text(text)

    assert "Chiropraktik" in display or "White" in display


def test_find_practice_name(invoice_detector):
    """Test finding practice name patterns."""
    lines = [
        "Header",
        "Chiropraktik White",
        "Address"
    ]

    result = invoice_detector._find_practice_name(lines)

    assert result is not None
    assert "Chiropraktik" in result[0] or "White" in result[0]


def test_find_legal_entity_gmbh(invoice_detector):
    """Test finding GmbH legal entity."""
    lines = [
        "ACME Solutions GmbH",
        "Address line"
    ]

    result = invoice_detector._find_legal_entity(lines)

    assert result is not None
    assert "ACME" in result[0] and "GmbH" in result[0]


def test_find_legal_entity_ag(invoice_detector):
    """Test finding AG legal entity."""
    lines = [
        "Tech Corp AG",
        "Address line"
    ]

    result = invoice_detector._find_legal_entity(lines)

    assert result is not None
    assert "AG" in result[0]


def test_find_header_company(invoice_detector):
    """Test finding company in header area."""
    lines = [
        "Company Name Here",
        "Address",
        "Phone"
    ]

    display, filename = invoice_detector._find_header_company(lines)

    assert display == "Company Name Here"


def test_is_potential_company_line_valid(invoice_detector):
    """Test potential company line detection - valid."""
    line = "ACME Corporation"

    result = invoice_detector._is_potential_company_line(line)

    assert result is True


def test_is_potential_company_line_too_short(invoice_detector):
    """Test potential company line detection - too short."""
    line = "AB"

    result = invoice_detector._is_potential_company_line(line)

    assert result is False


def test_convert_date_to_iso_ddmmyyyy(invoice_detector):
    """Test date conversion from DD.MM.YYYY."""
    date_str = "15.01.2024"

    iso_date = invoice_detector._convert_date_to_iso(date_str)

    assert iso_date == "2024-01-15"


def test_convert_date_to_iso_mmddyyyy(invoice_detector):
    """Test date conversion from MM/DD/YYYY."""
    date_str = "01/15/2024"

    iso_date = invoice_detector._convert_date_to_iso(date_str)

    assert iso_date == "2024-01-15"


def test_convert_date_to_iso_invalid(invoice_detector):
    """Test date conversion with invalid date."""
    date_str = "99/99/9999"

    iso_date = invoice_detector._convert_date_to_iso(date_str)

    assert iso_date == "0000-00-00"


def test_clean_display_name(invoice_detector):
    """Test display name cleaning."""
    text = "  ACME  Corp  "

    cleaned = invoice_detector._clean_display_name(text)

    assert cleaned == "ACME Corp"


def test_clean_display_name_removes_extra_spaces(invoice_detector):
    """Test display name removes multiple spaces."""
    text = "ACME    Corporation    Inc"

    cleaned = invoice_detector._clean_display_name(text)

    assert "    " not in cleaned
    assert "ACME" in cleaned
