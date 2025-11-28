"""Tests for document classifier."""

from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
import pytest

from docscan.classifier import DocumentClassifier


@pytest.fixture
def config():
    """Create test configuration."""
    return {
        "model": "test-model",
        "categories": ["invoice", "receipt", "contract", "other"],
        "mlx_config": {
            "max_tokens": 50,
            "temperature": 0.1,
        },
    }


@pytest.fixture
def config_with_cache_dir(config, tmp_path):
    """Create config with cache directory."""
    config["model_cache_dir"] = str(tmp_path / "cache")
    return config


@pytest.fixture
def mock_model():
    """Create mock model."""
    return MagicMock()


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer."""
    return MagicMock()


@pytest.fixture
def mock_model_manager(mock_model, mock_tokenizer):
    """Create mock ModelManager."""
    with patch('docscan.classifier.ModelManager') as MockModelManager:
        instance = MockModelManager.return_value
        instance.load_model.return_value = (mock_model, mock_tokenizer)
        yield MockModelManager


def test_classifier_init(config, mock_model_manager):
    """Test DocumentClassifier initialization."""
    classifier = DocumentClassifier(config)

    assert classifier.config == config
    assert classifier.model_id == "test-model"
    assert classifier.categories == ["invoice", "receipt", "contract", "other"]
    assert classifier.mlx_config == config["mlx_config"]
    assert classifier.model is not None
    assert classifier.tokenizer is not None


def test_classifier_init_with_cache_dir(config_with_cache_dir, mock_model_manager, tmp_path):
    """Test initialization with custom cache directory."""
    classifier = DocumentClassifier(config_with_cache_dir)

    # Verify ModelManager was initialized with expanded cache path
    mock_model_manager.assert_called_once()
    call_args = mock_model_manager.call_args[0]
    assert str(tmp_path / "cache") in str(call_args[0])


def test_classifier_init_no_model(mock_model_manager):
    """Test initialization without model ID."""
    config = {
        "categories": ["invoice", "other"],
    }

    with pytest.raises(ValueError, match="No model ID specified"):
        DocumentClassifier(config)


def test_classifier_init_model_load_failure(config):
    """Test initialization with model loading failure."""
    with patch('docscan.classifier.ModelManager') as MockModelManager:
        instance = MockModelManager.return_value
        instance.load_model.side_effect = Exception("Model not found")

        with pytest.raises(Exception, match="Model not found"):
            DocumentClassifier(config)


def test_load_model_success(config, mock_model_manager, mock_model, mock_tokenizer):
    """Test successful model loading."""
    classifier = DocumentClassifier(config)

    assert classifier.model == mock_model
    assert classifier.tokenizer == mock_tokenizer
    mock_model_manager.return_value.load_model.assert_called_once_with("test-model")


def test_classify_single_document(config, mock_model_manager, tmp_path):
    """Test classifying a single document."""
    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is an invoice document.")

    document = {
        "path": test_file,
        "extension": ".txt",
    }

    classifier = DocumentClassifier(config)

    with patch.object(classifier, '_infer_category', return_value='invoice'):
        result = classifier.classify(document)

    assert result["category"] == "invoice"
    assert result["text"] == "This is an invoice document."
    assert result["path"] == test_file


def test_classify_batch(config, mock_model_manager, tmp_path):
    """Test classifying multiple documents."""
    # Create test files
    file1 = tmp_path / "invoice.txt"
    file1.write_text("Invoice content")
    file2 = tmp_path / "receipt.txt"
    file2.write_text("Receipt content")

    documents = [
        {"path": file1, "extension": ".txt"},
        {"path": file2, "extension": ".txt"},
    ]

    classifier = DocumentClassifier(config)

    with patch.object(classifier, '_infer_category') as mock_infer:
        mock_infer.side_effect = ['invoice', 'receipt']

        results = classifier.classify_batch(documents)

    assert len(results) == 2
    assert results[0]["category"] == "invoice"
    assert results[1]["category"] == "receipt"


def test_extract_text_txt_file(config, mock_model_manager, tmp_path):
    """Test text extraction from .txt file."""
    test_file = tmp_path / "test.txt"
    test_content = "This is test content"
    test_file.write_text(test_content)

    document = {
        "path": test_file,
        "extension": ".txt",
    }

    classifier = DocumentClassifier(config)
    text = classifier._extract_text(document)

    assert text == test_content


def test_extract_text_unsupported_format(config, mock_model_manager):
    """Test text extraction from unsupported format."""
    document = {
        "path": Path("test.pdf"),
        "extension": ".pdf",
    }

    classifier = DocumentClassifier(config)
    text = classifier._extract_text(document)

    assert "not yet implemented" in text
    assert ".pdf" in text


def test_extract_text_various_extensions(config, mock_model_manager):
    """Test text extraction placeholder for various extensions."""
    extensions = [".pdf", ".docx", ".jpg", ".png"]

    classifier = DocumentClassifier(config)

    for ext in extensions:
        document = {
            "path": Path(f"test{ext}"),
            "extension": ext,
        }
        text = classifier._extract_text(document)
        assert ext in text
        assert "not yet implemented" in text


def test_create_classification_prompt(config, mock_model_manager):
    """Test prompt creation for classification."""
    classifier = DocumentClassifier(config)

    text = "This is a sample document text."
    prompt = classifier._create_classification_prompt(text)

    assert "invoice" in prompt
    assert "receipt" in prompt
    assert "contract" in prompt
    assert "other" in prompt
    assert text in prompt


def test_create_classification_prompt_truncates_long_text(config, mock_model_manager):
    """Test prompt truncates text to 1000 characters."""
    classifier = DocumentClassifier(config)

    text = "A" * 2000
    prompt = classifier._create_classification_prompt(text)

    # Should contain first 1000 chars + "..."
    assert "A" * 1000 in prompt
    assert "..." in prompt


def test_infer_category_success(config, mock_model_manager):
    """Test successful category inference."""
    classifier = DocumentClassifier(config)

    mock_generate = MagicMock(return_value="invoice")
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = mock_generate

    with patch.dict('sys.modules', {'mlx_lm': mock_mlx_lm}):
        category = classifier._infer_category("Test prompt")

    assert category == "invoice"
    mock_generate.assert_called_once()


def test_infer_category_uses_config(config, mock_model_manager):
    """Test inference uses MLX config parameters."""
    classifier = DocumentClassifier(config)

    mock_generate = MagicMock(return_value="invoice")
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = mock_generate

    with patch.dict('sys.modules', {'mlx_lm': mock_mlx_lm}):
        classifier._infer_category("Test prompt")

    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs['max_tokens'] == 50
    assert call_kwargs['temp'] == 0.1
    assert call_kwargs['verbose'] is False


def test_infer_category_model_not_loaded(config, mock_model_manager):
    """Test inference when model is not loaded."""
    classifier = DocumentClassifier(config)
    classifier.model = None

    category = classifier._infer_category("Test prompt")

    assert category == "other"


def test_infer_category_tokenizer_not_loaded(config, mock_model_manager):
    """Test inference when tokenizer is not loaded."""
    classifier = DocumentClassifier(config)
    classifier.tokenizer = None

    category = classifier._infer_category("Test prompt")

    assert category == "other"


def test_infer_category_exception(config, mock_model_manager):
    """Test inference with exception."""
    classifier = DocumentClassifier(config)

    mock_generate = MagicMock(side_effect=RuntimeError("Inference error"))
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = mock_generate

    with patch.dict('sys.modules', {'mlx_lm': mock_mlx_lm}):
        category = classifier._infer_category("Test prompt")

    assert category == "other"


def test_parse_category_exact_match(config, mock_model_manager):
    """Test parsing category with exact match."""
    classifier = DocumentClassifier(config)

    response = "invoice"
    category = classifier._parse_category_from_response(response)

    assert category == "invoice"


def test_parse_category_case_insensitive(config, mock_model_manager):
    """Test parsing is case insensitive."""
    classifier = DocumentClassifier(config)

    response = "INVOICE"
    category = classifier._parse_category_from_response(response)

    assert category == "invoice"


def test_parse_category_in_sentence(config, mock_model_manager):
    """Test parsing category from sentence."""
    classifier = DocumentClassifier(config)

    response = "This document is a receipt for the purchase."
    category = classifier._parse_category_from_response(response)

    assert category == "receipt"


def test_parse_category_first_word(config, mock_model_manager):
    """Test parsing category from first word."""
    classifier = DocumentClassifier(config)

    response = "contract for services"
    category = classifier._parse_category_from_response(response)

    assert category == "contract"


def test_parse_category_with_punctuation(config, mock_model_manager):
    """Test parsing category with punctuation."""
    classifier = DocumentClassifier(config)

    response = "invoice."
    category = classifier._parse_category_from_response(response)

    assert category == "invoice"


def test_parse_category_no_match(config, mock_model_manager):
    """Test parsing with no category match."""
    classifier = DocumentClassifier(config)

    response = "unknown document type"
    category = classifier._parse_category_from_response(response)

    # Should return last category (default)
    assert category == "other"


def test_parse_category_empty_response(config, mock_model_manager):
    """Test parsing empty response."""
    classifier = DocumentClassifier(config)

    response = ""
    category = classifier._parse_category_from_response(response)

    assert category == "other"


def test_parse_category_no_categories(config, mock_model_manager):
    """Test parsing when no categories are configured."""
    config["categories"] = []
    classifier = DocumentClassifier(config)

    response = "some text"
    category = classifier._parse_category_from_response(response)

    assert category == "other"


def test_parse_category_priority(config, mock_model_manager):
    """Test parsing prefers exact match over substring."""
    classifier = DocumentClassifier(config)

    # "invoice" appears in response, should match that category
    response = "invoice and receipt"
    category = classifier._parse_category_from_response(response)

    # Should match first occurrence
    assert category in ["invoice", "receipt"]


def test_classifier_default_mlx_config(mock_model_manager):
    """Test classifier with default MLX config."""
    config = {
        "model": "test-model",
        "categories": ["invoice", "other"],
    }

    classifier = DocumentClassifier(config)

    assert classifier.mlx_config == {}


def test_classifier_default_categories(mock_model_manager):
    """Test classifier with default categories."""
    config = {
        "model": "test-model",
    }

    classifier = DocumentClassifier(config)

    assert classifier.categories == []


def test_end_to_end_classification(config, mock_model_manager, tmp_path):
    """Test end-to-end document classification."""
    # Create test document
    test_file = tmp_path / "invoice.txt"
    test_file.write_text("Invoice #12345\nDate: 2024-01-15\nAmount: $100.00")

    document = {
        "path": test_file,
        "extension": ".txt",
        "name": "invoice.txt",
    }

    classifier = DocumentClassifier(config)

    mock_generate = MagicMock(return_value="This is an invoice document")
    mock_mlx_lm = MagicMock()
    mock_mlx_lm.generate = mock_generate

    with patch.dict('sys.modules', {'mlx_lm': mock_mlx_lm}):
        result = classifier.classify(document)

    assert result["category"] == "invoice"
    assert "Invoice #12345" in result["text"]
    assert result["name"] == "invoice.txt"
