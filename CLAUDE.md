# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Workflow

**IMPORTANT: Always use GitHub Flow for bug fixes and feature development**

When fixing bugs or implementing features, you MUST:

1. **Create a new branch** from `main` with a descriptive name:
   ```bash
   git checkout -b fix/issue-description
   # or
   git checkout -b feature/feature-name
   ```

2. **Commit changes** to the branch (NOT directly to `main`)
   - Use clear, descriptive commit messages
   - Include the Claude Code co-authorship footer

3. **Open a Pull Request** when ready for review:
   ```bash
   gh pr create --title "Fix: Description" --body "..."
   ```

4. **Do NOT commit directly to `main`** - all changes should go through Pull Requests

Branch naming conventions:
- Bug fixes: `fix/short-description`
- Features: `feature/short-description`
- Refactoring: `refactor/short-description`
- Documentation: `docs/short-description`

## Project Overview

**doc-scan-intelligent-operator** - An AI-powered invoice detection and renaming system that uses Vision-Language Models (VLMs) with MLX acceleration for Apple Silicon. The system analyzes PDF invoices, extracts key information (date, invoicing party), and renames files intelligently.

**Key Focus**: Invoice processing with format `YYYY-MM-DD_Rechnung_Company.pdf`

## Technology Stack

- **Language**: Python 3.9+
- **AI/ML Framework**: MLX VLM (Vision-Language Models on Apple Silicon)
- **Models**: Qwen-VL, Pixtral, other MLX-compatible VLMs
- **Package Manager**: pip + requirements.txt
- **Interface**: Command-line interface (CLI)
- **Document Processing**: PyMuPDF (PDF to image), Pillow (image handling)
- **Configuration**: YAML-based
- **Testing**: pytest with coverage and mocking

## Development Commands

### Setup
```bash
# Install in development mode with dev dependencies
make install-dev

# Or manually:
pip install -r requirements-dev.txt
pip install -e .
```

### Running Tests
```bash
# Run all tests with coverage
make test

# Run specific test file
pytest tests/test_scanner.py

# Run specific test
pytest tests/test_scanner.py::test_scan_single_file

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Run linters (ruff + mypy)
make lint

# Format code with black
make format

# Clean build artifacts
make clean
```

### Running the Application
```bash
# Analyze and rename an invoice
docscan invoice.pdf

# Dry run (preview without renaming)
docscan invoice.pdf --dry-run

# With custom config
docscan invoice.pdf -c config.yaml

# With different VLM model
docscan invoice.pdf -m Qwen/Qwen2-VL-2B-Instruct

# Verbose output
docscan invoice.pdf -v
```

## Project Architecture

### Core Components

The project follows a modular architecture focused on invoice processing:

1. **PDF Utils (`docscan/pdf_utils.py`)** - PDF to image conversion
   - Converts PDF pages to PIL Images using PyMuPDF
   - Validates PDF files
   - Optimized for invoice processing (150 DPI)

2. **Invoice Detector (`docscan/invoice_detector.py`)** - VLM-based analysis
   - Detects if document is an invoice
   - Extracts invoice date (Rechnungsdatum)
   - Extracts invoicing party (company name)
   - Supports multilingual invoices (DE, EN, FR, ES)
   - Generates standardized filenames

3. **Model Manager (`docscan/model_manager.py`)** - VLM loading and caching
   - Loads VLMs with MLX optimization
   - Supports both text-only and vision-language models
   - Caches models locally
   - Validates MLX compatibility and disk space

4. **File Renamer (`docscan/file_renamer.py`)** - Safe file renaming
   - Renames files with extracted data
   - Handles filename conflicts
   - Supports in-place or directory-based renaming

5. **Auth (`docscan/auth.py`)** - HuggingFace authentication
   - Environment variable support
   - Token file support
   - Secure credential management

6. **CLI (`docscan/cli.py`)** - Command-line interface
   - Single PDF file processing
   - Clear user feedback
   - Dry-run mode for testing

### Data Flow

```
PDF File → PDF to Images → VLM Analysis → Invoice Detection → Data Extraction → File Renaming
```

### Invoice Processing Workflow

1. **Validation**: Check if file is valid PDF
2. **Conversion**: Convert first page to image (150 DPI)
3. **Detection**: VLM determines if document is invoice
4. **Extraction**: If invoice, extract:
   - Invoice date (YYYY-MM-DD format)
   - Invoicing party (company name)
5. **Filename Generation**: Create `YYYY-MM-DD_Rechnung_Company.pdf`
6. **Renaming**: Safely rename file with collision handling

### Configuration

Configuration is loaded from YAML files with the following precedence:
1. CLI arguments (highest priority)
2. Custom config file (via `-c/--config`)
3. Default configuration in `config.py`

See `config.example.yaml` for all available options.

### MLX Integration

This project leverages MLX for Apple Silicon acceleration:
- Models are loaded from Hugging Face using `mlx-lm`
- Inference runs natively on Apple Neural Engine
- Models are automatically downloaded and cached locally
- MLX compatibility is verified before downloading
- Disk space is checked to ensure sufficient storage

**Key Components:**
- `model_manager.py` - Handles model loading, caching, and lifecycle
- `mlx_utils.py` - MLX compatibility checking and disk space management
- `classifier.py:_load_model()` - Loads model via ModelManager
- `classifier.py:_infer_category()` - Runs MLX inference with mlx_lm.generate()

**Model Caching:**
- Default cache: `~/.cache/docscan/models`
- Configurable via `model_cache_dir` in config
- Models are cached after first download
- Loaded models stay in memory to avoid reloading

### Text Extraction

Text extraction varies by document type:
- **PDF**: Use pypdf or pdfplumber
- **Images**: OCR with pytesseract (optional dependency)
- **Text files**: Direct reading
- **DOCX**: python-docx library
- **Current Status**: Only `.txt` files implemented, others are TODOs in `docscan/classifier.py:_extract_text()`

### HuggingFace Authentication

The system supports multiple authentication methods (in priority order):
1. **Environment variable** `HF_TOKEN` (highest priority)
2. **Environment variable** `HUGGING_FACE_TOKEN`
3. **Token file** `~/.huggingface/token`
4. **Token file** `~/.config/huggingface/token`

Authentication is handled automatically when loading models. Some models (like Llama-2) require authentication.

**Setting up authentication:**
```bash
# Option 1: Environment variable (recommended for scripts)
export HF_TOKEN="your_token_here"

# Option 2: Token file (persistent)
mkdir -p ~/.huggingface
echo "your_token_here" > ~/.huggingface/token
```

### Testing Strategy

Tests use pytest with fixtures and mocking:
- `test_scanner.py` - Document scanning and filtering
- `test_config.py` - Configuration loading/saving
- `test_organizer.py` - File organization logic
- `test_auth.py` - HuggingFace authentication
- `test_mlx_utils.py` - MLX compatibility and disk space checks
- `test_model_manager.py` - Model loading and caching
- Coverage target: Maintain good coverage as features are added

## Key Implementation Notes

### Adding New Document Formats

To support a new document format:
1. Add extension to `supported_formats` in `config.py` or config file
2. Implement text extraction in `classifier.py:_extract_text()`
3. Add tests in `tests/test_scanner.py`

### Changing Classification Categories

Categories are configured in `config.yaml` or `DEFAULT_CONFIG`. The classifier uses these categories in the prompt sent to the LLM.

### Custom Models

Any Hugging Face text generation model can be used:
- Smaller models (phi-2, Mistral-7B) work well for classification
- Larger models require more memory but may be more accurate
- Models must be MLX-compatible (Llama, Mistral, Phi, Qwen, GPT-2, StableLM, Mixtral)
- Model compatibility is automatically checked before download
- Disk space is verified before downloading large models

**Configuring model cache location:**
```yaml
# config.yaml
model_cache_dir: /Volumes/ExternalDrive/models  # For large models
```

### Troubleshooting

**Insufficient disk space:**
If you get a disk space error, you can:
1. Free up space in the cache directory
2. Configure a different cache directory with more space
3. Use a smaller model

**MLX compatibility errors:**
If a model is not MLX-compatible:
1. Check if the model architecture is supported (see MLX Integration section)
2. Try a different model from the same family
3. Check MLX documentation for newly supported architectures

**Authentication errors:**
If you can't access a model:
1. Verify your HuggingFace token is valid
2. Check if the model requires authentication
3. Ensure you have accepted the model's license on HuggingFace
