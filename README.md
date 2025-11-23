# doc-scan-intelligent-operator

An AI-powered invoice detection and renaming system using Vision-Language Models (VLMs) with MLX acceleration for Apple Silicon.

## Features

- 🧠 **Vision-Language Models**: Uses state-of-the-art VLMs to analyze PDF invoices directly (no OCR needed!)
- 🔍 **Smart Detection**: Automatically detects if a document is an invoice
- 📅 **Date Extraction**: Extracts invoice date for filename prefix (YYYY-MM-DD format)
- 🏢 **Company Extraction**: Identifies invoicing party from document
- 📝 **Intelligent Renaming**: Renames files as `YYYY-MM-DD_Rechnung_Company.pdf`
- ⚡ **MLX Optimized**: Fast inference on Apple Silicon
- 🌍 **Multilingual**: Supports German, English, French, and Spanish invoices
- 💾 **Smart Caching**: Models cached locally to avoid re-downloading
- ✅ **Safety Checks**: MLX compatibility and disk space validation
- 🔐 **Secure Auth**: Multiple HuggingFace authentication methods

## Quick Start

### Installation

```bash
# Install dependencies
make install-dev

# Or manually
pip install -r requirements-dev.txt
pip install -e .
```

### Authentication (Optional)

For models that require authentication (e.g., Llama-2):

```bash
# Set environment variable
export HF_TOKEN="your_huggingface_token"

# Or create token file
mkdir -p ~/.huggingface
echo "your_huggingface_token" > ~/.huggingface/token
```

### Basic Usage

```bash
# Analyze and rename an invoice
docscan invoice.pdf

# Preview without renaming (dry run)
docscan invoice.pdf --dry-run

# Use custom configuration
docscan invoice.pdf -c config.yaml

# Use a different VLM model
docscan invoice.pdf -m Qwen/Qwen2-VL-2B-Instruct

# Verbose output for debugging
docscan invoice.pdf -v
```

**Example Output:**
```
INFO: Loading vision-language model: Qwen/Qwen2-VL-7B-Instruct
INFO: Model loaded successfully!
INFO: Analyzing document: rechnung_scan.pdf

============================================================
✓ INVOICE DETECTED
============================================================
Invoice Date:      2024-01-15
Invoicing Party:   ACME_Corporation

New filename:      2024-01-15_Rechnung_ACME_Corporation.pdf

✓ File successfully renamed!
New location: /path/to/2024-01-15_Rechnung_ACME_Corporation.pdf
============================================================
```

## Configuration

Copy `config.example.yaml` to `config.yaml` and customize:

```yaml
# Vision-Language Model for invoice analysis
vlm_model: Qwen/Qwen2-VL-7B-Instruct

# Optional: Customize cache location (VLMs are large, ~14GB for 7B models)
model_cache_dir: /Volumes/ExternalDrive/models

# VLM inference parameters
vlm_config:
  max_tokens: 200
  temperature: 0.1
```

**Recommended VLMs:**
- `Qwen/Qwen2-VL-7B-Instruct` - Default, good balance (~ 14GB)
- `Qwen/Qwen2-VL-2B-Instruct` - Smaller, faster (~4GB)
- `mlx-community/pixtral-12b-8bit` - Larger, more accurate (~24GB)

## Development

See [CLAUDE.md](CLAUDE.md) for detailed architecture and development information.

### Running Tests

```bash
make test
```

### Code Quality

```bash
make lint    # Run linters
make format  # Format code
```

## How It Works

1. **PDF to Image**: Converts PDF pages to images using PyMuPDF
2. **VLM Analysis**: Vision-Language Model analyzes the image directly
   - Detects if document is an invoice
   - Extracts invoice date (Rechnungsdatum)
   - Extracts invoicing party (company name)
3. **Smart Renaming**: Generates filename in format `YYYY-MM-DD_Rechnung_Company.pdf`
4. **File Management**: Renames file in place or moves to output directory

## Supported Languages

The VLM models support invoices in:
- 🇩🇪 **German** (primary)
- 🇬🇧 **English**
- 🇫🇷 **French**
- 🇪🇸 **Spanish**

## Project Status

**✅ Completed - Invoice Processing:**
- Full VLM integration with MLX acceleration
- Invoice detection using vision-language models
- Date and company extraction
- Intelligent file renaming
- HuggingFace authentication
- Model caching and management
- MLX compatibility and disk space checking
- PDF-only support
- Multilingual invoice support

**🚧 Future Enhancements:**
- Support for additional document categories (receipts, contracts, etc.)
- Batch processing of multiple files
- Additional date formats and parsing improvements

## License

MIT
