# VLM-Based Invoice Processing System

This PR introduces a complete rewrite of the document scanning system, now focused on intelligent invoice processing using Vision-Language Models (VLMs) with MLX acceleration for Apple Silicon.

## 🎯 Overview

The system now uses state-of-the-art Vision-Language Models to analyze PDF invoices directly (no OCR required), extract key information, and rename files intelligently according to a standardized format.

## ✨ Key Features

### Invoice Processing
- **VLM-Based Detection**: Uses Qwen2-VL or similar models to detect invoices from PDF images
- **Smart Extraction**: Automatically extracts:
  - Invoice date (YYYY-MM-DD format)
  - Invoicing party (company name)
- **Intelligent Renaming**: Files renamed as `YYYY-MM-DD_Rechnung_Company.pdf`
- **Multilingual Support**: Handles German, English, French, and Spanish invoices

### Technical Implementation
- **MLX Optimization**: Leverages Apple Silicon's Neural Engine for fast inference
- **PDF Processing**: Direct PDF to image conversion using PyMuPDF (no OCR needed)
- **Model Management**: Smart caching, disk space checking, MLX compatibility validation
- **Secure Authentication**: HuggingFace token support via environment variables or config files

### User Experience
- **Simple CLI**: `docscan invoice.pdf` - that's it!
- **Clear Feedback**: Visual indicators (✓/✗) for invoice detection
- **Dry Run Mode**: Preview changes without modifying files
- **Error Handling**: Comprehensive validation and helpful error messages

## 📋 Changes by Commit

### 1. Core Implementation (adaa45f)
**New Modules:**
- `docscan/invoice_detector.py` - VLM-based invoice analysis
- `docscan/pdf_utils.py` - PDF to image conversion
- `docscan/file_renamer.py` - Safe file renaming
- `docscan/model_manager.py` - Enhanced with VLM support
- `docscan/mlx_utils.py` - VLM compatibility checking
- `docscan/auth.py` - HuggingFace authentication
- `docscan/cli.py` - Redesigned CLI

**Infrastructure:**
- Complete test suite (6 test modules)
- Package configuration (setup.py, pyproject.toml)
- Development tools (Makefile, pytest.ini)

### 2. Renovate Configuration (0d28d7d)
**Fixed Dependencies:**
- Production: mlx==0.21.0, mlx-vlm==0.1.1, PyMuPDF==1.24.14, etc.
- Development: pytest==8.3.4, black==24.10.0, ruff==0.8.4, etc.

**Renovate Setup:**
- Automated dependency updates via Pull Requests
- Grouped updates (production, development, MLX packages)
- Security vulnerability alerts
- GitHub Actions workflow for dependency testing

**GitHub Integration:**
- CODEOWNERS for automatic reviewer assignment
- PR template with dependency checklist
- Comprehensive setup guide

### 3. Documentation (ffa4b0c)
**Updated Files:**
- README.md - Complete rewrite for invoice processing
- CLAUDE.md - Detailed architecture and development guide
- config.example.yaml - VLM configuration examples

## 🚀 Usage Examples

### Basic Invoice Processing
```bash
# Analyze and rename an invoice
docscan invoice.pdf

# Output:
# ✓ INVOICE DETECTED
# Invoice Date:      2024-01-15
# Invoicing Party:   ACME_Corporation
# New filename:      2024-01-15_Rechnung_ACME_Corporation.pdf
# ✓ File successfully renamed!
```

### Dry Run
```bash
docscan invoice.pdf --dry-run
# Preview changes without modifying files
```

### Custom VLM Model
```bash
docscan invoice.pdf -m Qwen/Qwen2-VL-2B-Instruct
# Use smaller, faster model
```

## 🔧 Configuration

### Default VLM Model
`Qwen/Qwen2-VL-7B-Instruct` (~14GB)

### Alternative Models
- `Qwen/Qwen2-VL-2B-Instruct` - Smaller, faster (~4GB)
- `mlx-community/pixtral-12b-8bit` - Larger, more accurate (~24GB)

### Model Cache
Default: `~/.cache/docscan/models`
Configurable via `model_cache_dir` in config.yaml

## 📊 Testing

### Test Coverage
- ✅ HuggingFace authentication (env vars, token files)
- ✅ MLX compatibility checking (VLMs and text models)
- ✅ Disk space validation
- ✅ Model caching and loading
- ✅ Configuration management
- ✅ Document scanning
- ✅ File organization

### GitHub Actions
Automated testing workflow that:
- Tests on macOS (Apple Silicon)
- Verifies MLX installation
- Runs security audits
- Checks code quality

## 🔐 Security & Stability

### HuggingFace Authentication
Credentials loaded from (in priority order):
1. `HF_TOKEN` environment variable
2. `HUGGING_FACE_TOKEN` environment variable
3. `~/.huggingface/token` file
4. `~/.config/huggingface/token` file

### Fixed Dependencies
All dependencies use exact versions (`==`) for:
- Predictable behavior
- Renovate-managed updates via PRs
- Production stability

### Renovate Features
- Scheduled updates (Mondays before 10am)
- Grouped PRs by type
- 7-day delay for major updates
- Security patches prioritized
- MLX packages handled separately

## 📝 Migration Notes

### Breaking Changes
This is a complete rewrite focusing on invoice processing:
- Old multi-format document processing removed
- Now PDF-only
- Single-file processing (batch processing not yet implemented)
- Focus on invoices (other document types not yet supported)

### What's Removed
- Generic document classification
- Multi-format support (was: PDF, images, DOCX, etc.)
- Directory scanning and organization
- Text-based LLM classification

### What's Added
- Vision-Language Model integration
- Direct PDF image analysis
- Invoice-specific extraction
- Intelligent file renaming
- Multilingual support

## ✅ Testing Checklist

- [x] All commits are properly formatted
- [x] Tests pass locally (where implemented)
- [x] Code follows project style (black, ruff)
- [x] Documentation updated
- [x] Dependencies pinned to exact versions
- [x] Renovate configuration validated

## 🔗 Related Documentation

- [CLAUDE.md](./CLAUDE.md) - Complete architecture guide
- [README.md](./README.md) - User-facing documentation
- [.github/RENOVATE_SETUP.md](./.github/RENOVATE_SETUP.md) - Renovate usage guide
- [config.example.yaml](./config.example.yaml) - Configuration examples

## 🙏 Review Focus

Please pay special attention to:
1. **MLX dependencies** - Critical for Apple Silicon functionality
2. **Renovate configuration** - Ensures proper update management
3. **VLM prompts** - In `invoice_detector.py`, may need tuning for accuracy
4. **File naming logic** - Ensure safe handling of edge cases
5. **Authentication flow** - Verify secure credential handling

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
