.PHONY: help venv install install-dev test lint format clean clean-venv run

VENV := venv
PYTHON := python3.12
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

help:
	@echo "Available commands:"
	@echo "  make venv         - Create Python 3.12 virtual environment"
	@echo "  make install      - Install package and dependencies (creates venv if needed)"
	@echo "  make install-dev  - Install package with development dependencies (creates venv if needed)"
	@echo "  make test         - Run tests with coverage"
	@echo "  make lint         - Run linters (ruff, mypy)"
	@echo "  make format       - Format code with black"
	@echo "  make clean        - Remove build artifacts and cache files"
	@echo "  make clean-venv   - Remove virtual environment"
	@echo "  make run          - Run the CLI (use ARGS='...' for arguments)"

venv:
	@echo "Creating Python 3.12 virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created at ./$(VENV)"; \
	else \
		echo "Virtual environment already exists at ./$(VENV)"; \
	fi
	@echo "Upgrading pip..."
	@$(VENV_PIP) install --upgrade pip

install: venv
	@echo "Installing production dependencies..."
	@$(VENV_PIP) install -r requirements.txt
	@$(VENV_PIP) install -e .
	@echo ""
	@echo "✅ Installation complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"

install-dev: venv
	@echo "Installing development dependencies..."
	@$(VENV_PIP) install -r requirements-dev.txt
	@$(VENV_PIP) install -e .
	@echo ""
	@echo "✅ Development installation complete!"
	@echo "To activate the virtual environment, run:"
	@echo "  source $(VENV)/bin/activate"

test:
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/pytest; \
	else \
		pytest; \
	fi

lint:
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/ruff check docscan tests; \
		$(VENV)/bin/mypy docscan; \
	else \
		ruff check docscan tests; \
		mypy docscan; \
	fi

format:
	@if [ -d "$(VENV)" ]; then \
		$(VENV)/bin/black docscan tests; \
		$(VENV)/bin/ruff check --fix docscan tests; \
	else \
		black docscan tests; \
		ruff check --fix docscan tests; \
	fi

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

clean-venv:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "Virtual environment removed."

run:
	@if [ -d "$(VENV)" ]; then \
		$(VENV_PYTHON) -m docscan.cli $(ARGS); \
	else \
		python -m docscan.cli $(ARGS); \
	fi
