"""Tests for document scanner."""

import pytest
from pathlib import Path
from docscan.scanner import DocumentScanner


def test_scanner_initialization():
    """Test scanner initialization with config."""
    config = {
        "supported_formats": [".pdf", ".txt"]
    }
    scanner = DocumentScanner(config)
    assert scanner.supported_formats == [".pdf", ".txt"]


def test_is_supported(tmp_path):
    """Test file format support checking."""
    config = {
        "supported_formats": [".pdf", ".txt"]
    }
    scanner = DocumentScanner(config)

    # Create test files
    pdf_file = tmp_path / "test.pdf"
    pdf_file.touch()
    txt_file = tmp_path / "test.txt"
    txt_file.touch()
    doc_file = tmp_path / "test.doc"
    doc_file.touch()

    assert scanner._is_supported(pdf_file) is True
    assert scanner._is_supported(txt_file) is True
    assert scanner._is_supported(doc_file) is False


def test_scan_single_file(tmp_path):
    """Test scanning a single file."""
    config = {
        "supported_formats": [".txt"]
    }
    scanner = DocumentScanner(config)

    # Create test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    documents = scanner.scan(test_file)
    assert len(documents) == 1
    assert documents[0]["name"] == "test.txt"
    assert documents[0]["extension"] == ".txt"


def test_scan_directory(tmp_path):
    """Test scanning a directory."""
    config = {
        "supported_formats": [".txt", ".pdf"]
    }
    scanner = DocumentScanner(config)

    # Create test files
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.pdf").touch()
    (tmp_path / "file3.doc").touch()  # Should be ignored

    documents = scanner.scan(tmp_path)
    assert len(documents) == 2
    names = [doc["name"] for doc in documents]
    assert "file1.txt" in names
    assert "file2.pdf" in names
    assert "file3.doc" not in names
