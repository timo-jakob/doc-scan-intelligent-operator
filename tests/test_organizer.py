"""Tests for document organizer."""

import pytest
from pathlib import Path
from docscan.organizer import DocumentOrganizer


def test_organizer_initialization():
    """Test organizer initialization."""
    config = {
        "output_dir": "/test/output",
        "organize_by": "category"
    }
    organizer = DocumentOrganizer(config)
    assert organizer.output_dir == Path("/test/output")
    assert organizer.organize_by == "category"


def test_get_destination_path_by_category(tmp_path):
    """Test destination path generation by category."""
    config = {
        "output_dir": str(tmp_path / "output"),
        "organize_by": "category"
    }
    organizer = DocumentOrganizer(config)

    document = {
        "name": "invoice.pdf",
        "category": "invoice"
    }

    dest = organizer._get_destination_path(document)
    assert dest == tmp_path / "output" / "invoice" / "invoice.pdf"


def test_organize_documents(tmp_path):
    """Test organizing documents."""
    # Create source files
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    file1 = source_dir / "invoice.pdf"
    file1.write_text("invoice content")
    file2 = source_dir / "receipt.pdf"
    file2.write_text("receipt content")

    # Configure organizer
    config = {
        "output_dir": str(tmp_path / "output"),
        "organize_by": "category"
    }
    organizer = DocumentOrganizer(config)

    # Organize documents
    documents = [
        {"path": file1, "name": "invoice.pdf", "category": "invoice"},
        {"path": file2, "name": "receipt.pdf", "category": "receipt"},
    ]

    organizer.organize(documents)

    # Check results
    assert (tmp_path / "output" / "invoice" / "invoice.pdf").exists()
    assert (tmp_path / "output" / "receipt" / "receipt.pdf").exists()
