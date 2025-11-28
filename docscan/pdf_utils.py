"""PDF processing utilities."""

import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)

# PDF rendering constants
DEFAULT_PDF_DPI = 150  # Good balance between quality and performance
PDF_BASE_DPI = 72  # PDF standard DPI


def pdf_to_images(pdf_path: Path, dpi: int = DEFAULT_PDF_DPI) -> List[Image.Image]:
    """
    Convert PDF pages to PIL Images.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default 150 for good quality/speed balance)

    Returns:
        List of PIL Images, one per page

    Raises:
        ImportError: If required packages not installed
        ValueError: If PDF cannot be read
    """
    try:
        import fitz  # PyMuPDF

        logger.debug(f"Converting PDF to images: {pdf_path}")

        # Open PDF
        doc = fitz.open(pdf_path)
        images = []

        # Convert each page to image
        for page_num in range(len(doc)):
            page = doc[page_num]

            # Render page to pixmap with specified DPI
            # Calculate zoom factor: PDF_BASE_DPI is default, so zoom = target_dpi / PDF_BASE_DPI
            zoom = dpi / PDF_BASE_DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

            logger.debug(f"Converted page {page_num + 1}/{len(doc)}")

        doc.close()
        logger.info(f"Converted {len(images)} pages from {pdf_path.name}")

        return images

    except ImportError as e:
        raise ImportError(
            "PyMuPDF (fitz) is required for PDF processing. "
            "Install with: pip install PyMuPDF"
        ) from e
    except (OSError, RuntimeError, ValueError) as e:
        # Catches fitz.FileDataError (subclass of RuntimeError), file errors, and value errors
        raise ValueError(f"Failed to convert PDF to images: {e}") from e


def get_pdf_page_count(pdf_path: Path) -> int:
    """
    Get number of pages in PDF.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Number of pages
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    except (OSError, RuntimeError, ValueError) as e:
        # Catches fitz.FileDataError (subclass of RuntimeError), file errors, and value errors
        logger.error(f"Failed to get page count: {e}")
        return 0


def is_valid_pdf(pdf_path: Path) -> bool:
    """
    Check if file is a valid PDF.

    Args:
        pdf_path: Path to check

    Returns:
        True if valid PDF, False otherwise
    """
    if not pdf_path.exists() or not pdf_path.is_file():
        return False

    if pdf_path.suffix.lower() != '.pdf':
        return False

    try:
        import fitz
        doc = fitz.open(pdf_path)
        doc.close()
        return True
    except Exception:
        return False
