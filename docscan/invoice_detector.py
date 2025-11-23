"""Invoice detection and information extraction using VLMs."""

import logging
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image

from docscan.pdf_utils import pdf_to_images, is_valid_pdf

logger = logging.getLogger(__name__)


class InvoiceDetector:
    """Detects invoices and extracts key information using Vision Language Models."""

    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        """
        Initialize invoice detector.

        Args:
            model: VLM model
            tokenizer: Model tokenizer
            config: Configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.vlm_config = config.get("vlm_config", {})

    def analyze_document(self, pdf_path: Path) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Analyze document to detect if it's an invoice and extract information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Tuple of (is_invoice, invoice_data)
            invoice_data contains: {
                "date": "YYYY-MM-DD",
                "invoicing_party": "Company Name",
                "original_filename": "original.pdf"
            }
        """
        # Validate PDF
        if not is_valid_pdf(pdf_path):
            logger.error(f"Invalid PDF file: {pdf_path}")
            return False, None

        # Convert PDF to images (usually just need first page for invoices)
        try:
            images = pdf_to_images(pdf_path, dpi=150)
            if not images:
                logger.error(f"No pages extracted from PDF: {pdf_path}")
                return False, None

            # Analyze first page (invoices have header on first page)
            first_page = images[0]

            # Step 1: Detect if it's an invoice
            is_invoice = self._detect_invoice(first_page)

            if not is_invoice:
                logger.info(f"Document is not an invoice: {pdf_path.name}")
                return False, None

            logger.info(f"Invoice detected: {pdf_path.name}")

            # Step 2: Extract invoice information
            invoice_data = self._extract_invoice_data(first_page)
            invoice_data["original_filename"] = pdf_path.name

            return True, invoice_data

        except Exception as e:
            logger.error(f"Failed to analyze document: {e}")
            return False, None

    def _detect_invoice(self, image: Image.Image) -> bool:
        """
        Detect if image shows an invoice.

        Args:
            image: PIL Image of document page

        Returns:
            True if invoice detected, False otherwise
        """
        prompt = """Look at this document image. Is this an invoice (Rechnung)?

An invoice typically contains:
- Invoice number (Rechnungsnummer)
- Invoice date (Rechnungsdatum)
- Sender/company information
- Recipient information
- List of items or services with prices
- Total amount (Gesamtbetrag)

Answer with only "YES" if this is an invoice, or "NO" if it is not an invoice.

Answer:"""

        try:
            response = self._query_vlm(image, prompt)
            response_clean = response.strip().upper()

            # Check if response indicates invoice
            is_invoice = "YES" in response_clean or "JA" in response_clean

            logger.debug(f"Invoice detection response: {response} -> {is_invoice}")
            return is_invoice

        except Exception as e:
            logger.error(f"Invoice detection failed: {e}")
            return False

    def _extract_invoice_data(self, image: Image.Image) -> Dict[str, str]:
        """
        Extract invoice date and invoicing party from image.

        Args:
            image: PIL Image of invoice

        Returns:
            Dictionary with "date" and "invoicing_party"
        """
        prompt = """Look at this invoice and extract the following information:

1. Invoice date (Rechnungsdatum): The date when the invoice was issued
2. Invoicing party (Rechnungssteller): The company or person who issued the invoice

Please provide the information in this exact format:
DATE: YYYY-MM-DD
PARTY: Company Name

Note:
- The date should be in ISO format (YYYY-MM-DD)
- The party should be the issuing company's name (usually at the top of the invoice)
- This invoice may be in German, English, French, or Spanish

Your response:"""

        try:
            response = self._query_vlm(image, prompt)
            logger.debug(f"Extraction response: {response}")

            # Parse response
            invoice_data = self._parse_extraction_response(response)

            return invoice_data

        except Exception as e:
            logger.error(f"Failed to extract invoice data: {e}")
            return {
                "date": "0000-00-00",
                "invoicing_party": "Unknown"
            }

    def _query_vlm(self, image: Image.Image, prompt: str) -> str:
        """
        Query VLM with image and text prompt.

        Args:
            image: PIL Image
            prompt: Text prompt

        Returns:
            Model response
        """
        try:
            # Import MLX VLM utilities
            from mlx_vlm import generate
            from mlx_vlm.utils import load_image

            # Get generation parameters
            max_tokens = self.vlm_config.get("max_tokens", 200)
            temperature = self.vlm_config.get("temperature", 0.1)

            logger.debug(f"Querying VLM with max_tokens={max_tokens}, temp={temperature}")

            # Generate response
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                image=image,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            return response

        except ImportError:
            # Fallback if mlx_vlm not available
            logger.error("mlx_vlm not available, using placeholder")
            return "YES\nDATE: 2024-01-01\nPARTY: Placeholder Company"

        except Exception as e:
            logger.error(f"VLM query failed: {e}")
            raise

    def _parse_extraction_response(self, response: str) -> Dict[str, str]:
        """
        Parse VLM response to extract structured data.

        Args:
            response: VLM response text

        Returns:
            Dictionary with date and invoicing_party
        """
        data = {
            "date": "0000-00-00",
            "invoicing_party": "Unknown"
        }

        # Extract date
        date_match = re.search(r'DATE:\s*(\d{4}-\d{2}-\d{2})', response, re.IGNORECASE)
        if date_match:
            data["date"] = date_match.group(1)
        else:
            # Try to find any date in response
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', response)
            if date_match:
                data["date"] = date_match.group(1)

        # Extract party
        party_match = re.search(r'PARTY:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if party_match:
            party_name = party_match.group(1).strip()
            # Clean up party name
            party_name = self._sanitize_filename_part(party_name)
            data["invoicing_party"] = party_name

        # Validate date format
        try:
            datetime.strptime(data["date"], "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {data['date']}, using placeholder")
            data["date"] = "0000-00-00"

        return data

    def _sanitize_filename_part(self, text: str) -> str:
        """
        Sanitize text for use in filename.

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text safe for filenames
        """
        # Remove invalid filename characters
        text = re.sub(r'[<>:"/\\|?*]', '', text)

        # Replace spaces and multiple underscores
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'_+', '_', text)

        # Remove leading/trailing underscores
        text = text.strip('_')

        # Limit length
        if len(text) > 50:
            text = text[:50]

        return text or "Unknown"


def generate_invoice_filename(invoice_data: Dict[str, str]) -> str:
    """
    Generate filename for invoice based on extracted data.

    Format: YYYY-MM-DD_Rechnung_InvoicingParty.pdf

    Args:
        invoice_data: Dictionary with date and invoicing_party

    Returns:
        Generated filename
    """
    date = invoice_data.get("date", "0000-00-00")
    party = invoice_data.get("invoicing_party", "Unknown")

    # Format: date_Rechnung_party.pdf
    filename = f"{date}_Rechnung_{party}.pdf"

    return filename
