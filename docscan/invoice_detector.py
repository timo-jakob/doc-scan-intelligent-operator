"""Invoice detection and information extraction using VLMs."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
from PIL import Image

from docscan.pdf_utils import pdf_to_images, is_valid_pdf

logger = logging.getLogger(__name__)

# Constants for extraction method tracking
METHOD_VLM = "VLM"
METHOD_TEXT_LLM = "Text LLM"
METHOD_OCR = "OCR (pytesseract)"
METHOD_OCR_TEXT_LLM = "OCR + Text LLM"


class InvoiceDetector:
    """Detects invoices and extracts key information using Vision Language Models or Text LLMs."""

    def __init__(self, model, processor, config: Dict[str, Any], use_text_llm: bool = False):
        """
        Initialize invoice detector.

        Args:
            model: VLM or text LLM model
            processor: Model processor (image processor for VLMs, tokenizer for text LLMs)
            config: Configuration dictionary
            use_text_llm: If True, use OCR + text LLM instead of VLM
        """
        self.model = model
        self.processor = processor
        self.config = config
        self.vlm_config = config.get("vlm_config", {})
        self.use_text_llm = use_text_llm

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

            if self.use_text_llm:
                # Text LLM mode: use OCR for both detection and extraction
                is_invoice, invoice_data = self._analyze_with_text_llm(first_page)
                if invoice_data:
                    invoice_data["original_filename"] = pdf_path.name
            else:
                # VLM mode (default): use VLM for detection, OCR for extraction
                is_invoice = self._detect_invoice(first_page)

                if not is_invoice:
                    logger.info(f"Document is not an invoice: {pdf_path.name}")
                    return False, None

                logger.info(f"Invoice detected: {pdf_path.name}")

                # Extract invoice information using OCR
                invoice_data = self._extract_invoice_data(first_page)
                invoice_data["original_filename"] = pdf_path.name
                invoice_data["detection_method"] = METHOD_VLM

            if not is_invoice:
                logger.info(f"Document is not an invoice: {pdf_path.name}")
                return False, None

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
        prompt = """What type of document is this? Is it an invoice, bill, Rechnung, or Liquidation?

Answer only with INVOICE if this is an invoice/bill/Rechnung/Liquidation, or OTHER if it is not."""

        # Retry up to 3 times for empty responses
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._query_vlm(image, prompt)
                response_clean = response.strip().upper()

                # If response is empty or too short, retry
                if len(response_clean) < 2 and attempt < max_retries - 1:
                    logger.debug(f"Empty response on attempt {attempt + 1}, retrying...")
                    continue

                # Check if response indicates invoice
                is_invoice = "INVOICE" in response_clean or "RECHNUNG" in response_clean or "LIQUIDATION" in response_clean

                logger.debug(f"Invoice detection response: {response} -> {is_invoice}")
                return is_invoice

            except Exception as e:
                logger.error(f"Invoice detection failed: {e}")
                if attempt < max_retries - 1:
                    continue
                return False

        return False

    def _extract_invoice_data(self, image: Image.Image) -> Dict[str, str]:
        """
        Extract invoice date and invoicing party from image using OCR.

        Args:
            image: PIL Image of invoice

        Returns:
            Dictionary with "date", "invoicing_party", and "extraction_method"
        """
        try:
            # Use OCR for reliable text extraction
            import pytesseract
            
            # Extract text from image
            ocr_text = pytesseract.image_to_string(image)
            logger.debug(f"OCR extracted text (first 500 chars): {ocr_text[:500]}")
            
            # Parse the OCR text to extract invoice data
            invoice_data = self._parse_ocr_text(ocr_text)
            invoice_data["extraction_method"] = METHOD_OCR
            
            return invoice_data

        except ImportError:
            logger.warning("pytesseract not available, falling back to VLM extraction")
            data = self._extract_invoice_data_vlm(image)
            data["extraction_method"] = METHOD_VLM
            return data
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}, falling back to VLM")
            data = self._extract_invoice_data_vlm(image)
            data["extraction_method"] = METHOD_VLM
            return data
            return data

    def _parse_ocr_text(self, text: str) -> Dict[str, str]:
        """
        Parse OCR text to extract invoice date and issuing party.
        
        Args:
            text: OCR extracted text
            
        Returns:
            Dictionary with date and invoicing_party
        """
        data = {
            "date": "0000-00-00",
            "invoicing_party": "Unknown",
            "invoicing_party_filename": "Unknown"
        }
        
        # Extract date
        data["date"] = self._extract_date_from_text(text)
        
        # Extract company name (returns tuple of display, filename versions)
        display_name, filename_safe = self._extract_company_from_text(text)
        data["invoicing_party"] = display_name
        data["invoicing_party_filename"] = filename_safe
        
        return data

    def _extract_date_from_text(self, text: str) -> str:
        """Extract invoice date from OCR text."""
        date_patterns = [
            # Handle various OCR artifacts in separators (—_: or similar)
            r'Rechnungsdatum\s*[:\-—_\s]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'Invoice\s*Date\s*[:\-—_\s]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'Datum\s*[:\-—_\s]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
            r'Date\s*[:\-—_\s]+(\d{1,2}[./]\d{1,2}[./]\d{2,4})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._convert_date_to_iso(match.group(1))
        
        return "0000-00-00"

    def _extract_company_from_text(self, text: str) -> Tuple[str, str]:
        """Extract issuing company name from OCR text.
        
        Returns:
            Tuple of (display_name, filename_safe_name)
        """
        lines = text.split('\n')
        
        # First, look for specific practice/company patterns
        result = self._find_practice_name(lines)
        if result:
            return result
        
        # Then look for legal entity suffixes
        result = self._find_legal_entity(lines)
        if result:
            return result
        
        # Fallback: check header area
        return self._find_header_company(lines)

    def _find_practice_name(self, lines: list) -> Optional[Tuple[str, str]]:
        """Find practice names like 'Chiropraktik White' or 'Praxis Schmidt'.
        
        Returns:
            Tuple of (display_name, filename_safe_name) or None
        """
        patterns = [
            r'(Chiropraktik\s+\w+)',
            r'(Praxis\s+[\w\s]+)',
        ]
        
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    company = match.group(1).strip()
                    # Fixed: Use negated character class to prevent ReDoS vulnerability
                    company = re.sub(r'\s+(Kontakt|Bank|Telefon|Fax|IBAN)[^\n]*$', '', company, flags=re.IGNORECASE)
                    if len(company) >= 3:
                        return (self._clean_display_name(company), 
                                self._sanitize_filename_part(company))
        return None

    def _find_legal_entity(self, lines: list) -> Optional[Tuple[str, str]]:
        """Find company names with legal entity suffixes (GmbH, AG, etc.).
        
        Returns:
            Tuple of (display_name, filename_safe_name) or None
        """
        patterns = [
            r'^([\w\s]+\s+GmbH)\b',
            r'^([\w\s]+\s+AG)\b',
            r'^([\w\s]+\s+e\.?K\.?)\b',
            r'^([\w\s]+\s+(?:OHG|KG|UG))\b',
        ]
        
        for line in lines:
            for pattern in patterns:
                match = re.search(pattern, line.strip(), re.IGNORECASE)
                if match:
                    company = match.group(1).strip()
                    if len(company) >= 3:
                        return (self._clean_display_name(company),
                                self._sanitize_filename_part(company))
        return None

    def _find_header_company(self, lines: list) -> Tuple[str, str]:
        """Fallback: find company name in header area.
        
        Returns:
            Tuple of (display_name, filename_safe_name)
        """
        for line in lines[:10]:
            if self._is_potential_company_line(line):
                company = line.strip()[:50]
                return (self._clean_display_name(company),
                        self._sanitize_filename_part(company))
        return ("Unknown", "Unknown")

    def _analyze_with_text_llm(self, image: Image.Image) -> Tuple[bool, Optional[Dict[str, str]]]:
        """
        Analyze document using OCR + text LLM.

        Args:
            image: PIL Image of document page

        Returns:
            Tuple of (is_invoice, invoice_data)
        """
        try:
            import pytesseract

            # Extract text using OCR
            ocr_text = pytesseract.image_to_string(image)
            logger.debug(f"OCR extracted text (first 500 chars): {ocr_text[:500]}")

            if not ocr_text.strip():
                logger.warning("OCR extracted no text from image")
                return False, None

            # Use text LLM to analyze the OCR text
            is_invoice, llm_data = self._query_text_llm_for_invoice(ocr_text)

            if not is_invoice:
                return False, None

            # If LLM extraction failed, fall back to regex parsing
            if llm_data.get("date") == "0000-00-00" or llm_data.get("invoicing_party") == "Unknown":
                logger.debug("LLM extraction incomplete, supplementing with regex")
                regex_data = self._parse_ocr_text(ocr_text)
                if llm_data.get("date") == "0000-00-00":
                    llm_data["date"] = regex_data["date"]
                if llm_data.get("invoicing_party") == "Unknown":
                    llm_data["invoicing_party"] = regex_data["invoicing_party"]
                    llm_data["invoicing_party_filename"] = regex_data.get("invoicing_party_filename", "Unknown")

            llm_data["detection_method"] = METHOD_TEXT_LLM
            llm_data["extraction_method"] = METHOD_OCR_TEXT_LLM

            return True, llm_data

        except ImportError:
            logger.error("pytesseract not available for text LLM mode")
            return False, None
        except Exception as e:
            logger.error(f"Text LLM analysis failed: {e}")
            return False, None

    def _query_text_llm_for_invoice(self, ocr_text: str) -> Tuple[bool, Dict[str, str]]:
        """
        Query text LLM to detect invoice and extract data from OCR text.

        Args:
            ocr_text: Text extracted via OCR

        Returns:
            Tuple of (is_invoice, extracted_data)
        """
        # Build chat messages - will be formatted using tokenizer's apply_chat_template
        system_message = (
            "You are a document classifier specialized in German invoices. "
            "Extract information accurately and respond ONLY in the exact format requested."
        )
        
        user_message = (
            "Analyze this invoice text and extract:\n"
            "1. Is it an invoice/Rechnung/Liquidation? (YES/NO)\n"
            "2. Invoice date (Rechnungsdatum) in YYYY-MM-DD format\n"
            "3. The BUSINESS/PRACTICE NAME that issued this invoice\n\n"
            "IMPORTANT for finding the issuer:\n"
            "- Look for the practice/business name, NOT personal names or credentials\n"
            "- Common patterns: 'Praxis [Name]', 'Chiropraktik [Name]', '[Name] GmbH'\n"
            "- The issuer is usually in the letterhead or return address, NOT the recipient\n"
            "- Ignore academic titles (Dr., D.C.) and university names\n\n"
            f"Text:\n\"\"\"\n{ocr_text[:3000]}\n\"\"\"\n\n"
            "Answer ONLY in this format:\n"
            "INVOICE: YES or NO\n"
            "DATE: YYYY-MM-DD\n"
            "ISSUER: business/practice name only"
        )

        try:
            response = self._query_text_llm(system_message, user_message)
            logger.debug(f"Text LLM response: {response}")

            # Parse response
            is_invoice = "INVOICE: YES" in response.upper() or "INVOICE:YES" in response.upper()

            data = {
                "date": "0000-00-00",
                "invoicing_party": "Unknown",
                "invoicing_party_filename": "Unknown"
            }

            # Extract date
            date_match = re.search(r'DATE:\s*(\d{4}-\d{2}-\d{2})', response, re.IGNORECASE)
            if date_match:
                data["date"] = date_match.group(1)

            # Extract issuer
            issuer_match = re.search(r'ISSUER:\s*([^\n]+)', response, re.IGNORECASE)
            if issuer_match:
                issuer = issuer_match.group(1).strip()
                if issuer.upper() != "UNKNOWN":
                    data["invoicing_party"] = self._clean_display_name(issuer)
                    data["invoicing_party_filename"] = self._sanitize_filename_part(issuer)

            return is_invoice, data

        except Exception as e:
            logger.error(f"Text LLM query failed: {e}")
            return False, {"date": "0000-00-00", "invoicing_party": "Unknown", "invoicing_party_filename": "Unknown"}

    def _query_text_llm(self, system_message: str, user_message: str) -> str:
        """
        Query text-only LLM with a chat-style prompt.
        
        Uses the tokenizer's apply_chat_template() for model-agnostic prompting,
        which automatically formats the prompt correctly for any instruct model
        (Llama, Mistral, Qwen, etc.).

        Args:
            system_message: System instruction for the model
            user_message: User query/task

        Returns:
            Model response
        """
        try:
            from mlx_lm import generate
            from mlx_lm.sample_utils import make_sampler

            max_tokens = self.vlm_config.get("max_tokens", 200)
            temperature = self.vlm_config.get("temperature", 0.1)

            tokenizer = self.processor  # In text mode, processor is the tokenizer
            prompt = self._build_chat_prompt(tokenizer, system_message, user_message)

            logger.debug(f"Querying text LLM with max_tokens={max_tokens}, temp={temperature}")

            # Create sampler with temperature (mlx-lm 0.28+ uses sampler instead of temp)
            sampler = make_sampler(temp=temperature)

            response = generate(
                model=self.model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                sampler=sampler,
                verbose=False,
            )

            return response

        except Exception as e:
            logger.error(f"Text LLM query failed: {e}")
            raise

    def _build_chat_prompt(self, tokenizer, system_message: str, user_message: str) -> str:
        """
        Build a chat prompt using the tokenizer's template, with fallbacks.
        
        Args:
            tokenizer: The model's tokenizer
            system_message: System instruction
            user_message: User query
            
        Returns:
            Formatted prompt string
        """
        if not hasattr(tokenizer, 'apply_chat_template'):
            # Fallback for tokenizers without chat template support
            logger.warning("Tokenizer doesn't support apply_chat_template, using plain format")
            return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"

        # Try with system message first (Llama, Qwen style)
        messages_with_system = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
        
        try:
            return tokenizer.apply_chat_template(
                messages_with_system,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.debug(f"Chat template with system message failed: {e}")
        
        # Fallback: embed system message in user message (Mistral style)
        combined_user = f"{system_message}\n\n{user_message}"
        messages_no_system = [
            {"role": "user", "content": combined_user}
        ]
        
        try:
            return tokenizer.apply_chat_template(
                messages_no_system,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using plain format")
            return f"System: {system_message}\n\nUser: {user_message}\n\nAssistant:"

    def _is_potential_company_line(self, line: str) -> bool:
        """Check if line could be a company name."""
        line = line.strip()
        if len(line) <= 5:
            return False
        if re.match(r'^[\d\s\-/]+$', line):
            return False
        return bool(re.search(r'[A-Za-zäöüÄÖÜß]{3,}', line))

    def _convert_date_to_iso(self, date_str: str) -> str:
        """
        Convert various date formats to ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            ISO formatted date string
        """
        # Try common formats
        formats = [
            "%d.%m.%Y",  # German: 14.11.2025
            "%d/%m/%Y",  # 14/11/2025
            "%m/%d/%Y",  # US: 11/14/2025
            "%d.%m.%y",  # Short year: 14.11.25
            "%d/%m/%y",
            "%Y-%m-%d",  # Already ISO
        ]
        
        for fmt in formats:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return "0000-00-00"

    def _extract_invoice_data_vlm(self, image: Image.Image) -> Dict[str, str]:
        """
        Fallback: Extract invoice data using VLM (less reliable).

        Args:
            image: PIL Image of invoice

        Returns:
            Dictionary with "date" and "invoicing_party"
        """
        prompt = """Look carefully at this invoice image and extract:

1. The invoice date - look for "Datum", "Date", "Rechnungsdatum" or similar
2. The company/person who ISSUED this invoice (the sender, not recipient) - this is often in the header or footer, look for company name, address, contact details

IMPORTANT: Read the ACTUAL text from the image. Do not make up or guess information.
Look in the footer area as company details are often placed there.

Respond in this exact format:
DATE: YYYY-MM-DD
PARTY: [exact company name as written on the invoice]"""

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
                "invoicing_party": "Unknown",
                "invoicing_party_filename": "Unknown"
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

            # Get generation parameters
            max_tokens = self.vlm_config.get("max_tokens", 200)
            temperature = self.vlm_config.get("temperature", 0.1)

            logger.debug(f"Querying VLM with max_tokens={max_tokens}, temp={temperature}")

            # Save image to temp file - mlx_vlm expects a path, not PIL Image
            import os
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"docscan_{os.getpid()}.png")
            image.save(temp_path)
            logger.debug(f"Saved temp image to {temp_path}")

            try:
                # Generate response - mlx_vlm 0.3.x returns a GenerationResult object
                result = generate(
                    model=self.model,
                    processor=self.processor,
                    image=[temp_path],  # Pass as list of paths
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=False,
                )
                logger.debug(f"VLM raw result: {result}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

            # Extract text from GenerationResult
            return result.text if hasattr(result, 'text') else str(result)

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
            "invoicing_party": "Unknown",
            "invoicing_party_filename": "Unknown"
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
        party_match = re.search(r'PARTY:\s*([^\n]+)', response, re.IGNORECASE)
        if party_match:
            party_name = party_match.group(1).strip()
            # Store both display and filename versions
            data["invoicing_party"] = self._clean_display_name(party_name)
            data["invoicing_party_filename"] = self._sanitize_filename_part(party_name)

        # Validate date format
        try:
            datetime.strptime(data["date"], "%Y-%m-%d")
        except ValueError:
            logger.warning(f"Invalid date format: {data['date']}, using placeholder")
            data["date"] = "0000-00-00"

        return data

    def _clean_display_name(self, text: str) -> str:
        """
        Clean text for display purposes (keeps spaces).

        Args:
            text: Text to clean

        Returns:
            Cleaned text suitable for display
        """
        # Remove invalid filename characters but keep spaces
        text = re.sub(r'[<>:"/\\|?*]', '', text)

        # Normalize multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Limit length
        if len(text) > 50:
            text = text[:50]

        return text or "Unknown"

    def _sanitize_filename_part(self, text: str) -> str:
        """
        Sanitize text for use in filename (replaces spaces with underscores).

        Args:
            text: Text to sanitize

        Returns:
            Sanitized text safe for filenames
        """
        # First clean for display
        text = self._clean_display_name(text)

        # Replace spaces with underscores for filename safety
        text = re.sub(r'\s+', '_', text)
        text = re.sub(r'_+', '_', text)

        # Remove leading/trailing underscores
        text = text.strip('_')

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
    # Use filename-safe version if available, otherwise sanitize the display version
    party = invoice_data.get("invoicing_party_filename")
    if not party:
        # Fallback: sanitize the display name
        display_party = invoice_data.get("invoicing_party", "Unknown")
        party = re.sub(r'\s+', '_', display_party)

    # Format: date_Rechnung_party.pdf
    filename = f"{date}_Rechnung_{party}.pdf"

    return filename
