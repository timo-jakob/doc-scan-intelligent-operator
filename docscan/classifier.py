"""Document classification using MLX and Hugging Face models."""

from typing import List, Dict, Any
from pathlib import Path
import logging
import re

from docscan.model_manager import ModelManager

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """Classifies documents using MLX-optimized models."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document classifier.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model_id = config.get("model")
        self.categories = config.get("categories", [])
        self.mlx_config = config.get("mlx_config", {})

        # Initialize model manager with configured cache directory
        cache_dir = config.get("model_cache_dir")
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser()
        self.model_manager = ModelManager(cache_dir)

        self.model = None
        self.tokenizer = None

        self._load_model()

    def _load_model(self):
        """Load model and tokenizer using MLX."""
        if not self.model_id:
            raise ValueError("No model ID specified in configuration")

        logger.info(f"Loading model: {self.model_id}")

        try:
            self.model, self.tokenizer = self.model_manager.load_model(self.model_id)
            logger.info(f"Model {self.model_id} loaded successfully")
        except (ImportError, RuntimeError, OSError, ValueError) as e:
            # Catches model loading errors, MLX errors, file errors
            logger.error(f"Failed to load model: {e}")
            raise

    def classify(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single document.

        Args:
            document: Document dictionary with path and metadata

        Returns:
            Document dictionary with added classification results
        """
        # Extract text from document
        text = self._extract_text(document)

        # Generate classification prompt
        prompt = self._create_classification_prompt(text)

        # Run inference
        category = self._infer_category(prompt)

        # Add classification to document
        document["text"] = text
        document["category"] = category

        return document

    def classify_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple documents.

        Args:
            documents: List of document dictionaries

        Returns:
            List of classified documents
        """
        return [self.classify(doc) for doc in documents]

    def _extract_text(self, document: Dict[str, Any]) -> str:
        """
        Extract text from document based on file type.

        Args:
            document: Document dictionary

        Returns:
            Extracted text content
        """
        file_path = document["path"]
        extension = document["extension"].lower()

        # TODO: Implement text extraction for different formats
        # - PDF: pypdf or pdfplumber
        # - Images: OCR using pytesseract or similar
        # - Text files: direct reading
        # - DOCX: python-docx

        if extension == ".txt":
            return file_path.read_text()
        else:
            # Placeholder - implement actual extraction
            return f"[Text extraction for {extension} not yet implemented]"

    def _create_classification_prompt(self, text: str) -> str:
        """
        Create prompt for classification.

        Args:
            text: Document text

        Returns:
            Formatted prompt
        """
        categories_str = ", ".join(self.categories)

        prompt = f"""Classify the following document into one of these categories: {categories_str}

Document text:
{text[:1000]}...

Category:"""

        return prompt

    def _infer_category(self, prompt: str) -> str:
        """
        Run model inference to determine category.

        Args:
            prompt: Classification prompt

        Returns:
            Predicted category
        """
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded")
            return "other"

        try:
            from mlx_lm import generate

            # Get generation parameters from config
            max_tokens = self.mlx_config.get("max_tokens", 50)
            temperature = self.mlx_config.get("temperature", 0.1)

            logger.debug(f"Running inference with max_tokens={max_tokens}, temp={temperature}")

            # Generate response
            response = generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                verbose=False,
            )

            # Extract category from response
            category = self._parse_category_from_response(response)
            logger.debug(f"Classified as: {category}")

            return category

        except (RuntimeError, ValueError, TypeError, KeyError) as e:
            # Catches model inference errors, tokenization errors, and data access errors
            logger.error(f"Inference failed: {e}")
            return "other"

    def _parse_category_from_response(self, response: str) -> str:
        """
        Parse category from model response.

        Args:
            response: Model's text response

        Returns:
            Extracted category name
        """
        # Clean up response
        response = response.strip().lower()

        # Try to find exact category match
        for category in self.categories:
            if category.lower() in response:
                return category

        # Try to extract first word as category
        first_word = response.split()[0] if response.split() else ""
        first_word = re.sub(r'[^\w\s]', '', first_word)  # Remove punctuation

        # Check if first word matches a category
        for category in self.categories:
            if first_word == category.lower():
                return category

        # Default to "other" if no match
        logger.warning(f"Could not parse category from response: {response}")
        return self.categories[-1] if self.categories else "other"
