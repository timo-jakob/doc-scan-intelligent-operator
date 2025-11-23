"""Document scanning and ingestion."""

from pathlib import Path
from typing import List, Dict, Any


class DocumentScanner:
    """Scans directories and files for documents to process."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document scanner.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.supported_formats = config.get("supported_formats", [])

    def scan(self, input_path: Path) -> List[Dict[str, Any]]:
        """
        Scan path for documents.

        Args:
            input_path: Path to file or directory

        Returns:
            List of document dictionaries with metadata
        """
        documents = []

        if input_path.is_file():
            if self._is_supported(input_path):
                documents.append(self._create_document_info(input_path))
        elif input_path.is_dir():
            for file_path in input_path.rglob("*"):
                if file_path.is_file() and self._is_supported(file_path):
                    documents.append(self._create_document_info(file_path))
        else:
            raise ValueError(f"Invalid input path: {input_path}")

        return documents

    def _is_supported(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in self.supported_formats

    def _create_document_info(self, file_path: Path) -> Dict[str, Any]:
        """Create document information dictionary."""
        return {
            "path": file_path,
            "name": file_path.name,
            "extension": file_path.suffix,
            "size": file_path.stat().st_size,
        }
