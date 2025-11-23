"""Document organization and file management."""

from pathlib import Path
from typing import List, Dict, Any
import shutil


class DocumentOrganizer:
    """Organizes classified documents into directory structure."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize document organizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config.get("output_dir", "./organized_documents"))
        self.organize_by = config.get("organize_by", "category")

    def organize(self, documents: List[Dict[str, Any]]) -> None:
        """
        Organize documents into output directory structure.

        Args:
            documents: List of classified documents
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for doc in documents:
            destination = self._get_destination_path(doc)
            destination.parent.mkdir(parents=True, exist_ok=True)

            # Copy file to destination
            shutil.copy2(doc["path"], destination)

    def _get_destination_path(self, document: Dict[str, Any]) -> Path:
        """
        Determine destination path for document.

        Args:
            document: Classified document

        Returns:
            Destination path
        """
        category = document.get("category", "other")
        filename = document["name"]

        if self.organize_by == "category":
            return self.output_dir / category / filename
        elif self.organize_by == "date":
            # TODO: Extract date from document
            date_str = "unknown_date"
            return self.output_dir / date_str / filename
        elif self.organize_by == "category/date":
            # TODO: Extract date from document
            date_str = "unknown_date"
            return self.output_dir / category / date_str / filename
        else:
            return self.output_dir / filename
