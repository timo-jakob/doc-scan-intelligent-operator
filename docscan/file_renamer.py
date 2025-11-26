"""File renaming utilities."""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def rename_invoice(
    original_path: Path,
    new_filename: str,
    output_dir: Optional[Path] = None,
    dry_run: bool = False
) -> Optional[Path]:
    """
    Rename invoice file with new name.

    Args:
        original_path: Original file path
        new_filename: New filename (without directory)
        output_dir: Optional output directory. If None, rename in same directory
        dry_run: If True, don't actually rename

    Returns:
        New file path if successful, None if failed or dry_run
    """
    if not original_path.exists():
        logger.error(f"File does not exist: {original_path}")
        return None

    # Determine target directory
    if output_dir:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = original_path.parent

    # Create new path
    new_path = target_dir / new_filename

    # Check if target already exists
    if new_path.exists() and new_path != original_path:
        logger.warning(f"Target file already exists: {new_path}")
        # Add number suffix to avoid collision
        base = new_path.stem
        ext = new_path.suffix
        counter = 1
        while new_path.exists():
            new_path = target_dir / f"{base}_{counter}{ext}"
            counter += 1
        logger.info(f"Using alternate filename: {new_path.name}")

    if dry_run:
        logger.info(f"[DRY RUN] Would rename:\n  From: {original_path}\n  To: {new_path}")
        return None

    try:
        # If moving to different directory, copy instead of rename
        if target_dir != original_path.parent:
            shutil.copy2(original_path, new_path)
            logger.info(f"Copied and renamed file:\n  From: {original_path}\n  To: {new_path}")
        else:
            original_path.rename(new_path)
            logger.info(f"Renamed file:\n  From: {original_path.name}\n  To: {new_path.name}")

        return new_path

    except Exception as e:
        logger.error(f"Failed to rename file: {e}")
        return None
