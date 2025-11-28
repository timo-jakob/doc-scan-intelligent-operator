"""File renaming utilities."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from docscan.path_utils import validate_safe_path

logger = logging.getLogger(__name__)


def _validate_original_path(original_path: Path) -> Optional[Path]:
    """Validate and return the original file path."""
    try:
        validated_path = validate_safe_path(original_path, must_exist=True)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Invalid original path: {e}")
        return None
    
    if not validated_path.is_file():
        logger.error(f"Path is not a file: {validated_path}")
        return None
    
    return validated_path


def _sanitize_filename(new_filename: str) -> Optional[str]:
    """Sanitize filename to prevent path traversal."""
    sanitized = Path(new_filename).name
    if not sanitized or sanitized in ['.', '..'] or '\0' in sanitized:
        logger.error(f"Invalid filename: {new_filename}")
        return None
    return sanitized


def _get_target_directory(output_dir: Optional[Path], original_path: Path) -> Optional[Path]:
    """Determine and validate the target directory."""
    if output_dir:
        try:
            target_dir = validate_safe_path(output_dir, must_exist=False)
            target_dir.mkdir(parents=True, exist_ok=True)
            return target_dir
        except ValueError as e:
            logger.error(f"Invalid output directory: {e}")
            return None
    return original_path.parent


def _resolve_unique_path(target_dir: Path, new_filename: str, original_path: Path) -> Optional[Path]:
    """Resolve a unique path, handling collisions and validating safety."""
    new_path = (target_dir / new_filename).resolve()
    
    # Ensure the new path is within the target directory (prevent path traversal)
    try:
        new_path.relative_to(target_dir)
    except ValueError:
        logger.error("Invalid path: new file would be outside target directory")
        return None

    # Check if target already exists and handle collision
    if new_path.exists() and new_path != original_path:
        logger.warning(f"Target file already exists: {new_path}")
        base = new_path.stem
        ext = new_path.suffix
        counter = 1
        while new_path.exists():
            new_path = target_dir / f"{base}_{counter}{ext}"
            counter += 1
        logger.info(f"Using alternate filename: {new_path.name}")

    return new_path


def _perform_rename(original_path: Path, new_path: Path, target_dir: Path) -> bool:
    """Perform the actual file rename or copy operation."""
    try:
        if target_dir != original_path.parent:
            shutil.copy2(original_path, new_path)
            logger.info(f"Copied and renamed file:\n  From: {original_path}\n  To: {new_path}")
        else:
            original_path.rename(new_path)
            logger.info(f"Renamed file:\n  From: {original_path.name}\n  To: {new_path.name}")
        return True
    except OSError as e:
        logger.error(f"Failed to rename file: {e}")
        return False


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
    # Validate original path
    original_path = _validate_original_path(original_path)
    if not original_path:
        return None

    # Sanitize filename
    new_filename = _sanitize_filename(new_filename)
    if not new_filename:
        return None

    # Get target directory
    target_dir = _get_target_directory(output_dir, original_path)
    if not target_dir:
        return None

    # Resolve unique path
    new_path = _resolve_unique_path(target_dir, new_filename, original_path)
    if not new_path:
        return None

    # Handle dry run
    if dry_run:
        logger.info(f"[DRY RUN] Would rename:\n  From: {original_path}\n  To: {new_path}")
        return None

    # Perform rename
    if _perform_rename(original_path, new_path, target_dir):
        return new_path
    
    return None
