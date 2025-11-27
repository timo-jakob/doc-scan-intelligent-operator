"""File renaming utilities."""

import logging
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _validate_safe_path(path: Path, must_exist: bool = True) -> Path:
    """
    Validate that a path is safe to use (no path traversal).
    
    Args:
        path: Path to validate
        must_exist: Whether the path must exist
        
    Returns:
        Resolved absolute path
        
    Raises:
        ValueError: If path is unsafe
        FileNotFoundError: If path doesn't exist and must_exist is True
    """
    # Convert to Path object and resolve to absolute path
    resolved_path = Path(path).resolve()
    
    # Check for null bytes (path traversal attack vector)
    if '\0' in str(path):
        raise ValueError("Path contains null bytes")
    
    # Verify path exists if required
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")
    
    return resolved_path


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
    # Validate and resolve path to prevent path traversal
    try:
        original_path = _validate_safe_path(original_path, must_exist=True)
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Invalid original path: {e}")
        return None
    
    if not original_path.is_file():
        logger.error(f"Path is not a file: {original_path}")
        return None

    # Sanitize filename to prevent path traversal
    # Remove any directory separators from the filename
    new_filename = Path(new_filename).name
    if not new_filename or new_filename in ['.', '..'] or '\0' in new_filename:
        logger.error(f"Invalid filename: {new_filename}")
        return None

    # Determine target directory
    if output_dir:
        try:
            target_dir = _validate_safe_path(output_dir, must_exist=False)
            target_dir.mkdir(parents=True, exist_ok=True)
        except ValueError as e:
            logger.error(f"Invalid output directory: {e}")
            return None
    else:
        target_dir = original_path.parent

    # Create new path
    new_path = (target_dir / new_filename).resolve()
    
    # Ensure the new path is within the target directory (prevent path traversal)
    try:
        new_path.relative_to(target_dir)
    except ValueError:
        logger.error("Invalid path: new file would be outside target directory")
        return None

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
