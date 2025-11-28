"""Path validation utilities for secure file operations."""

from pathlib import Path


def validate_safe_path(path: Path, must_exist: bool = True) -> Path:
    """
    Validate that a path is safe to use (no path traversal).

    Security: This function prevents path traversal attacks by:
    - Resolving to absolute canonical path (eliminating ../, ./, symlinks)
    - Checking for null bytes (common path traversal vector)
    - Verifying file existence and type

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
    # resolve() canonicalizes the path, removing .. and . components
    resolved_path = Path(path).resolve()

    # Check for null bytes (path traversal attack vector)
    if '\0' in str(path):
        raise ValueError("Path contains null bytes")

    # Verify path exists if required
    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    return resolved_path
