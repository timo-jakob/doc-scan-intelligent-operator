"""HuggingFace authentication and credential management."""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_huggingface_token() -> Optional[str]:
    """
    Get HuggingFace token from environment variables or config files.

    Priority order:
    1. Environment variable HF_TOKEN
    2. Environment variable HUGGING_FACE_TOKEN
    3. ~/.huggingface/token file
    4. ~/.config/huggingface/token file

    Returns:
        Token string if found, None otherwise
    """
    # Check environment variables first (highest priority)
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.info("Using HuggingFace token from HF_TOKEN environment variable")
        return token.strip()

    token = os.environ.get("HUGGING_FACE_TOKEN")
    if token:
        logger.info("Using HuggingFace token from HUGGING_FACE_TOKEN environment variable")
        return token.strip()

    # Check standard HuggingFace token locations
    home = Path.home()
    token_paths = [
        home / ".huggingface" / "token",
        home / ".config" / "huggingface" / "token",
    ]

    for token_path in token_paths:
        if token_path.exists():
            try:
                token = token_path.read_text().strip()
                if token:
                    logger.info(f"Using HuggingFace token from {token_path}")
                    return token
            except (OSError, UnicodeDecodeError) as e:
                logger.warning(f"Failed to read token from {token_path}: {e}")

    logger.debug("No HuggingFace token found")
    return None


def login_to_huggingface(token: Optional[str] = None) -> bool:
    """
    Login to HuggingFace using token.

    Args:
        token: Optional token to use. If None, will try to find token automatically.

    Returns:
        True if login successful, False otherwise
    """
    if token is None:
        token = get_huggingface_token()

    if token is None:
        logger.warning(
            "No HuggingFace token found. Some models may not be accessible.\n"
            "To authenticate, set HF_TOKEN environment variable or create "
            "~/.huggingface/token file."
        )
        return False

    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        logger.info("Successfully logged in to HuggingFace")
        return True
    except ImportError:
        logger.error("huggingface_hub package not installed")
        return False
    except Exception as e:
        logger.error(f"Failed to login to HuggingFace: {e}")
        return False
