"""Command-line interface for document scanning and organization."""

import argparse
import sys
import logging
from pathlib import Path

from docscan.config import load_config
from docscan.scanner import DocumentScanner
from docscan.classifier import DocumentClassifier
from docscan.organizer import DocumentOrganizer


def setup_logging(verbose: bool = False):
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Scan and organize documents intelligently using AI"
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to document or directory to scan"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Output directory for organized documents"
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "-m", "--model",
        help="Model identifier from Hugging Face"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview classification without moving files"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Load configuration
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)

    # Override config with CLI arguments
    if args.model:
        config["model"] = args.model
    if args.output:
        config["output_dir"] = args.output

    # Initialize components
    scanner = DocumentScanner(config)
    classifier = DocumentClassifier(config)
    organizer = DocumentOrganizer(config)

    try:
        # Scan documents
        documents = scanner.scan(args.input_path)
        print(f"Found {len(documents)} documents to process")

        # Classify documents
        classified = classifier.classify_batch(documents)

        # Organize documents
        if args.dry_run:
            print("\nDry run - no files will be moved:")
            for doc in classified:
                print(f"  {doc['path']} -> {doc['category']}")
        else:
            organizer.organize(classified)
            print(f"\nSuccessfully organized {len(classified)} documents")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
