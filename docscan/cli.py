"""Command-line interface for invoice detection and renaming."""

import argparse
import sys
import logging
from pathlib import Path

from docscan.config import load_config
from docscan.model_manager import ModelManager
from docscan.invoice_detector import InvoiceDetector, generate_invoice_filename
from docscan.file_renamer import rename_invoice
from docscan.pdf_utils import is_valid_pdf


def setup_logging(verbose: bool = False):
    """
    Configure logging for the application.

    Args:
        verbose: Enable debug logging
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )


def main():
    """Main CLI entry point for invoice processing."""
    parser = argparse.ArgumentParser(
        description="Analyze PDF invoices and rename them intelligently using AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze and rename an invoice
  docscan invoice.pdf

  # Dry run (preview without renaming)
  docscan invoice.pdf --dry-run

  # Use custom configuration
  docscan invoice.pdf -c config.yaml

  # Use different VLM model
  docscan invoice.pdf -m Qwen/Qwen2-VL-7B-Instruct

  # Verbose output for debugging
  docscan invoice.pdf -v
        """
    )

    parser.add_argument(
        "pdf_file",
        type=Path,
        help="Path to PDF invoice file to analyze"
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to configuration file"
    )

    parser.add_argument(
        "-m", "--model",
        help="VLM model identifier from Hugging Face (e.g., Qwen/Qwen2-VL-7B-Instruct)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        help="Output directory for renamed file (default: same directory as input)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview detection and filename without actually renaming"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output for debugging"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Validate input file
    if not args.pdf_file.exists():
        logger.error(f"File not found: {args.pdf_file}")
        sys.exit(1)

    if not is_valid_pdf(args.pdf_file):
        logger.error(f"Invalid PDF file: {args.pdf_file}")
        logger.error("This application only supports PDF files.")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Override config with CLI arguments
    if args.model:
        config["vlm_model"] = args.model

    # Ensure VLM model is configured
    if "vlm_model" not in config or not config["vlm_model"]:
        logger.error("No VLM model configured. Please specify a model with -m or in config.yaml")
        logger.error("Example: -m Qwen/Qwen2-VL-7B-Instruct")
        sys.exit(1)

    try:
        # Initialize model manager
        cache_dir = config.get("model_cache_dir")
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser()
        model_manager = ModelManager(cache_dir)

        # Load VLM
        logger.info(f"Loading vision-language model: {config['vlm_model']}")
        logger.info("This may take a while on first run (downloading model)...")

        model, processor = model_manager.load_model(config["vlm_model"], is_vlm=True)

        logger.info("Model loaded successfully!")

        # Initialize invoice detector
        detector = InvoiceDetector(model, processor, config)

        # Analyze document
        logger.info(f"\nAnalyzing document: {args.pdf_file.name}")
        is_invoice, invoice_data = detector.analyze_document(args.pdf_file)

        # Report results
        print("\n" + "="*60)

        if is_invoice:
            print("✓ INVOICE DETECTED")
            print("="*60)
            print(f"Invoice Date:      {invoice_data['date']}")
            print(f"Invoicing Party:   {invoice_data['invoicing_party']}")

            # Generate new filename
            new_filename = generate_invoice_filename(invoice_data)
            print(f"\nNew filename:      {new_filename}")

            # Rename file
            if args.dry_run:
                print("\n[DRY RUN] File would be renamed (use without --dry-run to actually rename)")
            else:
                new_path = rename_invoice(
                    args.pdf_file,
                    new_filename,
                    args.output_dir,
                    dry_run=False
                )

                if new_path:
                    print(f"\n✓ File successfully renamed!")
                    print(f"New location: {new_path}")
                else:
                    print(f"\n✗ Failed to rename file")
                    sys.exit(1)
        else:
            print("✗ NOT AN INVOICE")
            print("="*60)
            print("This document does not appear to be an invoice.")
            print("The application only processes invoices at this time.")
            print("\nNo action taken.")

        print("="*60 + "\n")

    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
