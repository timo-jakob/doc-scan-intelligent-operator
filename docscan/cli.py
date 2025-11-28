"""Command-line interface for invoice detection and renaming."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

from docscan.config import load_config, DEFAULT_VLM_MODEL, DEFAULT_TEXT_LLM_MODEL
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


def _validate_input_file(pdf_file: Path, logger) -> None:
    """Validate that the input file exists and is a valid PDF."""
    if not pdf_file.exists():
        logger.error(f"File not found: {pdf_file}")
        sys.exit(1)

    if not is_valid_pdf(pdf_file):
        logger.error(f"Invalid PDF file: {pdf_file}")
        logger.error("This application only supports PDF files.")
        sys.exit(1)


def _prepare_config(args, logger):
    """Load and prepare configuration from file and CLI arguments."""
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Set mode based on flags
    config["use_text_llm"] = args.text_llm
    config["compare_mode"] = args.compare

    # Override models from CLI args if provided
    if args.model:
        config["vlm_model"] = args.model
    if args.text_model:
        config["text_llm_model"] = args.text_model

    return config


def _compare_results(
    vlm_result: Tuple[bool, Optional[Dict]],
    text_result: Tuple[bool, Optional[Dict]]
) -> Tuple[bool, list]:
    """
    Compare results from VLM and text LLM modes.
    
    Returns:
        Tuple of (are_same, list_of_differences)
    """
    vlm_is_invoice, vlm_data = vlm_result
    text_is_invoice, text_data = text_result
    
    differences = []
    
    # Compare detection results
    if vlm_is_invoice != text_is_invoice:
        differences.append({
            "field": "Invoice Detection",
            "vlm": "YES" if vlm_is_invoice else "NO",
            "text_llm": "YES" if text_is_invoice else "NO"
        })
    
    # If both detected as invoice, compare extracted data
    if vlm_is_invoice and text_is_invoice and vlm_data and text_data:
        if vlm_data.get("date") != text_data.get("date"):
            differences.append({
                "field": "Date",
                "vlm": vlm_data.get("date", "N/A"),
                "text_llm": text_data.get("date", "N/A")
            })
        
        if vlm_data.get("invoicing_party") != text_data.get("invoicing_party"):
            differences.append({
                "field": "Invoicing Party",
                "vlm": vlm_data.get("invoicing_party", "N/A"),
                "text_llm": text_data.get("invoicing_party", "N/A")
            })
    
    return len(differences) == 0, differences


def _display_comparison(
    vlm_result: Tuple[bool, Optional[Dict]],
    text_result: Tuple[bool, Optional[Dict]],
    are_same: bool,
    differences: list
) -> None:
    """Display comparison results between VLM and text LLM modes."""
    vlm_is_invoice, vlm_data = vlm_result
    text_is_invoice, text_data = text_result
    
    print("\n" + "="*70)
    print("COMPARISON: VLM vs Text LLM")
    print("="*70)
    
    # Display VLM results
    print("\n[Option 1] VLM Mode (Qwen2-VL)")
    print("-"*40)
    if vlm_is_invoice and vlm_data:
        print("  Invoice Detected: YES")
        print(f"  Date:             {vlm_data.get('date', 'N/A')}")
        print(f"  Invoicing Party:  {vlm_data.get('invoicing_party', 'N/A')}")
        print(f"  Filename:         {generate_invoice_filename(vlm_data)}")
    else:
        print("  Invoice Detected: NO")
    
    # Display Text LLM results
    print("\n[Option 2] Text LLM Mode (OCR + Llama)")
    print("-"*40)
    if text_is_invoice and text_data:
        print("  Invoice Detected: YES")
        print(f"  Date:             {text_data.get('date', 'N/A')}")
        print(f"  Invoicing Party:  {text_data.get('invoicing_party', 'N/A')}")
        print(f"  Filename:         {generate_invoice_filename(text_data)}")
    else:
        print("  Invoice Detected: NO")
    
    print("\n" + "-"*70)
    
    if are_same:
        print("✓ RESULTS MATCH - Both methods produced the same result")
    else:
        print("✗ RESULTS DIFFER - The following differences were found:")
        print()
        for diff in differences:
            print(f"  {diff['field']}:")
            print(f"    VLM:      {diff['vlm']}")
            print(f"    Text LLM: {diff['text_llm']}")
    
    print("="*70)


def _prompt_user_choice() -> int:
    """Prompt user to choose between VLM (1) or Text LLM (2) results."""
    while True:
        print("\nWhich result would you like to use?")
        print("  [1] VLM Mode result")
        print("  [2] Text LLM Mode result")
        print("  [0] Cancel (don't rename)")
        
        try:
            choice = input("\nEnter your choice (0/1/2): ").strip()
            if choice in ["0", "1", "2"]:
                return int(choice)
            print("Invalid choice. Please enter 0, 1, or 2.")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0


def _run_compare_mode(args, config, model_manager, logger):
    """Run both VLM and text LLM modes and compare results."""
    # Run both analysis modes
    vlm_result = _run_vlm_analysis(args, config, model_manager, logger)
    text_result = _run_text_llm_analysis(args, config, model_manager, logger)
    
    # Compare and display results
    are_same, differences = _compare_results(vlm_result, text_result)
    _display_comparison(vlm_result, text_result, are_same, differences)
    
    # Get the result to use
    chosen_result = _get_chosen_result(vlm_result, text_result, are_same)
    if chosen_result is None:
        return
    
    is_invoice, invoice_data = chosen_result
    _apply_chosen_result(is_invoice, invoice_data, args)


def _run_vlm_analysis(args, config, model_manager, logger):
    """Run VLM-based invoice analysis."""
    logger.info(f"Loading VLM: {config['vlm_model']}")
    vlm_model, vlm_processor = model_manager.load_model(config["vlm_model"], is_vlm=True)
    vlm_detector = InvoiceDetector(vlm_model, vlm_processor, config, use_text_llm=False)
    
    logger.info("Analyzing with VLM...")
    return vlm_detector.analyze_document(args.pdf_file)


def _run_text_llm_analysis(args, config, model_manager, logger):
    """Run text LLM-based invoice analysis."""
    logger.info(f"Loading Text LLM: {config['text_llm_model']}")
    text_model, text_tokenizer = model_manager.load_model(config["text_llm_model"], is_vlm=False)
    text_detector = InvoiceDetector(text_model, text_tokenizer, config, use_text_llm=True)
    
    logger.info("Analyzing with Text LLM...")
    return text_detector.analyze_document(args.pdf_file)


def _get_chosen_result(vlm_result, text_result, are_same):
    """Determine which result to use based on comparison."""
    if are_same:
        print("\nUsing the matching result.")
        return vlm_result
    
    choice = _prompt_user_choice()
    
    if choice == 0:
        print("\nOperation cancelled. No changes made.")
        return None
    elif choice == 1:
        print("\nUsing VLM result.")
        return vlm_result
    else:
        print("\nUsing Text LLM result.")
        return text_result


def _apply_chosen_result(is_invoice, invoice_data, args):
    """Apply the chosen invoice result (rename file or report)."""
    if not is_invoice or not invoice_data:
        print("\nDocument is not an invoice. No action taken.")
        return
    
    new_filename = generate_invoice_filename(invoice_data)
    print(f"\nNew filename: {new_filename}")
    
    if args.dry_run:
        print("[DRY RUN] File would be renamed (use without --dry-run to actually rename)")
        return
    
    new_path = rename_invoice(
        args.pdf_file,
        new_filename,
        args.output_dir,
        dry_run=False
    )
    if new_path:
        print(f"✓ File successfully renamed to: {new_path}")
    else:
        print("✗ Failed to rename file")
        sys.exit(1)


def _process_invoice_result(is_invoice: bool, invoice_data: dict, args) -> None:
    """Process and display the invoice detection results."""
    print("\n" + "="*60)

    if is_invoice:
        print("✓ INVOICE DETECTED")
        print("="*60)
        detection_method = invoice_data.get('detection_method', 'Unknown')
        extraction_method = invoice_data.get('extraction_method', 'Unknown')
        print(f"Detection Method:  {detection_method}")
        print(f"Extraction Method: {extraction_method}")
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
                print("\n✓ File successfully renamed!")
                print(f"New location: {new_path}")
            else:
                print("\n✗ Failed to rename file")
                sys.exit(1)
    else:
        print("✗ NOT AN INVOICE")
        print("="*60)
        print("This document does not appear to be an invoice.")
        print("The application only processes invoices at this time.")
        print("\nNo action taken.")

    print("="*60 + "\n")


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

  # Use text-only LLM mode (lighter weight, OCR + text LLM)
  docscan invoice.pdf --text-llm

  # Use text-only mode with custom model
  docscan invoice.pdf --text-llm --text-model mlx-community/Mistral-7B-Instruct-v0.3-4bit

  # Compare both modes and choose the best result
  docscan invoice.pdf --compare

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
        default=None,
        help=f"VLM model identifier from Hugging Face (default: {DEFAULT_VLM_MODEL})"
    )

    parser.add_argument(
        "--text-llm",
        action="store_true",
        help="Use text-only LLM with OCR instead of VLM (lighter weight, uses --text-model)"
    )

    parser.add_argument(
        "--text-model",
        default=None,
        help=f"Text LLM model for --text-llm mode (default: {DEFAULT_TEXT_LLM_MODEL})"
    )

    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run both VLM and text LLM modes, compare results, and let user choose if different"
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
    _validate_input_file(args.pdf_file, logger)

    # Load configuration
    config = _prepare_config(args, logger)

    try:
        # Initialize model manager
        cache_dir = config.get("model_cache_dir")
        if cache_dir:
            cache_dir = Path(cache_dir).expanduser()
        model_manager = ModelManager(cache_dir)

        # Handle compare mode separately
        if config.get("compare_mode"):
            _run_compare_mode(args, config, model_manager, logger)
            return

        # Load model based on mode
        if config.get("use_text_llm"):
            # Text-only LLM mode
            logger.info(f"Loading text LLM: {config['text_llm_model']}")
            logger.info("Using OCR + text LLM mode (lighter weight)")
            logger.info("This may take a while on first run (downloading model)...")

            model, tokenizer = model_manager.load_model(config["text_llm_model"], is_vlm=False)
            logger.info("Model loaded successfully!")

            # Initialize invoice detector in text-only mode
            detector = InvoiceDetector(model, tokenizer, config, use_text_llm=True)
        else:
            # VLM mode (default)
            logger.info(f"Loading vision-language model: {config['vlm_model']}")
            logger.info("This may take a while on first run (downloading model)...")

            model, processor = model_manager.load_model(config["vlm_model"], is_vlm=True)
            logger.info("Model loaded successfully!")

            # Initialize invoice detector in VLM mode
            detector = InvoiceDetector(model, processor, config, use_text_llm=False)

        # Analyze document
        logger.info(f"\nAnalyzing document: {args.pdf_file.name}")
        is_invoice, invoice_data = detector.analyze_document(args.pdf_file)

        # Report results
        _process_invoice_result(is_invoice, invoice_data, args)

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
