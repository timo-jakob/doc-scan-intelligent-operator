"""Tests for command-line interface."""

import sys
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

from docscan.cli import setup_logging, main


def test_setup_logging_default():
    """Test logging setup with default (non-verbose) mode."""
    with patch('logging.basicConfig') as mock_config:
        setup_logging(verbose=False)

    mock_config.assert_called_once()
    call_kwargs = mock_config.call_args.kwargs
    assert call_kwargs['level'] == logging.INFO


def test_setup_logging_verbose():
    """Test logging setup with verbose mode."""
    with patch('logging.basicConfig') as mock_config:
        setup_logging(verbose=True)

    mock_config.assert_called_once()
    call_kwargs = mock_config.call_args.kwargs
    assert call_kwargs['level'] == logging.DEBUG


def test_setup_logging_format():
    """Test logging format configuration."""
    with patch('logging.basicConfig') as mock_config:
        setup_logging()

    call_kwargs = mock_config.call_args.kwargs
    assert 'format' in call_kwargs
    assert '%(levelname)s' in call_kwargs['format']
    assert '%(message)s' in call_kwargs['format']


@pytest.fixture
def mock_args(tmp_path):
    """Create mock CLI arguments."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    args = MagicMock()
    args.pdf_file = pdf_file
    args.config = None
    args.model = None
    args.output_dir = None
    args.dry_run = False
    args.verbose = False

    return args


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return {
        "vlm_model": "test-model",
        "model_cache_dir": None,
    }


def test_main_file_not_found():
    """Test main with non-existent file."""
    with patch('sys.argv', ['docscan', 'nonexistent.pdf']):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.setup_logging'):
                main()

    # Verify exit was called with code 1
    assert mock_exit.called
    assert any(call[0][0] == 1 for call in mock_exit.call_args_list)


def test_main_invalid_pdf(tmp_path):
    """Test main with invalid PDF file."""
    pdf_file = tmp_path / "invalid.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.is_valid_pdf', return_value=False):
                with patch('docscan.cli.setup_logging'):
                    main()

    # Verify exit was called with code 1
    assert mock_exit.called
    assert any(call[0][0] == 1 for call in mock_exit.call_args_list)


def test_main_no_model_configured(tmp_path):
    """Test main without VLM model configured."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={}):
                    with patch('docscan.cli.setup_logging'):
                        main()

    # Verify exit was called with code 1
    assert mock_exit.called
    assert any(call[0][0] == 1 for call in mock_exit.call_args_list)


def test_main_model_from_cli_arg(tmp_path):
    """Test main with model specified via CLI argument."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '-m', 'custom-model']):
        with patch('sys.exit'):
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={}):
                    with patch('docscan.cli.setup_logging'):
                        with patch('docscan.cli.ModelManager') as MockModelManager:
                            mock_manager = MockModelManager.return_value
                            mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                            with patch('docscan.cli.InvoiceDetector') as MockDetector:
                                mock_detector = MockDetector.return_value
                                mock_detector.analyze_document.return_value = (True, invoice_data)

                                with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                    with patch('builtins.print'):
                                        main()

    # Verify model was loaded with CLI argument
    mock_manager.load_model.assert_called_once_with('custom-model', is_vlm=True)


def test_main_invoice_detected_success(tmp_path):
    """Test main with successful invoice detection and renaming."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    new_filename = "2024-01-15_Rechnung_ACME_Corp.pdf"
    new_path = tmp_path / new_filename

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=new_path) as mock_rename:
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify rename was called
    mock_rename.assert_called_once()
    assert mock_rename.call_args.args[0] == pdf_file
    assert mock_rename.call_args.args[1] == new_filename

    # Verify success message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("✓ INVOICE DETECTED" in str(call) for call in print_calls)
    assert any("successfully renamed" in str(call) for call in print_calls)


def test_main_invoice_detected_dry_run(tmp_path):
    """Test main with dry run mode."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--dry-run']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice') as mock_rename:
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify rename was NOT called
    mock_rename.assert_not_called()

    # Verify dry run message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("DRY RUN" in str(call) for call in print_calls)


def test_main_not_invoice(tmp_path):
    """Test main when document is not an invoice."""
    pdf_file = tmp_path / "document.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (False, None)

                            with patch('docscan.cli.rename_invoice') as mock_rename:
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify rename was NOT called
    mock_rename.assert_not_called()

    # Verify not invoice message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("✗ NOT AN INVOICE" in str(call) for call in print_calls)


def test_main_rename_failure(tmp_path):
    """Test main when file rename fails."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                    with patch('docscan.cli.setup_logging'):
                        with patch('docscan.cli.ModelManager') as MockModelManager:
                            mock_manager = MockModelManager.return_value
                            mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                            with patch('docscan.cli.InvoiceDetector') as MockDetector:
                                mock_detector = MockDetector.return_value
                                mock_detector.analyze_document.return_value = (True, invoice_data)

                                with patch('docscan.cli.rename_invoice', return_value=None):
                                    with patch('builtins.print'):
                                        main()

    mock_exit.assert_called_with(1)


def test_main_with_output_dir(tmp_path):
    """Test main with custom output directory."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    new_path = output_dir / "2024-01-15_Rechnung_ACME_Corp.pdf"

    with patch('sys.argv', ['docscan', str(pdf_file), '-o', str(output_dir)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=new_path) as mock_rename:
                                with patch('builtins.print'):
                                    main()

    # Verify output_dir was passed to rename_invoice
    assert mock_rename.call_args.args[2] == output_dir


def test_main_keyboard_interrupt(tmp_path):
    """Test main with keyboard interrupt."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                    with patch('docscan.cli.setup_logging'):
                        with patch('docscan.cli.ModelManager') as MockModelManager:
                            MockModelManager.side_effect = KeyboardInterrupt()

                            main()

    mock_exit.assert_called_with(130)


def test_main_general_exception(tmp_path):
    """Test main with general exception."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('sys.exit') as mock_exit:
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                    with patch('docscan.cli.setup_logging'):
                        with patch('docscan.cli.ModelManager', side_effect=Exception("Test error")):
                            main()

    mock_exit.assert_called_with(1)


def test_main_verbose_exception_traceback(tmp_path):
    """Test main prints traceback in verbose mode."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file), '-v']):
        with patch('sys.exit'):
            with patch('docscan.cli.is_valid_pdf', return_value=True):
                with patch('docscan.cli.load_config', return_value={"vlm_model": "test-model"}):
                    with patch('docscan.cli.setup_logging'):
                        with patch('docscan.cli.ModelManager', side_effect=Exception("Test error")):
                            with patch('traceback.print_exc') as mock_traceback:
                                main()

    mock_traceback.assert_called_once()


def test_main_with_custom_config(tmp_path):
    """Test main with custom config file."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()
    config_file = tmp_path / "config.yaml"

    custom_config = {
        "vlm_model": "custom-vlm-model",
    }

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '-c', str(config_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value=custom_config) as mock_load_config:
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                with patch('builtins.print'):
                                    main()

    # Verify custom config was loaded
    mock_load_config.assert_called_once_with(config_file)


def test_main_cache_dir_expansion(tmp_path):
    """Test main expands cache directory path."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    config = {
        "vlm_model": "test-model",
        "model_cache_dir": "~/models",
    }

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value=config):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                with patch('builtins.print'):
                                    main()

    # Verify ModelManager was initialized with expanded path
    MockModelManager.assert_called_once()
    call_args = MockModelManager.call_args[0]
    assert "~" not in str(call_args[0])


def test_main_text_llm_mode(tmp_path):
    """Test main with text LLM mode."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--text-llm']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"text_llm_model": "test-text-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                with patch('builtins.print'):
                                    main()

    # Verify text LLM model was loaded
    mock_manager.load_model.assert_called_once_with("test-text-model", is_vlm=False)

    # Verify InvoiceDetector was initialized in text LLM mode
    MockDetector.assert_called_once()
    assert MockDetector.call_args.kwargs['use_text_llm'] is True


def test_main_text_llm_mode_with_custom_model(tmp_path):
    """Test main with text LLM mode and custom model."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--text-llm', '--text-model', 'custom-text-model']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={"text_llm_model": "default-model"}):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                with patch('builtins.print'):
                                    main()

    # Verify custom text model was loaded (overriding config)
    mock_manager.load_model.assert_called_once_with("custom-text-model", is_vlm=False)


def test_main_config_file_not_found(tmp_path):
    """Test main with non-existent config file."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()
    config_file = tmp_path / "nonexistent_config.yaml"

    with patch('sys.argv', ['docscan', str(pdf_file), '-c', str(config_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', side_effect=FileNotFoundError("Config not found")):
                with patch('docscan.cli.setup_logging'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

    # Verify exit was called with code 1
    assert exc_info.value.code == 1


def test_main_invalid_config(tmp_path):
    """Test main with invalid config file."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()
    config_file = tmp_path / "invalid_config.yaml"

    with patch('sys.argv', ['docscan', str(pdf_file), '-c', str(config_file)]):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', side_effect=ValueError("Invalid config")):
                with patch('docscan.cli.setup_logging'):
                    with pytest.raises(SystemExit) as exc_info:
                        main()

    # Verify exit was called with code 1
    assert exc_info.value.code == 1


def test_main_compare_mode_matching_results(tmp_path):
    """Test compare mode when both VLM and text LLM produce the same result."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            # Both modes return the same result
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify both models were loaded
    assert mock_manager.load_model.call_count == 2

    # Verify "RESULTS MATCH" message was displayed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("RESULTS MATCH" in str(call) for call in print_calls)


def test_main_compare_mode_different_detection(tmp_path):
    """Test compare mode when VLM and text LLM disagree on invoice detection."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    vlm_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            # VLM says yes, text LLM says no
                            mock_detector.analyze_document.side_effect = [
                                (True, vlm_data),  # VLM result
                                (False, None),     # Text LLM result
                            ]

                            # Simulate user choosing option 1 (VLM)
                            with patch('builtins.input', return_value='1'):
                                with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                    with patch('builtins.print') as mock_print:
                                        main()

    # Verify "RESULTS DIFFER" message was displayed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("RESULTS DIFFER" in str(call) for call in print_calls)


def test_main_compare_mode_different_data(tmp_path):
    """Test compare mode when both detect invoice but extract different data."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    vlm_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    text_data = {
        "date": "2024-01-16",  # Different date
        "invoicing_party": "Beta_Inc",  # Different party
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.side_effect = [
                                (True, vlm_data),   # VLM result
                                (True, text_data),  # Text LLM result
                            ]

                            # Simulate user choosing option 2 (Text LLM)
                            with patch('builtins.input', return_value='2'):
                                with patch('docscan.cli.rename_invoice', return_value=pdf_file):
                                    with patch('builtins.print') as mock_print:
                                        main()

    # Verify differences were shown
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("Date" in str(call) or "Invoicing Party" in str(call) for call in print_calls)


def test_main_compare_mode_user_cancels(tmp_path):
    """Test compare mode when user chooses to cancel."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    vlm_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    text_data = {
        "date": "2024-01-16",
        "invoicing_party": "Beta_Inc",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.side_effect = [
                                (True, vlm_data),
                                (True, text_data),
                            ]

                            # Simulate user choosing option 0 (Cancel)
                            with patch('builtins.input', return_value='0'):
                                with patch('docscan.cli.rename_invoice') as mock_rename:
                                    with patch('builtins.print') as mock_print:
                                        main()

    # Verify rename was NOT called
    mock_rename.assert_not_called()

    # Verify cancellation message
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("cancelled" in str(call).lower() for call in print_calls)


def test_main_compare_mode_not_invoice(tmp_path):
    """Test compare mode when neither method detects an invoice."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            # Both return not an invoice
                            mock_detector.analyze_document.return_value = (False, None)

                            with patch('docscan.cli.rename_invoice') as mock_rename:
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify rename was NOT called
    mock_rename.assert_not_called()

    # Verify message about not being an invoice
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("not an invoice" in str(call).lower() for call in print_calls)


def test_main_compare_mode_dry_run(tmp_path):
    """Test compare mode with dry run."""
    pdf_file = tmp_path / "invoice.pdf"
    pdf_file.touch()

    invoice_data = {
        "date": "2024-01-15",
        "invoicing_party": "ACME_Corp",
    }

    with patch('sys.argv', ['docscan', str(pdf_file), '--compare', '--dry-run']):
        with patch('docscan.cli.is_valid_pdf', return_value=True):
            with patch('docscan.cli.load_config', return_value={
                "vlm_model": "test-vlm",
                "text_llm_model": "test-text-llm"
            }):
                with patch('docscan.cli.setup_logging'):
                    with patch('docscan.cli.ModelManager') as MockModelManager:
                        mock_manager = MockModelManager.return_value
                        mock_manager.load_model.return_value = (MagicMock(), MagicMock())

                        with patch('docscan.cli.InvoiceDetector') as MockDetector:
                            mock_detector = MockDetector.return_value
                            mock_detector.analyze_document.return_value = (True, invoice_data)

                            with patch('docscan.cli.rename_invoice') as mock_rename:
                                with patch('builtins.print') as mock_print:
                                    main()

    # Verify rename was NOT called
    mock_rename.assert_not_called()

    # Verify dry run message was printed
    print_calls = [str(call) for call in mock_print.call_args_list]
    assert any("DRY RUN" in str(call) for call in print_calls)


def test_prompt_user_choice_keyboard_interrupt():
    """Test user input with keyboard interrupt."""
    from docscan.cli import _prompt_user_choice

    with patch('builtins.input', side_effect=KeyboardInterrupt):
        with patch('builtins.print'):
            result = _prompt_user_choice()

    assert result == 0


def test_prompt_user_choice_eof():
    """Test user input with EOF."""
    from docscan.cli import _prompt_user_choice

    with patch('builtins.input', side_effect=EOFError):
        with patch('builtins.print'):
            result = _prompt_user_choice()

    assert result == 0


def test_prompt_user_choice_invalid_then_valid():
    """Test user input with invalid choice followed by valid choice."""
    from docscan.cli import _prompt_user_choice

    with patch('builtins.input', side_effect=['invalid', '99', '1']):
        with patch('builtins.print'):
            result = _prompt_user_choice()

    assert result == 1
