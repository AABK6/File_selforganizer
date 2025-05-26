import unittest
import hashlib
import tempfile
import os
import logging
from unittest.mock import patch, mock_open

# Attempt to import functions from src.utils
# This relative import assumes that the tests are run from the project root directory,
# or that the PYTHONPATH is set up correctly to include the 'src' directory.
try:
    from src.utils import get_file_hash, extract_text_from_txt
except ImportError:
    # Fallback for environments where 'src' might not be in PYTHONPATH directly
    # This might happen in some CI/IDE setups.
    # Adjusting sys.path should be a last resort and done carefully.
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.utils import get_file_hash, extract_text_from_txt

# Disable logging output during tests to keep test output clean
logging.disable(logging.CRITICAL)


class TestHashingUtils(unittest.TestCase):
    """Test cases for hashing utility functions."""

    def test_get_file_hash_smoke(self):
        """Test get_file_hash with basic content."""
        content = b"hello world"
        expected_hash = hashlib.sha256(content).hexdigest()
        
        temp_file = None
        try:
            # Create a temporary file
            fd, temp_file_path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(content)
            
            # Get hash using the utility function
            calculated_hash = get_file_hash(temp_file_path)
            self.assertEqual(calculated_hash, expected_hash)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_get_file_hash_empty_file(self):
        """Test get_file_hash with an empty file."""
        content = b""
        expected_hash = hashlib.sha256(content).hexdigest()
        
        temp_file = None
        try:
            fd, temp_file_path = tempfile.mkstemp()
            with os.fdopen(fd, 'wb') as tmp:
                tmp.write(content) # Write empty content
            
            calculated_hash = get_file_hash(temp_file_path)
            self.assertEqual(calculated_hash, expected_hash)
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)


class TestExtractionUtils(unittest.TestCase):
    """Test cases for text extraction utility functions."""

    @patch('src.utils.open', new_callable=mock_open, read_data="Test content from TXT file.")
    def test_extract_text_from_txt_success(self, mock_file_open):
        """Test successful text extraction from a mocked .txt file."""
        filepath = "dummy/path.txt"
        expected_content = "Test content from TXT file."
        
        extracted_content = extract_text_from_txt(filepath)
        
        self.assertEqual(extracted_content, expected_content)
        mock_file_open.assert_called_once_with(filepath, 'r', encoding='utf-8', errors='ignore')

    @patch('src.utils.logger.warning')
    @patch('src.utils.open', side_effect=FileNotFoundError("File not found for testing"))
    def test_extract_text_from_txt_file_not_found(self, mock_open_call, mock_logger_warning):
        """Test extract_text_from_txt when the file is not found."""
        filepath = "non_existent.txt"
        
        result = extract_text_from_txt(filepath)
        
        self.assertEqual(result, "") # Should return empty string on failure
        mock_open_call.assert_called_once_with(filepath, 'r', encoding='utf-8', errors='ignore')
        # Check that logger.warning was called with the expected message
        # The exact formatting of the logged message depends on the logger setup in utils.py
        # Here we check if it's called with a message containing the filepath and the error.
        self.assertTrue(mock_logger_warning.called)
        args, _ = mock_logger_warning.call_args
        self.assertIn(filepath, args[0]) # Check if filepath is in the log message
        self.assertIn("File not found for testing", args[0]) # Check if error message is in log


if __name__ == '__main__':
    unittest.main()
