"""Tests for the Config module."""

import os
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open
from prompt_enhancer.config import TemplateManager, DEFAULT_TEMPLATES

class TestTemplateManager:
    """Test suite for TemplateManager."""
    
    def test_initialization_defaults(self):
        """Test initialization with default templates."""
        with patch.object(Path, 'exists', return_value=False):
            manager = TemplateManager()
            assert manager.templates == DEFAULT_TEMPLATES
            
    def test_load_custom_config(self):
        """Test loading templates from a custom file."""
        custom_yaml = """
templates:
  instruction: "Custom Instruction: {intent}"
"""
        with patch('builtins.open', mock_open(read_data=custom_yaml)):
            with patch.object(Path, 'exists', return_value=True):
                manager = TemplateManager(config_path="dummy.yaml")
                
                # Check custom template
                assert manager.get_template("instruction") == "Custom Instruction: {intent}"
                # Check other defaults are preserved
                assert manager.get_template("question") == DEFAULT_TEMPLATES["question"]
                
    def test_get_template_fallback(self):
        """Test fallback for unknown template types."""
        with patch.object(Path, 'exists', return_value=False):
            manager = TemplateManager()
            # "random_type" doesn't exist, should return unknown template
            assert manager.get_template("random_type") == DEFAULT_TEMPLATES["unknown"]

    def test_save_default_config(self):
        """Test saving default configuration."""
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pathlib.Path.mkdir') as mock_mkdir:
                manager = TemplateManager()
                manager.CONFIG_FILE = Path("test_config.yaml")
                manager.CONFIG_DIR = Path("test_dir")
                
                path = manager.save_default_config()
                
                mock_mkdir.assert_called_once()
                mock_file.assert_called_with(Path("test_config.yaml"), 'w', encoding='utf-8')
                assert path == Path("test_config.yaml")

    def test_load_malformed_config(self):
        """Test graceful handling of malformed config."""
        # Use content that causes yaml.safe_load to fail or produce unexpected type
        with patch('builtins.open', mock_open(read_data=": invalid yaml")):
            with patch.object(Path, 'exists', return_value=True):
                # Should not raise exception despite loading error
                manager = TemplateManager(config_path="bad_config.yaml")
                # Should fallback to defaults
                assert manager.templates == DEFAULT_TEMPLATES
