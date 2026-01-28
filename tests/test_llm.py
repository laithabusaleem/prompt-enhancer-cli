"""Tests for the LLM Client module."""

import pytest
from unittest.mock import MagicMock, patch
from prompt_enhancer.llm import LLMClient, LLMProvider


class TestLLMClient:
    """Test suite for LLMClient."""
    
    def test_initialization_auto_openai(self):
        """Test auto-discovery of OpenAI provider."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test', 'MISTRAL_API_KEY': '', 'GOOGLE_API_KEY': ''}):
            client = LLMClient(provider='auto')
            assert client.provider == LLMProvider.OPENAI
            assert client.api_key == 'sk-test'

    def test_initialization_auto_mistral(self):
        """Test auto-discovery of Mistral provider."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': '', 'MISTRAL_API_KEY': 'mistral-test', 'GOOGLE_API_KEY': ''}):
            client = LLMClient(provider='auto')
            assert client.provider == LLMProvider.MISTRAL
            assert client.api_key == 'mistral-test'

    def test_initialization_auto_google(self):
        """Test auto-discovery of Google provider."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': '', 'MISTRAL_API_KEY': '', 'GOOGLE_API_KEY': 'google-test'}):
            client = LLMClient(provider='auto')
            assert client.provider == LLMProvider.GOOGLE
            assert client.api_key == 'google-test'

    def test_initialization_none(self):
        """Test initialization with no keys."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': '', 'MISTRAL_API_KEY': '', 'GOOGLE_API_KEY': ''}):
            client = LLMClient(provider='auto')
            assert client.provider == LLMProvider.NONE

    @patch('prompt_enhancer.llm.LLMClient._call_openai')
    def test_generate_enhancement_openai(self, mock_call):
        """Test enhancement generation with OpenAI."""
        mock_call.return_value = "Enhanced prompt"
        
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'sk-test'}):
            client = LLMClient(provider='openai')
            client.client = MagicMock()
            
            context = {"classification": {"primary_type": "code"}}
            result = client.generate_enhancement("test prompt", context)
            
            assert result == "Enhanced prompt"
            mock_call.assert_called_once()
            
    @patch('prompt_enhancer.llm.LLMClient._call_google')
    def test_generate_enhancement_google(self, mock_call):
        """Test enhancement generation with Google."""
        mock_call.return_value = "Gemini Enhanced"
        
        with patch.dict('os.environ', {'GOOGLE_API_KEY': 'google-test'}):
            client = LLMClient(provider='google')
            client.client = MagicMock()
            
            context = {"classification": {"primary_type": "creative"}}
            result = client.generate_enhancement("test prompt", context)
            
            assert result == "Gemini Enhanced"
            mock_call.assert_called_once()

    def test_build_system_message(self):
        """Test system message construction."""
        client = LLMClient(provider='none')
        context = {
            "classification": {"primary_type": "coding"},
            "parsed": {"intent": "write code"}
        }
        
        msg = client._build_system_message(context)
        assert "expert prompt engineer" in msg
        assert "coding" in msg
