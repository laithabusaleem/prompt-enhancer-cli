"""Tests for the PromptParser module."""

import pytest
from prompt_enhancer.parser import PromptParser, ParsedPrompt


class TestPromptParser:
    """Test suite for PromptParser."""
    
    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PromptParser(use_embeddings=False)
    
    # Normal case 1: Basic instruction prompt
    def test_parse_basic_instruction(self, parser):
        """Test parsing a basic instruction prompt."""
        prompt = "Write a Python function to calculate the factorial of a number."
        result = parser.parse(prompt)
        
        assert isinstance(result, ParsedPrompt)
        assert result.original == prompt
        assert "write" in result.action_verbs
        assert "Write" in result.intent.lower() or "write" in result.intent.lower()
    
    # Normal case 2: Prompt with constraints
    def test_parse_prompt_with_constraints(self, parser):
        """Test parsing a prompt with explicit constraints."""
        prompt = "Write a summary in 100 words or less. Must include key points. Don't use technical jargon."
        result = parser.parse(prompt)
        
        assert isinstance(result, ParsedPrompt)
        assert len(result.constraints) > 0
        # Should detect word limit constraint
        constraint_text = " ".join(result.constraints).lower()
        assert "100" in constraint_text or "word" in constraint_text or "don't" in constraint_text
    
    # Edge case: Empty prompt
    def test_parse_empty_prompt(self, parser):
        """Test parsing an empty prompt."""
        prompt = ""
        result = parser.parse(prompt)
        
        assert isinstance(result, ParsedPrompt)
        assert result.original == ""
        assert result.intent == ""
    
    def test_parse_extracts_context(self, parser):
        """Test that context is extracted from prompts."""
        prompt = "For a beginner programmer, explain how recursion works in Python."
        result = parser.parse(prompt)
        
        assert isinstance(result, ParsedPrompt)
        # Should detect context indicator "for a beginner programmer"
        assert len(result.context) > 0 or "beginner" in result.intent.lower()
    
    def test_parse_extracts_entities(self, parser):
        """Test that named entities are extracted."""
        prompt = "Compare Python and JavaScript for web development."
        result = parser.parse(prompt)
        
        assert isinstance(result, ParsedPrompt)
        # Python and JavaScript should be detected as entities
        entities_str = " ".join(result.entities)
        assert "Python" in entities_str or "JavaScript" in entities_str
    
    def test_parse_action_verbs(self, parser):
        """Test extraction of multiple action verbs."""
        prompt = "Analyze and compare the two approaches, then summarize your findings."
        result = parser.parse(prompt)
        
        assert "analyze" in result.action_verbs or "compare" in result.action_verbs or "summarize" in result.action_verbs
    
    def test_to_dict(self, parser):
        """Test conversion to dictionary."""
        prompt = "Write a simple function."
        result = parser.parse(prompt)
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "original" in result_dict
        assert "intent" in result_dict
        assert "context" in result_dict
        assert "constraints" in result_dict
