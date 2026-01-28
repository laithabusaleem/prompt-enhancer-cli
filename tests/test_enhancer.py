"""Tests for the PromptEnhancer module."""

import pytest
from prompt_enhancer.enhancer import PromptEnhancer, EnhancedPrompt
from prompt_enhancer.classifier import PromptType


class TestPromptEnhancer:
    """Test suite for PromptEnhancer."""
    
    @pytest.fixture
    def enhancer(self):
        """Create an enhancer instance."""
        return PromptEnhancer(use_embeddings=False)
    
    # Normal case 1: Basic enhancement
    def test_enhance_basic_prompt(self, enhancer):
        """Test enhancing a basic instruction prompt."""
        prompt = "Write a Python function to sort a list."
        result = enhancer.enhance(prompt)
        
        assert isinstance(result, EnhancedPrompt)
        assert result.original == prompt
        assert len(result.enhanced) > len(prompt)  # Enhanced should be longer
        assert result.parsed is not None
        assert result.classification is not None
    
    # Normal case 2: Enhancement with constraints
    def test_enhance_prompt_with_constraints(self, enhancer):
        """Test enhancing a prompt with constraints."""
        prompt = "Write a summary of the article. Must be under 100 words."
        result = enhancer.enhance(prompt)
        
        assert isinstance(result, EnhancedPrompt)
        # Enhanced prompt should include the constraint
        assert "100" in result.enhanced or "word" in result.enhanced.lower()
    
    # Edge case: Very short prompt
    def test_enhance_short_prompt(self, enhancer):
        """Test enhancing a very short prompt."""
        prompt = "Help"
        result = enhancer.enhance(prompt)
        
        assert isinstance(result, EnhancedPrompt)
        assert result.enhanced is not None
        assert len(result.enhanced) >= len(prompt)
    
    def test_enhance_code_prompt(self, enhancer):
        """Test enhancing a code-related prompt."""
        prompt = "Debug this Python code that has an error."
        result = enhancer.enhance(prompt)
        
        assert isinstance(result, EnhancedPrompt)
        # Should detect as code type
        assert result.classification.primary_type in [PromptType.CODE, PromptType.INSTRUCTION]
    
    def test_enhance_inline_mode(self, enhancer):
        """Test inline enhancement mode (no heavy restructuring)."""
        prompt = "Explain recursion."
        result = enhancer.enhance(prompt, add_structure=False)
        
        assert isinstance(result, EnhancedPrompt)
        assert result.enhanced is not None
    
    def test_enhanced_prompt_to_dict(self, enhancer):
        """Test conversion of enhanced prompt to dictionary."""
        prompt = "Write a poem."
        result = enhancer.enhance(prompt)
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "original" in result_dict
        assert "enhanced" in result_dict
        assert "parsed" in result_dict
        assert "classification" in result_dict
        assert "enhancement_notes" in result_dict
    
    def test_analyze_prompt(self, enhancer):
        """Test the analyze method."""
        prompt = "Summarize the key findings from the research paper."
        result = enhancer.analyze(prompt)
        
        assert isinstance(result, dict)
        assert "parsed" in result
        assert "classification" in result
        assert "type_description" in result
    
    def test_enhancement_notes_added(self, enhancer):
        """Test that enhancement notes are added during processing."""
        prompt = "Write a function."
        result = enhancer.enhance(prompt)
        
        assert isinstance(result.enhancement_notes, list)
        assert len(result.enhancement_notes) > 0
    
    def test_enhance_preserves_original(self, enhancer):
        """Test that original prompt is preserved."""
        prompt = "Create a story about space exploration."
        result = enhancer.enhance(prompt)
        
        assert result.original == prompt
        assert result.parsed.original == prompt
