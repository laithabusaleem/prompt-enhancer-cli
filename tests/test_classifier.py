"""Tests for the PromptClassifier module."""

import pytest
from prompt_enhancer.classifier import PromptClassifier, PromptType, ClassificationResult


class TestPromptClassifier:
    """Test suite for PromptClassifier."""
    
    @pytest.fixture
    def classifier(self):
        """Create a classifier instance."""
        return PromptClassifier(use_embeddings=False)
    
    # Normal case 1: Question classification
    def test_classify_question(self, classifier):
        """Test classifying a question prompt."""
        prompt = "What is the capital of France?"
        result = classifier.classify(prompt)
        
        assert isinstance(result, ClassificationResult)
        assert result.primary_type == PromptType.QUESTION
        assert result.confidence > 0
    
    # Normal case 2: Code/programming classification
    def test_classify_code_prompt(self, classifier):
        """Test classifying a code-related prompt."""
        prompt = "Write a Python function to calculate fibonacci numbers."
        result = classifier.classify(prompt)
        
        assert isinstance(result, ClassificationResult)
        # Should be classified as code or instruction
        assert result.primary_type in [PromptType.CODE, PromptType.INSTRUCTION]
    
    # Edge case: Ambiguous prompt
    def test_classify_ambiguous_prompt(self, classifier):
        """Test classifying an ambiguous or minimal prompt."""
        prompt = "Hello"
        result = classifier.classify(prompt)
        
        assert isinstance(result, ClassificationResult)
        # Should still return a valid type
        assert result.primary_type in PromptType
    
    def test_classify_summarization(self, classifier):
        """Test classifying a summarization prompt."""
        prompt = "Summarize the key points of this article about climate change."
        result = classifier.classify(prompt)
        
        assert result.primary_type == PromptType.SUMMARIZATION
    
    def test_classify_creative(self, classifier):
        """Test classifying a creative writing prompt."""
        prompt = "Write a short story about a dragon who learns to cook."
        result = classifier.classify(prompt)
        
        # Creative prompts can be detected as creative or instruction
        assert result.primary_type in [PromptType.CREATIVE, PromptType.INSTRUCTION]
    
    def test_classify_translation(self, classifier):
        """Test classifying a translation prompt."""
        prompt = "Translate the following text to Spanish: Hello, how are you?"
        result = classifier.classify(prompt)
        
        assert result.primary_type == PromptType.TRANSLATION
    
    def test_classify_analysis(self, classifier):
        """Test classifying an analysis prompt."""
        prompt = "Analyze the pros and cons of remote work."
        result = classifier.classify(prompt)
        
        assert result.primary_type == PromptType.ANALYSIS
    
    def test_classification_result_to_dict(self, classifier):
        """Test conversion of classification result to dictionary."""
        prompt = "What is Python?"
        result = classifier.classify(prompt)
        
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "primary_type" in result_dict
        assert "confidence" in result_dict
        assert "all_scores" in result_dict
    
    def test_get_type_description(self, classifier):
        """Test getting type descriptions."""
        description = classifier.get_type_description(PromptType.CODE)
        
        assert isinstance(description, str)
        assert len(description) > 0
    
    def test_all_prompt_types_have_descriptions(self, classifier):
        """Test that all prompt types have descriptions."""
        for ptype in PromptType:
            description = classifier.get_type_description(ptype)
            assert isinstance(description, str)
            assert len(description) > 0
