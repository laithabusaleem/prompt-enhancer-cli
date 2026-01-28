"""
Prompt Classifier Module - Classifies prompts into categories.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


class PromptType(str, Enum):
    """Enumeration of prompt types."""
    
    INSTRUCTION = "instruction"
    QUESTION = "question"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    CONVERSATION = "conversation"
    CODE = "code"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """Data class for classification results."""
    
    primary_type: PromptType
    confidence: float
    all_scores: Dict[str, float]
    features: Dict[str, bool]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "primary_type": self.primary_type.value,
            "confidence": round(self.confidence, 3),
            "all_scores": {k: round(v, 3) for k, v in self.all_scores.items()},
            "features": self.features,
        }


class PromptClassifier:
    """
    Classifies prompts into predefined categories.
    
    Uses rule-based classification with keyword matching and pattern detection.
    """
    
    # Keywords and patterns for each prompt type
    TYPE_PATTERNS = {
        PromptType.QUESTION: {
            "starts_with": ["what", "who", "where", "when", "why", "how", "which", "is", "are", "can", "could", "would", "should", "do", "does", "did"],
            "contains": ["?"],
            "weight": 1.0,
        },
        PromptType.INSTRUCTION: {
            "starts_with": ["write", "create", "make", "generate", "produce", "build", "develop", "design", "draft", "compose"],
            "contains": ["please", "i need", "i want", "give me", "provide"],
            "weight": 0.9,
        },
        PromptType.CREATIVE: {
            "starts_with": ["imagine", "pretend", "suppose", "create a story", "write a poem", "write a song"],
            "contains": ["story", "poem", "creative", "fiction", "narrative", "tale", "novel", "character", "plot", "scene"],
            "weight": 0.85,
        },
        PromptType.ANALYSIS: {
            "starts_with": ["analyze", "evaluate", "assess", "examine", "investigate", "study", "review"],
            "contains": ["analysis", "evaluate", "compare", "contrast", "pros and cons", "advantages", "disadvantages", "impact", "implications"],
            "weight": 0.9,
        },
        PromptType.CODE: {
            "starts_with": ["code", "program", "implement", "debug", "fix the code", "refactor"],
            "contains": ["function", "class", "variable", "algorithm", "python", "javascript", "java", "code", "programming", "bug", "error", "syntax", "api", "database", "sql"],
            "weight": 0.95,
        },
        PromptType.TRANSLATION: {
            "starts_with": ["translate", "convert to"],
            "contains": ["translate", "translation", "in english", "in spanish", "in french", "in german", "in chinese", "in japanese", "to english", "to spanish"],
            "weight": 1.0,
        },
        PromptType.SUMMARIZATION: {
            "starts_with": ["summarize", "sum up", "give a summary", "tldr", "brief"],
            "contains": ["summary", "summarize", "brief", "concise", "key points", "main points", "overview", "tldr", "in short"],
            "weight": 0.95,
        },
        PromptType.EXTRACTION: {
            "starts_with": ["extract", "find", "identify", "list", "get"],
            "contains": ["extract", "pull out", "identify all", "list all", "find all", "get all"],
            "weight": 0.85,
        },
        PromptType.CLASSIFICATION: {
            "starts_with": ["classify", "categorize", "label", "tag"],
            "contains": ["classify", "categorize", "category", "classification", "which type", "what type", "what kind"],
            "weight": 0.9,
        },
        PromptType.CONVERSATION: {
            "starts_with": ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"],
            "contains": ["chat", "talk", "discuss", "conversation", "let's talk"],
            "weight": 0.7,
        },
    }
    
    def __init__(self, use_embeddings: bool = False):
        """
        Initialize the classifier.
        
        Args:
            use_embeddings: Whether to use sentence embeddings for classification.
        """
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        self._type_embeddings = None
        
        if use_embeddings:
            self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Load the sentence transformer model and compute type embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create representative sentences for each type
            type_descriptions = {
                PromptType.INSTRUCTION: "Write, create, or generate something for me",
                PromptType.QUESTION: "What is the answer to this question",
                PromptType.CREATIVE: "Write a creative story, poem, or fictional content",
                PromptType.ANALYSIS: "Analyze, evaluate, or compare something in detail",
                PromptType.CODE: "Write code, fix bugs, or implement a programming solution",
                PromptType.TRANSLATION: "Translate this text to another language",
                PromptType.SUMMARIZATION: "Summarize this content into key points",
                PromptType.EXTRACTION: "Extract specific information from this text",
                PromptType.CLASSIFICATION: "Classify or categorize this into types",
                PromptType.CONVERSATION: "Have a casual conversation or chat",
            }
            
            self._type_embeddings = {
                ptype: self._embedding_model.encode(desc)
                for ptype, desc in type_descriptions.items()
            }
        except ImportError:
            self.use_embeddings = False
    
    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a prompt into a category.
        
        Args:
            prompt: The input prompt string.
            
        Returns:
            ClassificationResult with type, confidence, and scores.
        """
        prompt_lower = prompt.lower().strip()
        first_word = prompt_lower.split()[0] if prompt_lower else ""
        
        scores: Dict[str, float] = {}
        features: Dict[str, bool] = {}
        
        # Rule-based scoring
        for ptype, patterns in self.TYPE_PATTERNS.items():
            score = 0.0
            
            # Check starts_with patterns
            for start in patterns.get("starts_with", []):
                if prompt_lower.startswith(start):
                    score += 0.5
                    features[f"starts_with_{start}"] = True
                    break
            
            # Check contains patterns
            for keyword in patterns.get("contains", []):
                if keyword in prompt_lower:
                    score += 0.3
                    features[f"contains_{keyword}"] = True
            
            # Apply weight
            score *= patterns.get("weight", 1.0)
            scores[ptype.value] = min(score, 1.0)
        
        # Embedding-based scoring (if enabled)
        if self.use_embeddings and self._embedding_model is not None:
            embedding_scores = self._compute_embedding_scores(prompt)
            # Combine rule-based and embedding scores
            for ptype in PromptType:
                if ptype != PromptType.UNKNOWN and ptype.value in embedding_scores:
                    rule_score = scores.get(ptype.value, 0.0)
                    emb_score = embedding_scores[ptype.value]
                    # Weighted combination: 60% rules, 40% embeddings
                    scores[ptype.value] = 0.6 * rule_score + 0.4 * emb_score
        
        # Determine primary type
        if scores:
            primary_type_str = max(scores, key=scores.get)
            primary_type = PromptType(primary_type_str)
            confidence = scores[primary_type_str]
        else:
            primary_type = PromptType.UNKNOWN
            confidence = 0.0
        
        # If confidence is too low, mark as unknown
        if confidence < 0.1:
            primary_type = PromptType.UNKNOWN
        
        return ClassificationResult(
            primary_type=primary_type,
            confidence=confidence,
            all_scores=scores,
            features=features,
        )
    
    def _compute_embedding_scores(self, prompt: str) -> Dict[str, float]:
        """Compute similarity scores using embeddings."""
        import numpy as np
        
        if self._embedding_model is None or self._type_embeddings is None:
            return {}
        
        prompt_embedding = self._embedding_model.encode(prompt)
        
        scores = {}
        for ptype, type_embedding in self._type_embeddings.items():
            # Cosine similarity
            similarity = np.dot(prompt_embedding, type_embedding) / (
                np.linalg.norm(prompt_embedding) * np.linalg.norm(type_embedding)
            )
            # Normalize to 0-1 range
            scores[ptype.value] = max(0.0, (similarity + 1) / 2)
        
        return scores
    
    def get_type_description(self, prompt_type: PromptType) -> str:
        """Get a human-readable description for a prompt type."""
        descriptions = {
            PromptType.INSTRUCTION: "A request to create, write, or generate content",
            PromptType.QUESTION: "A question seeking information or clarification",
            PromptType.CREATIVE: "A request for creative or fictional content",
            PromptType.ANALYSIS: "A request to analyze, evaluate, or compare",
            PromptType.CODE: "A request related to programming or code",
            PromptType.TRANSLATION: "A request to translate between languages",
            PromptType.SUMMARIZATION: "A request to summarize or condense content",
            PromptType.EXTRACTION: "A request to extract specific information",
            PromptType.CLASSIFICATION: "A request to classify or categorize",
            PromptType.CONVERSATION: "Casual conversation or chat",
            PromptType.UNKNOWN: "Unable to determine the prompt type",
        }
        return descriptions.get(prompt_type, "Unknown prompt type")
