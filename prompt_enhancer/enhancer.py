"""
Prompt Enhancer Module - Enhances prompts based on parsed components.
"""

from dataclasses import dataclass
from typing import Optional

from prompt_enhancer.parser import PromptParser, ParsedPrompt
from prompt_enhancer.classifier import PromptClassifier, ClassificationResult, PromptType


@dataclass
class EnhancedPrompt:
    """Data class for enhanced prompt output."""
    
    original: str
    enhanced: str
    parsed: ParsedPrompt
    classification: ClassificationResult
    enhancement_notes: list
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original": self.original,
            "enhanced": self.enhanced,
            "parsed": self.parsed.to_dict(),
            "classification": self.classification.to_dict(),
            "enhancement_notes": self.enhancement_notes,
        }


class PromptEnhancer:
    """
    Enhances prompts by adding structure, clarity, and context.
    
    Combines parsing and classification to create improved prompts.
    """
    
    # Enhancement templates for different prompt types
    ENHANCEMENT_TEMPLATES = {
        PromptType.INSTRUCTION: """Task: {intent}

Context:
{context}

Requirements:
{constraints}

Please provide a clear and structured response.""",

        PromptType.QUESTION: """Question: {intent}

Background Context:
{context}

Please provide a comprehensive answer that addresses all aspects of the question.""",

        PromptType.CREATIVE: """Creative Writing Task: {intent}

Setting/Context:
{context}

Guidelines:
{constraints}

Be imaginative and engaging in your response.""",

        PromptType.ANALYSIS: """Analysis Task: {intent}

Subject Context:
{context}

Analysis Requirements:
{constraints}

Provide a thorough and objective analysis.""",

        PromptType.CODE: """Programming Task: {intent}

Technical Context:
{context}

Technical Requirements:
{constraints}

Provide clean, well-documented code with explanations.""",

        PromptType.TRANSLATION: """Translation Task: {intent}

Source Context:
{context}

Translation Guidelines:
{constraints}

Provide an accurate translation that preserves meaning and tone.""",

        PromptType.SUMMARIZATION: """Summarization Task: {intent}

Content Context:
{context}

Summary Requirements:
{constraints}

Provide a concise yet comprehensive summary.""",

        PromptType.EXTRACTION: """Information Extraction Task: {intent}

Data Context:
{context}

Extraction Requirements:
{constraints}

Extract and organize the requested information clearly.""",

        PromptType.CLASSIFICATION: """Classification Task: {intent}

Classification Context:
{context}

Classification Criteria:
{constraints}

Provide clear categorization with justification.""",

        PromptType.CONVERSATION: """{intent}

{context}""",

        PromptType.UNKNOWN: """Request: {intent}

Additional Context:
{context}

Requirements:
{constraints}""",
    }
    
    def __init__(self, use_embeddings: bool = False):
        """
        Initialize the prompt enhancer.
        
        Args:
            use_embeddings: Whether to use sentence embeddings for better analysis.
        """
        self.parser = PromptParser(use_embeddings=use_embeddings)
        self.classifier = PromptClassifier(use_embeddings=use_embeddings)
        self.use_embeddings = use_embeddings
    
    def enhance(self, prompt: str, add_structure: bool = True) -> EnhancedPrompt:
        """
        Enhance a prompt with structure and clarity.
        
        Args:
            prompt: The original prompt string.
            add_structure: Whether to add structured formatting.
            
        Returns:
            EnhancedPrompt with original, enhanced text, and metadata.
        """
        # Parse the prompt
        parsed = self.parser.parse(prompt)
        
        # Classify the prompt
        classification = self.classifier.classify(prompt)
        
        # Track enhancement notes
        notes = []
        
        if add_structure:
            enhanced = self._apply_template(parsed, classification, notes)
        else:
            enhanced = self._enhance_inline(parsed, classification, notes)
        
        return EnhancedPrompt(
            original=prompt,
            enhanced=enhanced,
            parsed=parsed,
            classification=classification,
            enhancement_notes=notes,
        )
    
    def _apply_template(
        self, 
        parsed: ParsedPrompt, 
        classification: ClassificationResult,
        notes: list
    ) -> str:
        """Apply a structured template based on prompt type."""
        
        template = self.ENHANCEMENT_TEMPLATES.get(
            classification.primary_type,
            self.ENHANCEMENT_TEMPLATES[PromptType.UNKNOWN]
        )
        
        # Prepare context string
        if parsed.context:
            context_str = "\n".join(f"- {ctx}" for ctx in parsed.context)
            notes.append("Added structured context from parsed elements")
        else:
            context_str = "- General context from original prompt"
            notes.append("No specific context detected, using general context")
        
        # Prepare constraints string
        if parsed.constraints:
            constraints_str = "\n".join(f"- {con}" for con in parsed.constraints)
            notes.append("Extracted and formatted constraints")
        else:
            constraints_str = "- Follow standard best practices"
            notes.append("No specific constraints detected, using defaults")
        
        # Format the template
        enhanced = template.format(
            intent=parsed.intent,
            context=context_str,
            constraints=constraints_str,
        )
        
        notes.append(f"Applied {classification.primary_type.value} template")
        
        # Clean up any empty sections
        enhanced = self._clean_enhanced_prompt(enhanced)
        
        return enhanced
    
    def _enhance_inline(
        self,
        parsed: ParsedPrompt,
        classification: ClassificationResult,
        notes: list
    ) -> str:
        """Enhance the prompt inline without heavy restructuring."""
        
        enhanced_parts = [parsed.original]
        
        # Add clarifying elements if missing
        if not parsed.constraints:
            # Add helpful default guidance
            if classification.primary_type == PromptType.CODE:
                enhanced_parts.append("\nPlease include comments and error handling.")
                notes.append("Added code quality guidance")
            elif classification.primary_type == PromptType.CREATIVE:
                enhanced_parts.append("\nBe creative and engaging.")
                notes.append("Added creative guidance")
            elif classification.primary_type == PromptType.ANALYSIS:
                enhanced_parts.append("\nProvide a balanced and thorough analysis.")
                notes.append("Added analysis guidance")
        
        return " ".join(enhanced_parts)
    
    def _clean_enhanced_prompt(self, prompt: str) -> str:
        """Clean up the enhanced prompt by removing empty sections."""
        lines = prompt.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip lines that are just placeholders
            if stripped in ["- ", "-", ""]:
                skip_next_empty = True
                continue
            
            # Skip empty lines after sections with no content
            if skip_next_empty and stripped == "":
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def analyze(self, prompt: str) -> dict:
        """
        Analyze a prompt without enhancement.
        
        Args:
            prompt: The prompt to analyze.
            
        Returns:
            Dictionary with parsing and classification results.
        """
        parsed = self.parser.parse(prompt)
        classification = self.classifier.classify(prompt)
        
        return {
            "parsed": parsed.to_dict(),
            "classification": classification.to_dict(),
            "type_description": self.classifier.get_type_description(
                classification.primary_type
            ),
        }
