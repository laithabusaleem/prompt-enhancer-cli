"""
Configuration Module - Handles custom templates and settings.
"""

import os
import yaml
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional

from prompt_enhancer.classifier import PromptType

# Default templates moved from enhancer.py
DEFAULT_TEMPLATES = {
    "instruction": """Task: {intent}

Context:
{context}

Requirements:
{constraints}

Please provide a clear and structured response.""",

    "question": """Question: {intent}

Background Context:
{context}

Please provide a comprehensive answer that addresses all aspects of the question.""",

    "creative": """Creative Writing Task: {intent}

Setting/Context:
{context}

Guidelines:
{constraints}

Be imaginative and engaging in your response.""",

    "analysis": """Analysis Task: {intent}

Subject Context:
{context}

Analysis Requirements:
{constraints}

Provide a thorough and objective analysis.""",

    "code": """Programming Task: {intent}

Technical Context:
{context}

Technical Requirements:
{constraints}

Provide clean, well-documented code with explanations.""",

    "translation": """Translation Task: {intent}

Source Context:
{context}

Translation Guidelines:
{constraints}

Provide an accurate translation that preserves meaning and tone.""",

    "summarization": """Summarization Task: {intent}

Content Context:
{context}

Summary Requirements:
{constraints}

Provide a concise yet comprehensive summary.""",

    "extraction": """Information Extraction Task: {intent}

Data Context:
{context}

Extraction Requirements:
{constraints}

Extract and organize the requested information clearly.""",

    "classification": """Classification Task: {intent}

Classification Context:
{context}

Classification Criteria:
{constraints}

Provide clear categorization with justification.""",

    "conversation": """{intent}

{context}""",

    "unknown": """Request: {intent}

Additional Context:
{context}

Requirements:
{constraints}""",
}

class TemplateManager:
    """Manages prompt enhancement templates."""
    
    CONFIG_DIR = Path.home() / ".prompt-enhancer"
    CONFIG_FILE = CONFIG_DIR / "templates.yaml"
    
    def __init__(self, config_path: str = None):
        """
        Initialize the template manager.
        
        Args:
            config_path: Path to custom config file (optional).
                         If None, checks default location.
        """
        self.templates = deepcopy(DEFAULT_TEMPLATES)
        self.config_path = Path(config_path) if config_path else self.CONFIG_FILE
        self._load_config()
        
    def _load_config(self):
        """Load templates from configuration file if it exists."""
        if not self.config_path.exists():
            return
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                custom_config = yaml.safe_load(f)
                
            if custom_config and "templates" in custom_config:
                # Merge custom templates, overwriting defaults where keys match
                for key, template in custom_config["templates"].items():
                    # Normalize key to match PromptType values if possible
                    # or allow custom keys
                    self.templates[key] = template.strip()
                    
        except Exception as e:
            # We don't want to crash if config is bad, just warn/log
            print(f"Warning: Failed to load config from {self.config_path}: {e}")
            
    def get_template(self, prompt_type: str) -> str:
        """
        Get template for a specific prompt type.
        
        Args:
            prompt_type: The type of prompt (instruction, code, etc.)
            
        Returns:
            The template string.
        """
        # Try direct match
        if prompt_type in self.templates:
            return self.templates[prompt_type]
            
        # Fallback to unknown or first available
        return self.templates.get("unknown", DEFAULT_TEMPLATES["unknown"])
        
    def save_default_config(self) -> Path:
        """
        Save default configuration to disk.
        
        Returns:
            Path to the saved configuration file.
        """
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        config_data = {
            "templates": DEFAULT_TEMPLATES,
            "description": "Custom templates for Prompt Enhancer CLI",
            "version": "1.0"
        }
        
        with open(self.CONFIG_FILE, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            
        return self.CONFIG_FILE
