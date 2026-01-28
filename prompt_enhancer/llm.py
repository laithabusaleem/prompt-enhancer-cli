"""
LLM Client Module - Handles interactions with OpenAI and Mistral APIs.
"""

import os
import json
from enum import Enum
from typing import Optional, Dict, Any, List

import httpx
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    MISTRAL = "mistral"
    GOOGLE = "google"
    NONE = "none"

class LLMClient:
    """Client for interacting with LLM APIs."""
    
    def __init__(self, provider: str = "auto", model: str = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: 'openai', 'mistral', 'google', or 'auto'
            model: Specific model to use (default depends on provider)
        """
        self.provider = self._resolve_provider(provider)
        self.model = model
        self.api_key = self._get_api_key(self.provider)
        
        self.client = None
        if self.provider == LLMProvider.OPENAI:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.model = model or "gpt-3.5-turbo"
        elif self.provider == LLMProvider.MISTRAL:
            from mistralai import Mistral
            self.client = Mistral(api_key=self.api_key)
            self.model = model or "mistral-tiny"
        elif self.provider == LLMProvider.GOOGLE:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = model or "gemini-2.0-flash"
            self.client = genai.GenerativeModel(self.model)
            
    def _resolve_provider(self, requested: str) -> LLMProvider:
        """Resolve the provider to use."""
        if requested != "auto":
            return LLMProvider(requested)
            
        if os.getenv("OPENAI_API_KEY"):
            return LLMProvider.OPENAI
        elif os.getenv("MISTRAL_API_KEY"):
            return LLMProvider.MISTRAL
        elif os.getenv("GOOGLE_API_KEY"):
            return LLMProvider.GOOGLE
        else:
            return LLMProvider.NONE
            
    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for the provider."""
        if provider == LLMProvider.OPENAI:
            return os.getenv("OPENAI_API_KEY")
        elif provider == LLMProvider.MISTRAL:
            return os.getenv("MISTRAL_API_KEY")
        elif provider == LLMProvider.GOOGLE:
            return os.getenv("GOOGLE_API_KEY")
        return None
        
    def generate_enhancement(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Generate an enhanced prompt using the LLM.
        
        Args:
            prompt: The original prompt.
            context: Parsed context dictionary (intent, type, etc.)
            
        Returns:
            The enhanced prompt text.
        """
        if self.provider == LLMProvider.NONE:
            raise ValueError("No valid API keys found. Please set OPENAI_API_KEY, MISTRAL_API_KEY, or GOOGLE_API_KEY.")
            
        system_msg = self._build_system_message(context)
        user_msg = f"Original prompt: {prompt}"
        
        if self.provider == LLMProvider.OPENAI:
            return self._call_openai(system_msg, user_msg)
        elif self.provider == LLMProvider.MISTRAL:
            return self._call_mistral(system_msg, user_msg)
        elif self.provider == LLMProvider.GOOGLE:
            return self._call_google(system_msg, user_msg)
            
        return prompt

    def _build_system_message(self, context: Dict[str, Any]) -> str:
        """Build the system message for the LLM."""
        prompt_type = context.get('classification', {}).get('primary_type', 'general')
        
        return f"""You are an expert prompt engineer. Your task is to rewrite and enhance the user's prompt to be more effective for an AI model.
        
Context:
- Prompt Type: {prompt_type}
- Intent: {context.get('parsed', {}).get('intent', 'unknown')}

Guidelines:
1. Make the prompt specific, clear, and context-rich.
2. Use a structured format (e.g., Role, Context, Task, Constraints).
3. Do not change the original intent.
4. If constraints are missing, utilize best practices to suggest reasonable defaults.
5. Output ONLY the enhanced prompt, no conversational filler.
"""

    def _call_openai(self, system: str, user: str) -> str:
        """Call OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _call_mistral(self, system: str, user: str) -> str:
        """Call Mistral API."""
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise RuntimeError(f"Mistral API call failed: {str(e)}")

    def _call_google(self, system: str, user: str) -> str:
        """Call Google Gemini API."""
        try:
            # Gemini models often handle system instructions differently or prefer combined prompts.
            # We'll combine system and user message for simplicity as 'generate_content' is standard.
            combined_prompt = f"{system}\n\n{user}"
            response = self.client.generate_content(combined_prompt)
            return response.text.strip()
        except Exception as e:
            raise RuntimeError(f"Google Gemini API call failed: {str(e)}")
