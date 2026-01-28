"""
Prompt Parser Module - Extracts intent, context, and constraints from prompts.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedPrompt:
    """Data class representing a parsed prompt with extracted components."""
    
    original: str
    intent: str
    context: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    action_verbs: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert parsed prompt to dictionary."""
        return {
            "original": self.original,
            "intent": self.intent,
            "context": self.context,
            "constraints": self.constraints,
            "entities": self.entities,
            "action_verbs": self.action_verbs,
        }


class PromptParser:
    """
    Parses prompts to extract intent, context, and constraints.
    
    Uses rule-based extraction with optional semantic embeddings.
    """
    
    # Common action verbs indicating intent
    ACTION_VERBS = [
        "write", "create", "generate", "explain", "summarize", "analyze",
        "translate", "convert", "list", "describe", "compare", "evaluate",
        "suggest", "recommend", "help", "show", "tell", "find", "search",
        "define", "calculate", "solve", "design", "build", "develop",
        "review", "edit", "improve", "fix", "debug", "refactor",
        "outline", "draft", "compose", "elaborate", "simplify",
    ]
    
    # Constraint indicators
    CONSTRAINT_PATTERNS = [
        r"(?:must|should|need to|have to|required to)\s+(.+?)(?:\.|$)",
        r"(?:limit|maximum|minimum|at least|at most|no more than|no less than)\s+(.+?)(?:\.|$)",
        r"(?:in|using|with)\s+(\d+\s*(?:words?|sentences?|paragraphs?|characters?|lines?))",
        r"(?:format|style|tone):\s*(.+?)(?:\.|$)",
        r"(?:don't|do not|avoid|exclude|without)\s+(.+?)(?:\.|$)",
        r"(?:only|just|specifically)\s+(.+?)(?:\.|$)",
    ]
    
    # Context indicators
    CONTEXT_PATTERNS = [
        r"(?:for|about|regarding|concerning|related to)\s+(.+?)(?:\.|,|$)",
        r"(?:in the context of|given that|assuming|considering)\s+(.+?)(?:\.|$)",
        r"(?:as a|acting as|you are a|imagine you're a)\s+(.+?)(?:\.|,|$)",
        r"(?:background|context):\s*(.+?)(?:\.|$)",
    ]
    
    def __init__(self, use_embeddings: bool = False):
        """
        Initialize the prompt parser.
        
        Args:
            use_embeddings: Whether to use sentence embeddings for semantic analysis.
        """
        self.use_embeddings = use_embeddings
        self._embedding_model = None
        
        if use_embeddings:
            self._load_embedding_model()
    
    def _load_embedding_model(self):
        """Lazy load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            )
    
    def parse(self, prompt: str) -> ParsedPrompt:
        """
        Parse a prompt to extract its components.
        
        Args:
            prompt: The input prompt string.
            
        Returns:
            ParsedPrompt object with extracted components.
        """
        prompt = prompt.strip()
        
        intent = self._extract_intent(prompt)
        context = self._extract_context(prompt)
        constraints = self._extract_constraints(prompt)
        entities = self._extract_entities(prompt)
        action_verbs = self._extract_action_verbs(prompt)
        
        return ParsedPrompt(
            original=prompt,
            intent=intent,
            context=context,
            constraints=constraints,
            entities=entities,
            action_verbs=action_verbs,
        )
    
    def _extract_intent(self, prompt: str) -> str:
        """
        Extract the primary intent from the prompt.
        
        The intent is typically the main action or request in the prompt.
        """
        prompt_lower = prompt.lower()
        
        # Find the first action verb to determine intent
        for verb in self.ACTION_VERBS:
            pattern = rf'\b{verb}\b'
            match = re.search(pattern, prompt_lower)
            if match:
                # Extract the clause containing the verb
                start = match.start()
                # Find the end of the sentence or clause
                end_match = re.search(r'[.!?]', prompt[start:])
                end = start + end_match.start() if end_match else len(prompt)
                
                intent_clause = prompt[start:end].strip()
                return intent_clause[:200]  # Limit length
        
        # Fallback: use the first sentence as intent
        first_sentence = re.split(r'[.!?]', prompt)[0].strip()
        return first_sentence[:200]
    
    def _extract_context(self, prompt: str) -> List[str]:
        """Extract context information from the prompt."""
        contexts = []
        
        for pattern in self.CONTEXT_PATTERNS:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                context = match.group(1).strip()
                if context and len(context) > 3:
                    contexts.append(context)
        
        # Deduplicate while preserving order
        seen = set()
        unique_contexts = []
        for ctx in contexts:
            ctx_lower = ctx.lower()
            if ctx_lower not in seen:
                seen.add(ctx_lower)
                unique_contexts.append(ctx)
        
        return unique_contexts
    
    def _extract_constraints(self, prompt: str) -> List[str]:
        """Extract constraints and requirements from the prompt."""
        constraints = []
        
        for pattern in self.CONSTRAINT_PATTERNS:
            matches = re.finditer(pattern, prompt, re.IGNORECASE)
            for match in matches:
                constraint = match.group(0).strip()
                if constraint:
                    constraints.append(constraint)
        
        # Deduplicate while preserving order
        seen = set()
        unique_constraints = []
        for con in constraints:
            con_lower = con.lower()
            if con_lower not in seen:
                seen.add(con_lower)
                unique_constraints.append(con)
        
        return unique_constraints
    
    def _extract_entities(self, prompt: str) -> List[str]:
        """
        Extract named entities and important nouns from the prompt.
        
        Uses basic heuristics - capitalized words that aren't at sentence starts.
        """
        entities = []
        
        # Find capitalized words/phrases not at sentence starts
        words = prompt.split()
        for i, word in enumerate(words):
            # Skip first word of sentences
            if i == 0:
                continue
            prev_word = words[i - 1] if i > 0 else ""
            if prev_word.endswith(('.', '!', '?', ':')):
                continue
            
            # Check if word is capitalized
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word and clean_word[0].isupper() and not clean_word.isupper():
                entities.append(clean_word)
        
        # Also extract quoted strings as potential entities
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', prompt)
        for match in quoted:
            entity = match[0] or match[1]
            if entity:
                entities.append(entity)
        
        return list(set(entities))
    
    def _extract_action_verbs(self, prompt: str) -> List[str]:
        """Extract action verbs present in the prompt."""
        prompt_lower = prompt.lower()
        found_verbs = []
        
        for verb in self.ACTION_VERBS:
            if re.search(rf'\b{verb}\b', prompt_lower):
                found_verbs.append(verb)
        
        return found_verbs
    
    def get_embedding(self, text: str):
        """
        Get the semantic embedding for a text.
        
        Args:
            text: The text to embed.
            
        Returns:
            numpy array of embeddings, or None if embeddings not enabled.
        """
        if not self.use_embeddings or self._embedding_model is None:
            return None
        
        return self._embedding_model.encode(text)
