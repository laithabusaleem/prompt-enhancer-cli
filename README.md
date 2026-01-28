# Prompt Enhancer CLI

A Python CLI tool that parses, classifies, and enhances prompts for AI models. Extracts intent, context, and constraints from prompts, classifies them by type, and generates structured, enhanced versions for better AI responses.

## Purpose

This tool helps users:
- **Parse prompts** to extract key components (intent, context, constraints)
- **Classify prompts** into categories (question, instruction, code, creative, etc.)
- **Enhance prompts** with structure and clarity for improved AI responses

## Architecture

```
prompt-enhancer-cli/
├── prompt_enhancer/
│   ├── __init__.py      # Package exports
│   ├── cli.py           # Click-based CLI interface
│   ├── parser.py        # Prompt parsing (intent/context/constraints)
│   ├── classifier.py    # Prompt type classification
│   └── enhancer.py      # Prompt enhancement logic
├── tests/
│   ├── test_parser.py
│   ├── test_classifier.py
│   └── test_enhancer.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

### Pipeline

```
Input Prompt → Parser → Classifier → Enhancer → Enhanced Prompt
                ↓            ↓            ↓
            Intent      Type Label    Structured
            Context     Confidence    Output
            Constraints  Scores
```

## Installation

### Prerequisites
- Python 3.9+
- pip or Poetry

### Install with pip

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-enhancer-cli.git
cd prompt-enhancer-cli

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Install with Poetry

```bash
# Clone the repository
git clone https://github.com/yourusername/prompt-enhancer-cli.git
cd prompt-enhancer-cli

# Install with Poetry
poetry install

# Activate the environment
poetry shell
```

## Usage

### Enhance a Prompt

```bash
# Basic enhancement
prompt-enhancer enhance "Write a Python function to sort a list"

# With embeddings for better analysis (slower, more accurate)
prompt-enhancer enhance "Write a Python function to sort a list" --embeddings

# Output as JSON
prompt-enhancer enhance "Write a Python function" -j
```

### Parse a Prompt

```bash
# Extract intent, context, constraints
prompt-enhancer parse "Write a Python function for sorting in 50 lines or less"

# JSON output
prompt-enhancer parse "Create a summary of the article about AI" -j
```

### Classify a Prompt

```bash
# Classify prompt type
prompt-enhancer classify "What is the capital of France?"

# Show all scores
prompt-enhancer classify "What is the capital of France?" --all-scores
```

### Full Analysis

```bash
# Parse + classify combined
prompt-enhancer analyze "Summarize this article about climate change"
```

### Batch Processing

```bash
# Process multiple prompts from file
prompt-enhancer batch prompts.txt enhanced.json
```

## Example Input/Output

### Input
```
Write a Python function to sort a list of numbers in ascending order. 
The function must handle empty lists and should use no external libraries.
```

### Output (Enhanced)
```
Programming Task: Write a Python function to sort a list of numbers in ascending order

Technical Context:
- Python function
- sorting in ascending order

Technical Requirements:
- must handle empty lists
- should use no external libraries

Provide clean, well-documented code with explanations.
```

### Classification Output
```
Primary Type: code
Confidence: 95%
Description: A request related to programming or code
```

## Prompt Types

| Type | Description |
|------|-------------|
| `instruction` | Request to create, write, or generate content |
| `question` | Question seeking information or clarification |
| `creative` | Request for creative or fictional content |
| `analysis` | Request to analyze, evaluate, or compare |
| `code` | Request related to programming or code |
| `translation` | Request to translate between languages |
| `summarization` | Request to summarize or condense content |
| `extraction` | Request to extract specific information |
| `classification` | Request to classify or categorize |
| `conversation` | Casual conversation or chat |

## Limitations

- **Rule-based parsing**: Uses regex patterns rather than ML models; may miss complex implicit constraints
- **No LLM integration**: Enhancement is template-based, not generative
- **English only**: Optimized for English prompts
- **Embedding model size**: Using sentence-transformers adds ~90MB model download on first use with `--embeddings`
- **No learning**: Does not improve from user feedback

## Development

### Run Tests

```bash
# With pytest
pytest tests/ -v

# With coverage
pytest tests/ --cov=prompt_enhancer
```

### Code Formatting

```bash
# Format with black
black prompt_enhancer/

# Lint with ruff
ruff check prompt_enhancer/
```

## License

MIT License - See LICENSE file for details.

## Credits

Based on architectural patterns from [vaibkumr/prompt-optimizer](https://github.com/vaibkumr/prompt-optimizer).