"""
Prompt Enhancer CLI - Command-line interface using Click.
"""

import json
import sys
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from prompt_enhancer import PromptEnhancer, PromptParser, PromptClassifier, __version__


console = Console()


def print_error(message: str):
    """Print error message in red."""
    console.print(f"[bold red]Error:[/bold red] {message}")


def print_success(message: str):
    """Print success message in green."""
    console.print(f"[bold green]✓[/bold green] {message}")


@click.group()
@click.version_option(version=__version__, prog_name="prompt-enhancer")
def main():
    """
    Prompt Enhancer CLI - Parse, classify, and enhance prompts for AI models.
    
    This tool analyzes prompts to extract intent, context, and constraints,
    classifies them by type, and generates enhanced versions for better AI responses.
    """
    pass


@main.command()
@click.argument("prompt", type=str)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use sentence embeddings for semantic analysis (slower but more accurate)"
)
@click.option(
    "--structured/--inline",
    default=True,
    help="Output structured format or inline enhancement"
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON"
)
@click.option(
    "--llm", "-l",
    type=click.Choice(['openai', 'mistral', 'google', 'auto'], case_sensitive=False),
    help="Use LLM for enhancement (requires API key)"
)
@click.option(
    "--config", "-c",
    type=click.Path(exists=True),
    help="Path to custom template configuration file"
)
def enhance(prompt: str, embeddings: bool, structured: bool, json_output: bool, llm: Optional[str], config: Optional[str]):
    """
    Enhance a prompt with structure and clarity.
    
    PROMPT: The prompt text to enhance (use quotes for multi-word prompts)
    
    Example:
        prompt-enhancer enhance "Write a function" --llm openai
    """
    try:
        enhancer = PromptEnhancer(use_embeddings=embeddings, llm_provider=llm, config_path=config)
        result = enhancer.enhance(prompt, add_structure=structured)
        
        if json_output:
            console.print(json.dumps(result.to_dict(), indent=2))
        else:
            # Display original
            console.print(Panel(
                prompt,
                title="[bold blue]Original Prompt[/bold blue]",
                border_style="blue"
            ))
            
            # Display classification
            console.print(f"\n[bold]Type:[/bold] {result.classification.primary_type.value} "
                         f"(confidence: {result.classification.confidence:.2f})")
            
            # Display enhanced
            console.print(Panel(
                result.enhanced,
                title="[bold green]Enhanced Prompt[/bold green]",
                border_style="green"
            ))
            
            # Display notes
            if result.enhancement_notes:
                console.print("\n[bold]Enhancement Notes:[/bold]")
                for note in result.enhancement_notes:
                    console.print(f"  • {note}")
                    
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


@main.command()
def init_config():
    """
    Initialize default configuration file.
    
    Creates a 'templates.yaml' file in ~/.prompt-enhancer/
    which you can edit to customize enhancement templates.
    """
    try:
        from prompt_enhancer.config import TemplateManager
        manager = TemplateManager()
        path = manager.save_default_config()
        print_success(f"Configuration initialized at: {path}")
        console.print("You can now edit this file to customize your templates.")
    except Exception as e:
        print_error(str(e))
        sys.exit(1)



@main.command()
@click.argument("prompt", type=str)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use sentence embeddings for semantic analysis"
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON"
)
def parse(prompt: str, embeddings: bool, json_output: bool):
    """
    Parse a prompt to extract intent, context, and constraints.
    
    PROMPT: The prompt text to parse
    
    Example:
        prompt-enhancer parse "Write a Python function for sorting in 50 lines or less"
    """
    try:
        parser = PromptParser(use_embeddings=embeddings)
        result = parser.parse(prompt)
        
        if json_output:
            console.print(json.dumps(result.to_dict(), indent=2))
        else:
            console.print(Panel(prompt, title="[bold blue]Input Prompt[/bold blue]"))
            
            table = Table(title="Parsed Components", show_header=True)
            table.add_column("Component", style="cyan", width=15)
            table.add_column("Value", style="white")
            
            table.add_row("Intent", result.intent)
            table.add_row("Context", ", ".join(result.context) if result.context else "(none detected)")
            table.add_row("Constraints", ", ".join(result.constraints) if result.constraints else "(none detected)")
            table.add_row("Entities", ", ".join(result.entities) if result.entities else "(none detected)")
            table.add_row("Action Verbs", ", ".join(result.action_verbs) if result.action_verbs else "(none detected)")
            
            console.print(table)
            
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


@main.command()
@click.argument("prompt", type=str)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use sentence embeddings for classification"
)
@click.option(
    "--json-output", "-j",
    is_flag=True,
    help="Output results as JSON"
)
@click.option(
    "--all-scores", "-a",
    is_flag=True,
    help="Show scores for all prompt types"
)
def classify(prompt: str, embeddings: bool, json_output: bool, all_scores: bool):
    """
    Classify a prompt by type (question, instruction, creative, etc.).
    
    PROMPT: The prompt text to classify
    
    Example:
        prompt-enhancer classify "What is the capital of France?"
    """
    try:
        classifier = PromptClassifier(use_embeddings=embeddings)
        result = classifier.classify(prompt)
        
        if json_output:
            console.print(json.dumps(result.to_dict(), indent=2))
        else:
            console.print(Panel(prompt, title="[bold blue]Input Prompt[/bold blue]"))
            
            console.print(f"\n[bold]Primary Type:[/bold] [green]{result.primary_type.value}[/green]")
            console.print(f"[bold]Confidence:[/bold] {result.confidence:.2%}")
            console.print(f"[bold]Description:[/bold] {classifier.get_type_description(result.primary_type)}")
            
            if all_scores and result.all_scores:
                console.print("\n[bold]All Type Scores:[/bold]")
                sorted_scores = sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True)
                for ptype, score in sorted_scores:
                    if score > 0:
                        bar_length = int(score * 20)
                        bar = "█" * bar_length + "░" * (20 - bar_length)
                        console.print(f"  {ptype:15} [{bar}] {score:.2%}")
                        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


@main.command()
@click.argument("prompt", type=str)
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use sentence embeddings for analysis"
)
def analyze(prompt: str, embeddings: bool):
    """
    Perform full analysis on a prompt (parse + classify).
    
    PROMPT: The prompt text to analyze
    
    Example:
        prompt-enhancer analyze "Summarize this article about climate change"
    """
    try:
        enhancer = PromptEnhancer(use_embeddings=embeddings)
        result = enhancer.analyze(prompt)
        
        console.print(Panel(prompt, title="[bold blue]Input Prompt[/bold blue]"))
        
        # Classification
        console.print("\n[bold cyan]═══ Classification ═══[/bold cyan]")
        console.print(f"Type: [green]{result['classification']['primary_type']}[/green]")
        console.print(f"Confidence: {result['classification']['confidence']:.2%}")
        console.print(f"Description: {result['type_description']}")
        
        # Parsed components
        console.print("\n[bold cyan]═══ Parsed Components ═══[/bold cyan]")
        parsed = result['parsed']
        console.print(f"[bold]Intent:[/bold] {parsed['intent']}")
        
        if parsed['context']:
            console.print("[bold]Context:[/bold]")
            for ctx in parsed['context']:
                console.print(f"  • {ctx}")
        
        if parsed['constraints']:
            console.print("[bold]Constraints:[/bold]")
            for con in parsed['constraints']:
                console.print(f"  • {con}")
        
        if parsed['action_verbs']:
            console.print(f"[bold]Action Verbs:[/bold] {', '.join(parsed['action_verbs'])}")
        
        if parsed['entities']:
            console.print(f"[bold]Entities:[/bold] {', '.join(parsed['entities'])}")
            
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option(
    "--embeddings/--no-embeddings",
    default=False,
    help="Use sentence embeddings"
)
def batch(input_file: str, output_file: str, embeddings: bool):
    """
    Process multiple prompts from a file (one per line).
    
    INPUT_FILE: File containing prompts (one per line)
    OUTPUT_FILE: File to write enhanced prompts (JSON)
    
    Example:
        prompt-enhancer batch prompts.txt enhanced.json
    """
    try:
        enhancer = PromptEnhancer(use_embeddings=embeddings)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        console.print(f"Processing {len(prompts)} prompts...")
        
        results = []
        with console.status("[bold green]Enhancing prompts...") as status:
            for i, prompt in enumerate(prompts):
                status.update(f"[bold green]Processing prompt {i+1}/{len(prompts)}...")
                result = enhancer.enhance(prompt)
                results.append(result.to_dict())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        print_success(f"Processed {len(prompts)} prompts → {output_file}")
        
    except Exception as e:
        print_error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
