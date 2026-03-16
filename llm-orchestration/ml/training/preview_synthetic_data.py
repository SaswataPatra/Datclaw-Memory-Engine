"""
Generate and Preview Synthetic Training Data

Step 1: Generate synthetic data and save to files
Step 2: Preview the saved data
Step 3: Train models using the saved data (separate command)

This approach:
- Generates data once (saves API costs)
- Lets you inspect before training
- Reusable for multiple training runs
"""

import asyncio
import json
import os
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()


async def preview_distilbert_data(num_variations=3):
    """Generate and preview DistilBERT training data"""
    from ml.training.distilbert_data_generator import DistilBERTDataGenerator
    
    console.print("\n[bold cyan]📊 Generating DistilBERT Synthetic Data...[/bold cyan]\n")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]")
        console.print("Set it with: export OPENAI_API_KEY='sk-...'")
        return
    
    generator = DistilBERTDataGenerator(openai_api_key=api_key)
    
    # Show seed examples first
    console.print("[bold yellow]🌱 Seed Examples (Human-Labeled):[/bold yellow]\n")
    seed_examples = generator.get_seed_examples()
    
    # Group by label
    by_label = {}
    for text, label in seed_examples:
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(text)
    
    for label, texts in sorted(by_label.items()):
        console.print(f"[bold green]{label.upper()}[/bold green] ({len(texts)} examples):")
        for i, text in enumerate(texts[:2], 1):  # Show first 2
            console.print(f"  {i}. {text}")
        if len(texts) > 2:
            console.print(f"  ... and {len(texts) - 2} more")
        console.print()
    
    # Generate synthetic variations for a few examples
    console.print(f"\n[bold yellow]🤖 Generating {num_variations} Synthetic Variations...[/bold yellow]\n")
    
    sample_seeds = [
        ("My name is Sarah Johnson", "identity"),
        ("I love playing basketball", "preference"),
        ("My mother is a doctor", "family")
    ]
    
    for seed_text, label in sample_seeds:
        console.print(f"[bold cyan]Original ({label}):[/bold cyan] {seed_text}")
        
        try:
            variations = await generator.generate_variations(
                seed_text, 
                label, 
                num_variations=num_variations
            )
            
            console.print(f"[bold green]✅ Generated {len(variations)} variations:[/bold green]")
            for i, (var_text, var_label) in enumerate(variations, 1):
                console.print(f"  {i}. {var_text}")
            console.print()
            
        except Exception as e:
            console.print(f"[bold red]❌ Error: {e}[/bold red]\n")
    
    # Show statistics
    total_labels = len(by_label)
    total_seeds = len(seed_examples)
    estimated_total = total_seeds * (num_variations + 1)  # +1 for seed itself
    
    console.print(Panel.fit(
        f"[bold]Dataset Statistics:[/bold]\n\n"
        f"Labels: {total_labels}\n"
        f"Seed Examples: {total_seeds}\n"
        f"Variations per Seed: {num_variations}\n"
        f"Estimated Total: {estimated_total} examples",
        title="📊 DistilBERT Dataset",
        border_style="cyan"
    ))


async def preview_lightgbm_data(num_variations=3):
    """Generate and preview LightGBM training data"""
    from ml.training.lightgbm_data_generator import LightGBMDataGenerator
    
    console.print("\n[bold cyan]📊 Generating LightGBM Synthetic Data...[/bold cyan]\n")
    
    # Check for API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]")
        return
    
    generator = LightGBMDataGenerator(openai_api_key=api_key)
    
    # Show seed examples
    console.print("[bold yellow]🌱 Seed Examples (Human-Labeled with Ego Scores):[/bold yellow]\n")
    seed_examples = generator.get_seed_examples()
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Text", style="cyan", width=50)
    table.add_column("Label", style="green", width=15)
    table.add_column("Ego Score", style="yellow", justify="center", width=10)
    table.add_column("Tier", style="blue", justify="center", width=8)
    
    for text, label, ego_score, reasoning in seed_examples[:10]:  # Show first 10
        # Determine tier
        if ego_score >= 0.75:
            tier = "Tier 1"
        elif ego_score >= 0.50:
            tier = "Tier 2"
        elif ego_score >= 0.20:
            tier = "Tier 3"
        else:
            tier = "Tier 4"
        
        table.add_row(
            text[:47] + "..." if len(text) > 50 else text,
            label,
            f"{ego_score:.2f}",
            tier
        )
    
    console.print(table)
    console.print(f"\n[dim]... and {len(seed_examples) - 10} more seed examples[/dim]\n")
    
    # Generate synthetic variations for one example
    console.print(f"\n[bold yellow]🤖 Generating {num_variations} Synthetic Variations...[/bold yellow]\n")
    
    sample_seed = seed_examples[0]  # Take first seed
    seed_text, label, target_score, reasoning = sample_seed
    
    console.print(f"[bold cyan]Original ({label}, score={target_score}):[/bold cyan]")
    console.print(f"  {seed_text}")
    console.print(f"[dim]  Reasoning: {reasoning}[/dim]\n")
    
    try:
        variations = await generator.generate_variations(
            seed_text,
            label,
            target_score,
            num_variations=num_variations
        )
        
        console.print(f"[bold green]✅ Generated {len(variations)} variations:[/bold green]")
        for i, (var_text, var_label, var_score) in enumerate(variations, 1):
            console.print(f"  {i}. [cyan]{var_text}[/cyan] (score: {var_score:.2f})")
        console.print()
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {e}[/bold red]\n")
    
    # Show distribution
    ego_dist = generator.get_ego_score_distribution(seed_examples)
    label_dist = generator.get_label_distribution(seed_examples)
    
    # Ego score distribution
    console.print("[bold yellow]📊 Ego Score Distribution:[/bold yellow]")
    for tier, count in ego_dist.items():
        console.print(f"  {tier}: {count} examples")
    
    console.print("\n[bold yellow]📊 Label Distribution:[/bold yellow]")
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        console.print(f"  {label}: {count} examples")
    
    # Statistics
    total_seeds = len(seed_examples)
    estimated_total = total_seeds * (num_variations + 1)
    
    console.print(Panel.fit(
        f"[bold]Dataset Statistics:[/bold]\n\n"
        f"Seed Examples: {total_seeds}\n"
        f"Variations per Seed: {num_variations}\n"
        f"Estimated Total: {estimated_total} examples\n"
        f"Tiers Covered: {len(ego_dist)}\n"
        f"Labels Covered: {len(label_dist)}",
        title="📊 LightGBM Dataset",
        border_style="cyan"
    ))


async def main():
    """Main preview function"""
    console.print(Panel.fit(
        "[bold cyan]Synthetic Data Preview[/bold cyan]\n\n"
        "This will generate a small sample of synthetic training data\n"
        "so you can inspect the quality before full training.\n\n"
        "[dim]Note: This uses OpenAI API (minimal cost: ~$0.10)[/dim]",
        title="🔍 Preview Mode",
        border_style="green"
    ))
    
    # Check API key
    if not os.getenv('OPENAI_API_KEY'):
        console.print("\n[bold red]❌ Error: OPENAI_API_KEY environment variable not set[/bold red]")
        console.print("\nSet it with:")
        console.print("  export OPENAI_API_KEY='sk-...'")
        return
    
    # Preview DistilBERT data
    await preview_distilbert_data(num_variations=3)
    
    # Preview LightGBM data
    await preview_lightgbm_data(num_variations=3)
    
    # Summary
    console.print("\n" + "="*70 + "\n")
    console.print("[bold green]✅ Preview Complete![/bold green]\n")
    console.print("If the synthetic data looks good, proceed with full training:")
    console.print("\n[bold cyan]DistilBERT:[/bold cyan]")
    console.print("  python -m ml.training.train_distilbert --generate-data --train --variations-per-seed 5")
    console.print("\n[bold cyan]LightGBM:[/bold cyan]")
    console.print("  python -m ml.training.train_lightgbm --generate-data --train --variations-per-seed 10")
    console.print()


if __name__ == "__main__":
    asyncio.run(main())

