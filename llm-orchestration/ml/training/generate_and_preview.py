"""
Generate and Preview Synthetic Training Data

Workflow:
1. Generate synthetic data and save to JSON files (one-time cost)
2. Preview the saved data (inspect quality)
3. Train models using saved data (separate command)

This saves API costs and allows inspection before training.
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
from rich.progress import Progress, SpinnerColumn, TextColumn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

console = Console()


async def generate_distilbert_data(output_path: str, variations_per_seed: int = 5):
    """Generate DistilBERT training data and save to file"""
    from ml.training.distilbert_data_generator import DistilBERTDataGenerator
    
    console.print(f"\n[bold cyan]📊 Generating DistilBERT Dataset...[/bold cyan]")
    console.print(f"Output: {output_path}")
    console.print(f"Variations per seed: {variations_per_seed}")
    
    # Calculate expected API calls
    num_seeds = 80  # Approximate
    expected_calls = num_seeds * variations_per_seed
    console.print(f"[yellow]⏳ Expected API calls: ~{expected_calls}[/yellow]")
    console.print(f"[yellow]⏳ Estimated time: ~5-10 minutes[/yellow]")
    console.print(f"[dim]Tip: This is slow because API calls are sequential. Grab a coffee! ☕[/dim]\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]")
        return False
    
    generator = DistilBERTDataGenerator(openai_api_key=api_key)
    
    # Generate full dataset
    console.print("[cyan]Starting generation...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating variations (be patient, this takes time)...", total=None)
        
        dataset = await generator.generate_full_dataset(
            variations_per_seed=variations_per_seed
        )
        
        progress.update(task, completed=True)
    
    # Save to file
    generator.save_dataset(dataset, output_path)
    
    # Show statistics
    label_dist = generator.get_label_distribution(dataset)
    
    console.print(f"\n[bold green]✅ Dataset generated and saved![/bold green]")
    console.print(f"Total examples: {len(dataset)}")
    console.print(f"Labels: {len(label_dist)}")
    console.print(f"File: {output_path}")
    
    return True


async def generate_lightgbm_data(output_path: str, variations_per_seed: int = 10):
    """Generate LightGBM training data and save to file"""
    from ml.training.lightgbm_data_generator import LightGBMDataGenerator
    
    console.print(f"\n[bold cyan]📊 Generating LightGBM Dataset...[/bold cyan]")
    console.print(f"Output: {output_path}")
    console.print(f"Variations per seed: {variations_per_seed}\n")
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]")
        return False
    
    generator = LightGBMDataGenerator(openai_api_key=api_key)
    
    # Generate full dataset
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Generating synthetic variations...", total=None)
        
        dataset = await generator.generate_full_dataset(
            variations_per_seed=variations_per_seed
        )
        
        progress.update(task, completed=True)
    
    # Save to file
    generator.save_dataset(dataset, output_path)
    
    # Show statistics
    ego_dist = generator.get_ego_score_distribution(dataset)
    label_dist = generator.get_label_distribution(dataset)
    
    console.print(f"\n[bold green]✅ Dataset generated and saved![/bold green]")
    console.print(f"Total examples: {len(dataset)}")
    console.print(f"Tiers: {len(ego_dist)}")
    console.print(f"Labels: {len(label_dist)}")
    console.print(f"File: {output_path}")
    
    return True


def preview_distilbert_file(file_path: str):
    """Preview saved DistilBERT dataset"""
    console.print(f"\n[bold cyan]🔍 Previewing DistilBERT Dataset[/bold cyan]")
    console.print(f"File: {file_path}\n")
    
    if not os.path.exists(file_path):
        console.print(f"[bold red]❌ File not found: {file_path}[/bold red]")
        console.print("Generate it first with: --generate distilbert")
        return
    
    # Load dataset
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Group by label
    by_label = {}
    for item in dataset:
        labels = item['labels']  # Note: 'labels' is a list
        for label in labels:
            if label not in by_label:
                by_label[label] = []
            by_label[label].append(item['text'])
    
    # Show statistics
    console.print(f"[bold]Total Examples:[/bold] {len(dataset)}")
    console.print(f"[bold]Labels:[/bold] {len(by_label)}\n")
    
    # Show samples per label
    console.print("[bold yellow]📊 Label Distribution & Samples:[/bold yellow]\n")
    
    for label in sorted(by_label.keys()):
        examples = by_label[label]
        console.print(f"[bold green]{label.upper()}[/bold green] ({len(examples)} examples):")
        
        # Show first 3 examples
        for i, text in enumerate(examples[:3], 1):
            console.print(f"  {i}. {text}")
        
        if len(examples) > 3:
            console.print(f"  [dim]... and {len(examples) - 3} more[/dim]")
        console.print()
    
    # Summary panel
    console.print(Panel.fit(
        f"[bold]Dataset Ready for Training[/bold]\n\n"
        f"Total Examples: {len(dataset)}\n"
        f"Labels: {', '.join(sorted(by_label.keys()))}\n"
        f"File: {file_path}\n\n"
        f"[bold cyan]Train with:[/bold cyan]\n"
        f"python -m ml.training.train_distilbert --dataset-path {file_path} --train",
        title="✅ DistilBERT Dataset",
        border_style="green"
    ))


def preview_lightgbm_file(file_path: str):
    """Preview saved LightGBM dataset"""
    console.print(f"\n[bold cyan]🔍 Previewing LightGBM Dataset[/bold cyan]")
    console.print(f"File: {file_path}\n")
    
    if not os.path.exists(file_path):
        console.print(f"[bold red]❌ File not found: {file_path}[/bold red]")
        console.print("Generate it first with: --generate lightgbm")
        return
    
    # Load dataset
    with open(file_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    
    # Calculate distributions
    ego_dist = {}
    label_dist = {}
    
    for item in dataset:
        ego_score = item.get('target_ego_score', item.get('ego_score', 0.5))  # Handle both formats
        label = item['label']
        
        # Tier distribution
        if ego_score >= 0.75:
            tier = "Tier 1 (>= 0.75)"
        elif ego_score >= 0.50:
            tier = "Tier 2 (0.50-0.75)"
        elif ego_score >= 0.20:
            tier = "Tier 3 (0.20-0.50)"
        else:
            tier = "Tier 4 (< 0.20)"
        
        ego_dist[tier] = ego_dist.get(tier, 0) + 1
        label_dist[label] = label_dist.get(label, 0) + 1
    
    # Show statistics
    console.print(f"[bold]Total Examples:[/bold] {len(dataset)}")
    console.print(f"[bold]Tiers:[/bold] {len(ego_dist)}")
    console.print(f"[bold]Labels:[/bold] {len(label_dist)}\n")
    
    # Show sample examples in table
    console.print("[bold yellow]📊 Sample Examples:[/bold yellow]\n")
    
    table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
    table.add_column("Text", style="cyan", width=50)
    table.add_column("Label", style="green", width=12)
    table.add_column("Ego Score", style="yellow", justify="center", width=10)
    table.add_column("Tier", style="blue", justify="center", width=8)
    
    for item in dataset[:15]:  # Show first 15
        text = item['text']
        label = item['label']
        ego_score = item.get('target_ego_score', item.get('ego_score', 0.5))
        
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
    console.print(f"\n[dim]... and {len(dataset) - 15} more examples[/dim]\n")
    
    # Show distributions
    console.print("[bold yellow]📊 Tier Distribution:[/bold yellow]")
    for tier in ["Tier 1 (>= 0.75)", "Tier 2 (0.50-0.75)", "Tier 3 (0.20-0.50)", "Tier 4 (< 0.20)"]:
        count = ego_dist.get(tier, 0)
        console.print(f"  {tier}: {count} examples")
    
    console.print("\n[bold yellow]📊 Top Labels:[/bold yellow]")
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
        console.print(f"  {label}: {count} examples")
    
    # Summary panel
    console.print("\n")
    console.print(Panel.fit(
        f"[bold]Dataset Ready for Training[/bold]\n\n"
        f"Total Examples: {len(dataset)}\n"
        f"Tiers Covered: {len(ego_dist)}\n"
        f"Labels Covered: {len(label_dist)}\n"
        f"File: {file_path}\n\n"
        f"[bold cyan]Train with:[/bold cyan]\n"
        f"python -m ml.training.train_lightgbm --dataset-path {file_path} --train",
        title="✅ LightGBM Dataset",
        border_style="green"
    ))


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate and preview synthetic training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate DistilBERT data
  python generate_and_preview.py --generate distilbert --variations 5
  
  # Generate LightGBM data
  python generate_and_preview.py --generate lightgbm --variations 10
  
  # Generate both
  python generate_and_preview.py --generate both --variations 5
  
  # Preview saved data
  python generate_and_preview.py --preview distilbert
  python generate_and_preview.py --preview lightgbm
  python generate_and_preview.py --preview both
        """
    )
    
    parser.add_argument(
        '--generate',
        choices=['distilbert', 'lightgbm', 'both'],
        help='Generate synthetic data for specified model(s)'
    )
    
    parser.add_argument(
        '--preview',
        choices=['distilbert', 'lightgbm', 'both'],
        help='Preview saved dataset for specified model(s)'
    )
    
    parser.add_argument(
        '--variations',
        type=int,
        default=5,
        help='Number of variations per seed example (default: 5 for DistilBERT, 10 for LightGBM)'
    )
    
    parser.add_argument(
        '--distilbert-output',
        default='data/distilbert_dataset.jsonl',
        help='Output path for DistilBERT dataset (default: data/distilbert_dataset.jsonl)'
    )
    
    parser.add_argument(
        '--lightgbm-output',
        default='data/lightgbm_dataset.jsonl',
        help='Output path for LightGBM dataset (default: data/lightgbm_dataset.jsonl)'
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    Path('data').mkdir(exist_ok=True)
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]Synthetic Data Generator & Preview[/bold cyan]\n\n"
        "Step 1: Generate data (uses OpenAI API)\n"
        "Step 2: Preview data (inspect quality)\n"
        "Step 3: Train models (separate command)\n\n"
        "[dim]This workflow saves API costs and allows inspection before training[/dim]",
        title="🔬 Data Generation Workflow",
        border_style="green"
    ))
    
    # Generate mode
    if args.generate:
        if not os.getenv('OPENAI_API_KEY'):
            console.print("\n[bold red]❌ Error: OPENAI_API_KEY not set[/bold red]")
            console.print("Set it with: export OPENAI_API_KEY='sk-...'")
            return
        
        if args.generate in ['distilbert', 'both']:
            success = await generate_distilbert_data(
                args.distilbert_output,
                variations_per_seed=args.variations
            )
            if not success:
                return
        
        if args.generate in ['lightgbm', 'both']:
            # Use 2x variations for LightGBM (more data needed)
            lightgbm_variations = args.variations * 2 if args.generate == 'both' else args.variations
            success = await generate_lightgbm_data(
                args.lightgbm_output,
                variations_per_seed=lightgbm_variations
            )
            if not success:
                return
        
        console.print("\n[bold green]✅ Generation complete![/bold green]")
        console.print("\nNext step: Preview the data")
        console.print(f"  python {__file__} --preview {args.generate}")
    
    # Preview mode
    elif args.preview:
        if args.preview in ['distilbert', 'both']:
            preview_distilbert_file(args.distilbert_output)
        
        if args.preview in ['lightgbm', 'both']:
            preview_lightgbm_file(args.lightgbm_output)
        
        console.print("\n[bold green]✅ Preview complete![/bold green]")
        console.print("\nIf the data looks good, proceed with training:")
        
        if args.preview in ['distilbert', 'both']:
            console.print(f"\n[bold cyan]DistilBERT:[/bold cyan]")
            console.print(f"  python -m ml.training.train_distilbert --dataset-path {args.distilbert_output} --train")
        
        if args.preview in ['lightgbm', 'both']:
            console.print(f"\n[bold cyan]LightGBM:[/bold cyan]")
            console.print(f"  python -m ml.training.train_lightgbm --dataset-path {args.lightgbm_output} --train")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())

