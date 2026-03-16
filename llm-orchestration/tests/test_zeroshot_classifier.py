"""
Standalone test script for Zero-Shot Memory Classifier

Tests the new zero-shot classifier and compares it with DistilBERT.
Run this to evaluate performance before integration.

Usage:
    python test_zeroshot_classifier.py
"""

import asyncio
import time
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ml.extractors.zeroshot_memory_classifier import (
    ZeroShotMemoryClassifier,
    DynamicLabelDiscovery,
    LabelStore
)

console = Console()


# Test cases covering different memory types
TEST_CASES = [
    # Identity
    "My name is Sarah and I'm 28 years old",
    "I use she/her pronouns",
    
    # Family
    "My mom is a doctor and my dad is a teacher",
    "I have two younger siblings",
    
    # Preference
    "I love pizza but hate mushrooms",
    "My favorite color is blue",
    
    # Fact
    "I work as a software engineer at Google",
    "I live in San Francisco",
    
    # Goal
    "I want to learn Spanish by next year",
    "My goal is to run a marathon",
    
    # Relationship
    "My best friend John is getting married",
    "I'm dating someone from work",
    
    # Event
    "I went to Paris last summer",
    "Yesterday I had a job interview",
    
    # Opinion
    "I think climate change is the biggest issue",
    "I believe in universal healthcare",
    
    # High value
    "My salary is $150k per year",
    "My bank account number is 123456",
    
    # Edge cases (should trigger label discovery)
    "I have type 2 diabetes and take metformin daily",
    "I'm planning a trip to Japan in March",
    "I'm learning to play the guitar",
    "I'm allergic to peanuts and shellfish",
]


async def test_zeroshot_classifier():
    """Test zero-shot classifier on various inputs"""
    
    console.print("\n[bold cyan]🚀 Testing Zero-Shot Memory Classifier[/bold cyan]\n")
    
    # Initialize classifier
    console.print("[yellow]Loading zero-shot classifier...[/yellow]")
    start_time = time.time()
    
    classifier = ZeroShotMemoryClassifier(
        low_confidence_threshold=0.3
    )
    
    load_time = time.time() - start_time
    console.print(f"[green]✅ Loaded in {load_time:.2f}s[/green]\n")
    
    # Test predictions
    results = []
    total_inference_time = 0
    
    for i, text in enumerate(TEST_CASES, 1):
        console.print(f"[cyan]Test {i}/{len(TEST_CASES)}:[/cyan] {text[:60]}...")
        
        start_time = time.time()
        predicted_labels, scores, needs_discovery = classifier.predict_single(
            text,
            threshold=0.5
        )
        inference_time = time.time() - start_time
        total_inference_time += inference_time
        
        # Get top 3 scores
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        results.append({
            'text': text,
            'labels': predicted_labels,
            'top_3': top_3,
            'needs_discovery': needs_discovery,
            'time': inference_time
        })
        
        # Print results
        console.print(f"  [green]Labels:[/green] {', '.join(predicted_labels)}")
        console.print(f"  [blue]Top 3:[/blue] {', '.join([f'{l}({s:.2f})' for l, s in top_3])}")
        console.print(f"  [yellow]Time:[/yellow] {inference_time:.2f}s")
        if needs_discovery:
            console.print(f"  [red]⚠️  Low confidence - label discovery recommended[/red]")
        console.print()
    
    # Summary statistics
    avg_time = total_inference_time / len(TEST_CASES)
    needs_discovery_count = sum(1 for r in results if r['needs_discovery'])
    
    summary = Table(title="📊 Performance Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", style="green")
    
    summary.add_row("Total Tests", str(len(TEST_CASES)))
    summary.add_row("Total Time", f"{total_inference_time:.2f}s")
    summary.add_row("Avg Time/Prediction", f"{avg_time:.2f}s")
    summary.add_row("Needs Discovery", f"{needs_discovery_count}/{len(TEST_CASES)}")
    summary.add_row("Current Labels", str(len(classifier.current_labels)))
    
    console.print(summary)
    
    return results, classifier


async def test_label_discovery():
    """Test dynamic label discovery"""
    
    console.print("\n[bold cyan]🔍 Testing Dynamic Label Discovery[/bold cyan]\n")
    
    # Mock LLM client (you'll need to provide real one)
    from openai import AsyncOpenAI
    import os
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        console.print("[red]⚠️  OPENAI_API_KEY not set. Skipping label discovery test.[/red]")
        return
    
    llm_client = AsyncOpenAI(api_key=api_key)
    
    # Initialize discovery
    discovery = DynamicLabelDiscovery(
        llm_client=llm_client,
        config={'llm': {'model': 'gpt-4o-mini'}},
        enabled=True
    )
    
    # Test cases that should trigger discovery
    edge_cases = [
        "I have type 2 diabetes and take metformin daily",
        "I'm planning a trip to Japan in March",
        "I'm learning to play the guitar",
    ]
    
    classifier = ZeroShotMemoryClassifier()
    
    for text in edge_cases:
        console.print(f"[cyan]Testing:[/cyan] {text}")
        
        # Get initial classification
        labels, scores, needs_discovery = classifier.predict_single(text, threshold=0.5)
        
        if needs_discovery:
            console.print(f"  [yellow]Low confidence detected, discovering labels...[/yellow]")
            
            # Discover new labels
            new_labels = await discovery.discover_labels(
                text=text,
                existing_labels=classifier.current_labels,
                current_scores=scores
            )
            
            if new_labels:
                console.print(f"  [green]✅ Discovered:[/green] {', '.join(new_labels)}")
                
                # Add to classifier
                classifier.add_labels(new_labels)
                
                # Re-classify with new labels
                labels, scores, _ = classifier.predict_single(text, threshold=0.5)
                console.print(f"  [blue]Re-classified as:[/blue] {', '.join(labels)}")
            else:
                console.print(f"  [red]❌ No new labels discovered[/red]")
        else:
            console.print(f"  [green]Confident classification:[/green] {', '.join(labels)}")
        
        console.print()


async def test_label_store():
    """Test label persistence"""
    
    console.print("\n[bold cyan]💾 Testing Label Store[/bold cyan]\n")
    
    # Initialize store
    store = LabelStore(storage_path="data/test_discovered_labels.json")
    
    # Add some labels
    test_labels = [
        ("health_info", "I have diabetes", "user123"),
        ("travel_plans", "Going to Japan", "user456"),
        ("hobbies", "Learning guitar", "user789"),
    ]
    
    for label, context, user_id in test_labels:
        store.add_label(label, context, user_id)
        console.print(f"[green]✅ Added:[/green] {label}")
    
    # Increment usage
    store.increment_usage("health_info")
    store.increment_usage("health_info")
    
    # Display all labels
    console.print(f"\n[cyan]Stored Labels:[/cyan]")
    for label in store.get_all_labels():
        metadata = store.get_label_metadata(label)
        console.print(f"  • {label}: used {metadata['usage_count']} times")
    
    console.print()


def compare_with_distilbert():
    """
    Compare zero-shot vs DistilBERT (if available)
    """
    console.print("\n[bold cyan]⚖️  Comparison: Zero-Shot vs DistilBERT[/bold cyan]\n")
    
    try:
        from ml.extractors.memory_classifier import DistilBERTMemoryClassifier
        
        # Try to load DistilBERT
        console.print("[yellow]Loading DistilBERT...[/yellow]")
        start_time = time.time()
        distilbert = DistilBERTMemoryClassifier()
        try:
            distilbert.load_model("models/distilbert/best_model.pt")
            distilbert_available = True
            load_time = time.time() - start_time
            console.print(f"[green]✅ DistilBERT loaded in {load_time:.2f}s[/green]\n")
        except FileNotFoundError:
            console.print("[red]⚠️  DistilBERT model not found. Skipping comparison.[/red]\n")
            distilbert_available = False
    except Exception as e:
        console.print(f"[red]⚠️  Could not load DistilBERT: {e}[/red]\n")
        distilbert_available = False
    
    if not distilbert_available:
        return
    
    # Compare on a few examples
    test_texts = TEST_CASES[:5]
    
    # Load zero-shot
    console.print("[yellow]Loading zero-shot...[/yellow]")
    zeroshot = ZeroShotMemoryClassifier()
    console.print("[green]✅ Zero-shot loaded[/green]\n")
    
    # Create comparison table
    table = Table(title="Comparison Results")
    table.add_column("Text", style="cyan", width=40)
    table.add_column("Zero-Shot", style="green")
    table.add_column("DistilBERT", style="blue")
    table.add_column("Time (ZS)", style="yellow")
    table.add_column("Time (DB)", style="yellow")
    
    for text in test_texts:
        # Zero-shot
        start = time.time()
        zs_labels, _, _ = zeroshot.predict_single(text, threshold=0.5)
        zs_time = time.time() - start
        
        # DistilBERT
        start = time.time()
        db_labels, _ = distilbert.predict_single(text, threshold=0.5)
        db_time = time.time() - start
        
        table.add_row(
            text[:40] + "...",
            ", ".join(zs_labels),
            ", ".join(db_labels),
            f"{zs_time:.2f}s",
            f"{db_time:.2f}s"
        )
    
    console.print(table)
    console.print()


async def main():
    """Run all tests"""
    
    console.print(Panel.fit(
        "[bold cyan]Zero-Shot Memory Classifier Test Suite[/bold cyan]\n"
        "Testing new classifier before integration",
        border_style="cyan"
    ))
    
    # Test 1: Basic classification
    results, classifier = await test_zeroshot_classifier()
    
    # Test 2: Label discovery (if API key available)
    await test_label_discovery()
    
    # Test 3: Label persistence
    await test_label_store()
    
    # Test 4: Comparison with DistilBERT
    compare_with_distilbert()
    
    # Final summary
    console.print(Panel.fit(
        "[bold green]✅ All tests completed![/bold green]\n\n"
        "[cyan]Next steps:[/cyan]\n"
        "1. Review results above\n"
        "2. Adjust thresholds if needed\n"
        "3. Integrate into chatbot_service.py\n"
        "4. Add configuration flag to switch between classifiers",
        border_style="green"
    ))


if __name__ == "__main__":
    asyncio.run(main())

