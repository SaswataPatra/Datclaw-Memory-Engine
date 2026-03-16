"""
Lightweight test for Zero-Shot Classifier using smaller model

Uses facebook/bart-base instead of bart-large to reduce memory usage.
"""

import time
from rich.console import Console
from rich.panel import Panel

console = Console()

# Test with smaller model
TEST_CASES = [
    "My name is Sarah and I'm 28 years old",
    "I love pizza but hate mushrooms",
    "I work as a software engineer at Google",
    "I want to learn Spanish by next year",
    "My best friend John is getting married",
]

def test_small_model():
    """Test with smaller BART model"""
    
    console.print("\n[bold cyan]🚀 Testing Zero-Shot Classifier (Lite)[/bold cyan]\n")
    console.print("[yellow]Using smaller model: facebook/bart-base[/yellow]\n")
    
    from ml.extractors.zeroshot_memory_classifier import ZeroShotMemoryClassifier
    
    # Initialize with smaller model
    console.print("[yellow]Loading classifier...[/yellow]")
    start_time = time.time()
    
    classifier = ZeroShotMemoryClassifier(
        model_name="facebook/bart-base",  # Smaller model
        low_confidence_threshold=0.3
    )
    
    load_time = time.time() - start_time
    console.print(f"[green]✅ Loaded in {load_time:.2f}s[/green]\n")
    
    # Test predictions
    total_time = 0
    
    for i, text in enumerate(TEST_CASES, 1):
        console.print(f"[cyan]Test {i}/{len(TEST_CASES)}:[/cyan] {text}")
        
        start_time = time.time()
        labels, scores, needs_discovery = classifier.predict_single(text, threshold=0.5)
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Get top 3
        top_3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        console.print(f"  [green]Labels:[/green] {', '.join(labels)}")
        console.print(f"  [blue]Top 3:[/blue] {', '.join([f'{l}({s:.2f})' for l, s in top_3])}")
        console.print(f"  [yellow]Time:[/yellow] {inference_time:.2f}s")
        if needs_discovery:
            console.print(f"  [red]⚠️  Low confidence[/red]")
        console.print()
    
    avg_time = total_time / len(TEST_CASES)
    
    console.print(Panel.fit(
        f"[bold green]✅ Test Complete![/bold green]\n\n"
        f"[cyan]Results:[/cyan]\n"
        f"  • Total time: {total_time:.2f}s\n"
        f"  • Avg time: {avg_time:.2f}s per prediction\n"
        f"  • Model: facebook/bart-base (smaller, faster)\n\n"
        f"[yellow]Note:[/yellow] bart-large is too big for your Mac's RAM.\n"
        f"Use bart-base for local testing or deploy to cloud with more RAM.",
        border_style="green"
    ))

if __name__ == "__main__":
    try:
        test_small_model()
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        import traceback
        traceback.print_exc()

