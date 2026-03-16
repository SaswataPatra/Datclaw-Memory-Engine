"""
Quick test for sentence-level classification with DistilBERT
"""

import asyncio
from ml.extractors.memory_classifier import DistilBERTMemoryClassifier


async def test_sentence_classification():
    """Test that sentence splitting works correctly"""
    
    # Initialize classifier
    classifier = DistilBERTMemoryClassifier(device='cpu')
    
    # Test paragraph with multiple memory types
    test_messages = [
        # Single sentence
        "My name is John",
        
        # Multiple sentences, different types
        "My name is John. I love pizza. My mother is a teacher.",
        
        # Longer paragraph
        "Hi! My name is Sarah Johnson and I'm a software engineer. I love playing basketball on weekends. My father is a doctor and my sister just graduated from MIT. I'm currently working at Google in San Francisco.",
        
        # Edge cases
        "Dr. Smith works at U.S.A. Inc.",  # Should not split on abbreviations
    ]
    
    # Use the new text splitter utility
    from ml.utils import split_for_memory
    
    print("=" * 80)
    print("SENTENCE SPLITTING TEST")
    print("=" * 80)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {message}")
        print(f"{'='*80}")
        
        # Split into sentences using the utility
        sentences = split_for_memory(message, min_words=3, filter_questions=True)
        
        print(f"\n📝 Split into {len(sentences)} sentence(s):")
        for j, sent in enumerate(sentences, 1):
            print(f"  {j}. {sent}")
        
        # Classify each sentence
        print(f"\n🔍 Classification results:")
        all_labels = set()
        
        for j, sentence in enumerate(sentences, 1):
            labels, scores = classifier.predict_single(sentence, threshold=0.5)
            
            if labels:
                print(f"  Sentence {j}: {labels}")
                all_labels.update(labels)
            else:
                print(f"  Sentence {j}: [no labels detected]")
        
        print(f"\n✅ Aggregated labels: {list(all_labels) if all_labels else '[none]'}")
    
    print("\n" + "=" * 80)
    print("✅ Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_sentence_classification())

