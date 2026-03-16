"""
Training Script for DistilBERT Memory Classifier

Usage:
    python -m ml.training.train_distilbert --generate-data --train --epochs 5
"""

import argparse
import asyncio
import os
import logging
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
import numpy as np

from ml.extractors.memory_classifier import DistilBERTMemoryClassifier, create_training_batch
from ml.training.distilbert_data_generator import DistilBERTDataGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryDataset(Dataset):
    """PyTorch Dataset for memory classification"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], label_to_idx: dict):
        self.texts = texts
        self.labels = labels
        self.label_to_idx = label_to_idx
        self.num_labels = len(label_to_idx)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        text_labels = self.labels[idx]
        
        # Create multi-hot label vector
        label_vector = torch.zeros(self.num_labels, dtype=torch.float32)
        for label in text_labels:
            if label in self.label_to_idx:
                label_vector[self.label_to_idx[label]] = 1.0
        
        return text, label_vector


def collate_fn(batch, tokenizer, max_length=128, device='cpu'):
    """Custom collate function for DataLoader"""
    texts, labels = zip(*batch)
    
    # Tokenize
    encoded = tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    labels = torch.stack(labels).to(device)
    
    return input_ids, attention_mask, labels


async def generate_dataset(api_key: str, output_path: str, variations_per_seed: int = 5):
    """Generate training dataset"""
    logger.info("Generating training dataset...")
    
    generator = DistilBERTDataGenerator(api_key)
    dataset = await generator.generate_full_dataset(
        variations_per_seed=variations_per_seed,
        include_multi_label=True
    )
    
    # Save dataset
    generator.save_dataset(dataset, output_path)
    
    # Print statistics
    label_dist = generator.get_label_distribution(dataset)
    logger.info(f"Dataset statistics:")
    for label, count in sorted(label_dist.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {label}: {count} examples")
    
    return dataset


def train_epoch(
    model: DistilBERTMemoryClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(dataloader):
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(
    model: DistilBERTMemoryClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    threshold: float = 0.5
) -> dict:
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / len(dataloader),
        'hamming_loss': hamming_loss(all_labels, all_preds),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_labels, all_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_labels, all_preds, average='micro', zero_division=0),
    }
    
    return metrics


def train_model(
    dataset_path: str,
    output_dir: str,
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    max_length: int = 128,
    device: str = None
):
    """Main training function"""
    
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}...")
    generator = DistilBERTDataGenerator(openai_api_key="dummy")  # Just for loading
    dataset = generator.load_dataset(dataset_path)
    
    # Split dataset
    texts, labels = zip(*dataset)
    texts = list(texts)
    labels = list(labels)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"Train size: {len(train_texts)}, Val size: {len(val_texts)}")
    
    # Initialize model
    logger.info("Initializing DistilBERT model...")
    model = DistilBERTMemoryClassifier(device=device)
    
    # Create datasets
    train_dataset = MemoryDataset(train_texts, train_labels, model.label_to_idx)
    val_dataset = MemoryDataset(val_texts, val_labels, model.label_to_idx)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer, max_length, device)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, model.tokenizer, max_length, device)
    )
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()  # Multi-label classification
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info(f"Starting training for {epochs} epochs...")
    best_f1 = 0.0
    
    for epoch in range(epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        logger.info(f"Training loss: {train_loss:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, criterion, device)
        logger.info(f"Validation metrics:")
        for metric, value in val_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save best model
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            output_path = os.path.join(output_dir, 'best_model.pt')
            model.save_model(output_path)
            logger.info(f"✅ New best model saved! F1: {best_f1:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best F1: {best_f1:.4f}")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Train DistilBERT Memory Classifier")
    
    parser.add_argument('--generate-data', action='store_true', help='Generate training dataset')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--dataset-path', type=str, default='data/distilbert_dataset.jsonl', help='Path to dataset file')
    parser.add_argument('--output-dir', type=str, default='models/distilbert', help='Output directory for model')
    parser.add_argument('--openai-api-key', type=str, help='OpenAI API key for data generation')
    parser.add_argument('--variations-per-seed', type=int, default=5, help='Variations per seed example')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(os.path.dirname(args.dataset_path), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    if args.generate_data:
        if not args.openai_api_key:
            # Try to get from environment
            args.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not args.openai_api_key:
                raise ValueError("OpenAI API key required for data generation. Set --openai-api-key or OPENAI_API_KEY env var.")
        
        asyncio.run(generate_dataset(
            args.openai_api_key,
            args.dataset_path,
            args.variations_per_seed
        ))
    
    # Train model
    if args.train:
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}. Run with --generate-data first.")
        
        train_model(
            args.dataset_path,
            args.output_dir,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            device=args.device
        )


if __name__ == '__main__':
    main()

