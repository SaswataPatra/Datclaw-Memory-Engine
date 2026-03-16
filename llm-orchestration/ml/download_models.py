"""
Download required ML models for DAPPY

This script pre-downloads models to avoid timeouts during tests or runtime.
"""

import logging
from transformers import DistilBertModel, DistilBertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_distilbert():
    """Download DistilBERT model and tokenizer"""
    model_name = "distilbert-base-uncased"
    
    logger.info(f"Downloading DistilBERT model: {model_name}")
    logger.info("This may take a few minutes on first run...")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        logger.info("✅ Tokenizer downloaded successfully")
        
        # Download model
        logger.info("Downloading model...")
        model = DistilBertModel.from_pretrained(model_name)
        logger.info("✅ Model downloaded successfully")
        
        logger.info(f"\n🎉 DistilBERT model cached successfully!")
        logger.info(f"Model will be loaded from cache in future runs.")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download model: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DAPPY Model Downloader")
    print("="*60 + "\n")
    
    success = download_distilbert()
    
    if success:
        print("\n✅ All models downloaded successfully!")
        print("You can now run tests or train the classifier.")
    else:
        print("\n❌ Model download failed. Check your internet connection.")
        exit(1)

