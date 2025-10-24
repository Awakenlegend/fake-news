#!/usr/bin/env python3
"""
Test script for Kannada fine-tuning implementation
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mbert_kannada_finetuning import KannadaMBertFineTuner
from data_preprocessor import DravidianDataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_data_preprocessing():
    """Test data preprocessing functionality"""
    logger.info("Testing data preprocessing...")
    
    # Create sample data
    sample_data = {
        'text': [
            'ನಾನು ಈ ಚಿತ್ರವನ್ನು ಇಷ್ಟಪಟ್ಟೆ. It was amazing!',
            'ಇದು ಚೆನ್ನಾಗಿಲ್ಲ. Not good at all.',
            'Movie was okay. ಚಿತ್ರ ಸಾಮಾನ್ಯವಾಗಿತ್ತು.',
            'This is stupid. ಇದು ಮೂರ್ಖತನ.',
            'Great movie! ಉತ್ತಮ ಚಿತ್ರ!'
        ],
        'language': ['kannada'] * 5,
        'sentiment': ['positive', 'negative', 'neutral', 'negative', 'positive'],
        'offensive': ['no', 'no', 'no', 'yes', 'no']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Test preprocessor
    preprocessor = DravidianDataPreprocessor()
    
    # Test language detection
    for text in df['text']:
        lang_info = preprocessor.detect_language_mix(text)
        logger.info(f"Text: {text[:30]}...")
        logger.info(f"Language info: {lang_info}")
    
    # Test filtering
    kannada_df = preprocessor.filter_kannada_data(df)
    logger.info(f"Filtered to {len(kannada_df)} Kannada samples")
    
    # Test sentiment preparation
    sentiment_df = preprocessor.prepare_sentiment_data(kannada_df)
    logger.info(f"Sentiment data prepared: {len(sentiment_df)} samples")
    logger.info(f"Label distribution: {sentiment_df['label'].value_counts().to_dict()}")
    
    # Test offensive preparation
    offensive_df = preprocessor.prepare_offensive_data(kannada_df)
    logger.info(f"Offensive data prepared: {len(offensive_df)} samples")
    logger.info(f"Label distribution: {offensive_df['label'].value_counts().to_dict()}")
    
    logger.info("✓ Data preprocessing test passed!")
    return True

def test_model_loading():
    """Test model loading functionality"""
    logger.info("Testing model loading...")
    
    try:
        # Test with a smaller model for faster testing
        fine_tuner = KannadaMBertFineTuner(
            model_name="distilbert-base-multilingual-cased",  # Smaller model for testing
            num_labels=3
        )
        
        # This will download the model if not cached
        fine_tuner.load_model_and_tokenizer()
        
        logger.info("✓ Model loading test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Model loading test failed: {e}")
        return False

def test_tokenization():
    """Test tokenization with Kannada text"""
    logger.info("Testing tokenization...")
    
    try:
        fine_tuner = KannadaMBertFineTuner(
            model_name="distilbert-base-multilingual-cased",
            num_labels=3
        )
        fine_tuner.load_model_and_tokenizer()
        
        # Test Kannada text
        kannada_texts = [
            "ನಾನು ಈ ಚಿತ್ರವನ್ನು ಇಷ್ಟಪಟ್ಟೆ",
            "ಇದು ಚೆನ್ನಾಗಿಲ್ಲ",
            "Movie was okay. ಚಿತ್ರ ಸಾಮಾನ್ಯವಾಗಿತ್ತು."
        ]
        
        for text in kannada_texts:
            tokens = fine_tuner.tokenizer.tokenize(text)
            logger.info(f"Text: {text}")
            logger.info(f"Tokens: {tokens[:10]}...")  # Show first 10 tokens
        
        logger.info("✓ Tokenization test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Tokenization test failed: {e}")
        return False

def test_prediction():
    """Test prediction functionality"""
    logger.info("Testing prediction...")
    
    try:
        fine_tuner = KannadaMBertFineTuner(
            model_name="distilbert-base-multilingual-cased",
            num_labels=3
        )
        fine_tuner.load_model_and_tokenizer()
        
        # Test predictions
        test_texts = [
            "ನಾನು ಈ ಚಿತ್ರವನ್ನು ಇಷ್ಟಪಟ್ಟೆ",  # I liked this movie
            "ಇದು ಚೆನ್ನಾಗಿಲ್ಲ"  # This is not good
        ]
        
        predictions = fine_tuner.predict(test_texts)
        logger.info(f"Predictions: {predictions}")
        
        logger.info("✓ Prediction test passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Prediction test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("Running Kannada fine-tuning tests...")
    logger.info("=" * 50)
    
    tests = [
        ("Data Preprocessing", test_data_preprocessing),
        ("Model Loading", test_model_loading),
        ("Tokenization", test_tokenization),
        ("Prediction", test_prediction)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} test passed!")
            else:
                logger.error(f"✗ {test_name} test failed!")
        except Exception as e:
            logger.error(f"✗ {test_name} test failed with exception: {e}")
    
    logger.info("\n" + "=" * 50)
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Your setup is ready for Kannada fine-tuning.")
    else:
        logger.warning(f"⚠ {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

def main():
    """Main test function"""
    print("Kannada Fine-tuning Test Suite")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("mbert_kannada_finetuning.py"):
        print("Error: Please run this script from the kannada_finetuning directory")
        return
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! You can now proceed with fine-tuning.")
        print("\nNext steps:")
        print("1. Download dataset: python download_dataset.py")
        print("2. Run training: python train_mbert_kannada.py --data_path data/dataset.csv --task sentiment")
    else:
        print("\n⚠ Some tests failed. Please fix the issues before proceeding.")

if __name__ == "__main__":
    main()
