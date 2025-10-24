#!/usr/bin/env python3
"""
Training script for mBERT Kannada fine-tuning
Command-line interface for training mBERT on Kannada data
"""

import argparse
import os
import json
import logging
from pathlib import Path
import pandas as pd

from mbert_kannada_finetuning import KannadaMBertFineTuner

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Fine-tune mBERT for Kannada language tasks')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the DravidianCodeMix dataset')
    parser.add_argument('--task', type=str, choices=['sentiment', 'offensive'], 
                       default='sentiment', help='Task type: sentiment or offensive')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of the text column in the dataset')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='bert-base-multilingual-cased',
                       help='mBERT model name')
    parser.add_argument('--output_dir', type=str, default='./kannada_mbert_model',
                       help='Output directory for the fine-tuned model')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='Number of warmup steps')
    
    # Evaluation arguments
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Number of steps between evaluations')
    parser.add_argument('--save_steps', type=int, default=500,
                       help='Number of steps between model saves')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    logger.info("Starting mBERT Kannada fine-tuning pipeline...")
    logger.info(f"Task: {args.task}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Output: {args.output_dir}")
    
    # Initialize mBERT fine-tuner
    fine_tuner = KannadaMBertFineTuner(
        model_name=args.model_name,
        num_labels=3 if args.task == 'sentiment' else 2,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps
    )
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Load and prepare data
    df = fine_tuner.load_dravidian_dataset(args.data_path)
    train_dataset, val_dataset = fine_tuner.prepare_data(df)
    
    # Train the model
    logger.info("Starting mBERT training...")
    fine_tuner.train(
        train_dataset, 
        val_dataset, 
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps
    )
    
    # Evaluate the model
    logger.info("Evaluating mBERT model...")
    results = fine_tuner.evaluate(val_dataset)
    
    # Save evaluation results
    with open(os.path.join(args.output_dir, 'eval_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("mBERT fine-tuning completed successfully!")
    logger.info(f"Model saved to: {args.output_dir}")
    logger.info(f"Final F1 macro: {results.get('eval_f1_macro', 'N/A')}")
    logger.info(f"Final accuracy: {results.get('eval_accuracy', 'N/A')}")

if __name__ == "__main__":
    main()
