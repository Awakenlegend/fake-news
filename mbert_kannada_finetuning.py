#!/usr/bin/env python3
"""
mBERT Fine-tuning for Kannada using DravidianCodeMix Dataset
Specialized implementation for mBERT-based Kannada language tasks
"""

import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    BertConfig,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_recall_fscore_support
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KannadaMBertDataset(Dataset):
    """Custom dataset class for Kannada text data with mBERT"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize with mBERT tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class KannadaMBertFineTuner:
    """mBERT fine-tuner specifically optimized for Kannada language tasks"""
    
    def __init__(self, 
                 model_name: str = "bert-base-multilingual-cased",
                 num_labels: int = 2,
                 max_length: int = 512,
                 learning_rate: float = 2e-5,
                 warmup_steps: int = 500):
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def load_model_and_tokenizer(self):
        """Load mBERT model and tokenizer with Kannada-specific optimizations"""
        logger.info(f"Loading mBERT model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Add special tokens for Kannada-English code-mixing
        special_tokens = {
            'additional_special_tokens': [
                '<kn>', '</kn>',  # Kannada markers
                '<en>', '</en>',  # English markers
                '<cm>', '</cm>'   # Code-mixed markers
            ]
        }
        
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("mBERT model and tokenizer loaded successfully")
        logger.info(f"Vocabulary size: {len(self.tokenizer)}")
    
    def preprocess_kannada_text(self, text: str) -> str:
        """Preprocess Kannada text for better mBERT performance"""
        if pd.isna(text):
            return ""
        
        text = str(text).strip()
        
        # Handle code-mixed text
        # Add language markers for better understanding
        import re
        
        # Simple heuristic to detect code-mixed text
        has_kannada = bool(re.search(r'[\u0C80-\u0CFF]', text))
        has_english = bool(re.search(r'[a-zA-Z]', text))
        
        if has_kannada and has_english:
            # Add code-mixed markers
            text = f"<cm>{text}</cm>"
        elif has_kannada:
            # Add Kannada markers
            text = f"<kn>{text}</kn>"
        elif has_english:
            # Add English markers
            text = f"<en>{text}</en>"
        
        return text
    
    def load_dravidian_dataset(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess DravidianCodeMix dataset for Kannada"""
        logger.info("Loading DravidianCodeMix dataset...")
        
        # Load dataset
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or JSON.")
        
        logger.info(f"Total samples loaded: {len(df)}")
        
        # Filter for Kannada data
        if 'language' in df.columns:
            kannada_data = df[df['language'].str.lower().isin(['kannada', 'kn', 'kannada-english'])].copy()
        else:
            # Auto-detect Kannada text
            kannada_data = df.copy()
            kannada_data['has_kannada'] = kannada_data['text'].str.contains(r'[\u0C80-\u0CFF]', na=False)
            kannada_data = kannada_data[kannada_data['has_kannada']].copy()
        
        logger.info(f"Kannada samples: {len(kannada_data)}")
        
        # Preprocess text
        kannada_data['processed_text'] = kannada_data['text'].apply(self.preprocess_kannada_text)
        
        # Handle labels based on task
        if 'sentiment' in kannada_data.columns:
            # Sentiment analysis task
            sentiment_mapping = {'positive': 2, 'negative': 0, 'neutral': 1}
            kannada_data['label'] = kannada_data['sentiment'].map(sentiment_mapping)
            self.num_labels = 3
        elif 'offensive' in kannada_data.columns:
            # Offensive language detection task
            offensive_mapping = {'yes': 1, 'no': 0, 'Yes': 1, 'No': 0, 'OFF': 1, 'NOT': 0}
            kannada_data['label'] = kannada_data['offensive'].map(offensive_mapping)
            self.num_labels = 2
        else:
            raise ValueError("No sentiment or offensive column found in dataset")
        
        # Remove rows with missing labels
        kannada_data = kannada_data.dropna(subset=['label'])
        
        logger.info(f"Final dataset size: {len(kannada_data)}")
        logger.info(f"Label distribution: {kannada_data['label'].value_counts().to_dict()}")
        
        return kannada_data
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'processed_text', 
                    label_column: str = 'label') -> Tuple[KannadaMBertDataset, KannadaMBertDataset]:
        """Prepare data for training with proper stratification"""
        texts = df[text_column].tolist()
        labels = df[label_column].tolist()
        
        # Stratified split to maintain label distribution
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )
        
        # Create datasets
        train_dataset = KannadaMBertDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = KannadaMBertDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute comprehensive evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average='macro')
        f1_weighted = f1_score(labels, predictions, average='weighted')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
        }
        
        # Add per-class metrics
        for i in range(len(precision)):
            metrics[f'precision_class_{i}'] = precision[i]
            metrics[f'recall_class_{i}'] = recall[i]
            metrics[f'f1_class_{i}'] = f1[i]
            metrics[f'support_class_{i}'] = support[i]
        
        return metrics
    
    def train(self, train_dataset: KannadaMBertDataset, 
              val_dataset: KannadaMBertDataset, 
              output_dir: str = "./kannada_mbert_model",
              num_epochs: int = 3,
              batch_size: int = 16,
              eval_steps: int = 500,
              save_steps: int = 500):
        """Train mBERT model with optimized parameters for Kannada"""
        
        logger.info("Starting mBERT training for Kannada...")
        
        # Training arguments optimized for mBERT
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            save_total_limit=3,
            report_to=None,  # Disable wandb/tensorboard
            dataloader_drop_last=False,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            gradient_accumulation_steps=1,
            adam_epsilon=1e-8,
            max_grad_norm=1.0
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train the model
        logger.info("Starting training...")
        self.trainer.train()
        
        # Save the final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'max_length': self.max_length,
            'learning_rate': self.learning_rate,
            'num_epochs': num_epochs,
            'batch_size': batch_size
        }
        
        with open(os.path.join(output_dir, 'training_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"mBERT model saved to {output_dir}")
    
    def evaluate(self, test_dataset: KannadaMBertDataset) -> Dict:
        """Evaluate the model on test data"""
        logger.info("Evaluating mBERT model...")
        
        results = self.trainer.evaluate(test_dataset)
        
        logger.info("Evaluation Results:")
        for key, value in results.items():
            logger.info(f"{key}: {value:.4f}")
        
        return results
    
    def predict(self, texts: List[str]) -> List[int]:
        """Make predictions on new texts"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Preprocess text
                processed_text = self.preprocess_kannada_text(text)
                
                # Tokenize
                inputs = self.tokenizer(
                    processed_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get prediction
                outputs = self.model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=-1).item()
                predictions.append(prediction)
        
        return predictions
    
    def predict_with_confidence(self, texts: List[str]) -> List[Dict]:
        """Make predictions with confidence scores"""
        self.model.eval()
        results = []
        
        with torch.no_grad():
            for text in texts:
                processed_text = self.preprocess_kannada_text(text)
                
                inputs = self.tokenizer(
                    processed_text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
                
                results.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': probabilities.cpu().numpy().tolist()[0]
                })
        
        return results

def main():
    """Main function for mBERT Kannada fine-tuning"""
    
    # Initialize mBERT fine-tuner
    fine_tuner = KannadaMBertFineTuner(
        model_name="bert-base-multilingual-cased",
        num_labels=3,  # Will be adjusted based on task
        max_length=512,
        learning_rate=2e-5
    )
    
    # Load model and tokenizer
    fine_tuner.load_model_and_tokenizer()
    
    # Load dataset
    data_path = "dravidian_dataset.csv"  # Update this path
    try:
        df = fine_tuner.load_dravidian_dataset(data_path)
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {data_path}")
        logger.info("Please download the DravidianCodeMix dataset and update the path")
        return
    
    # Prepare data
    train_dataset, val_dataset = fine_tuner.prepare_data(df)
    
    # Train the model
    fine_tuner.train(train_dataset, val_dataset)
    
    # Evaluate
    results = fine_tuner.evaluate(val_dataset)
    
    logger.info("mBERT fine-tuning completed successfully!")
    
    # Example predictions
    sample_texts = [
        "ನಾನು ಈ ಚಿತ್ರವನ್ನು ಇಷ್ಟಪಟ್ಟೆ",  # I liked this movie
        "ಇದು ಚೆನ್ನಾಗಿಲ್ಲ",  # This is not good
        "Movie was okay. ಚಿತ್ರ ಸಾಮಾನ್ಯವಾಗಿತ್ತು."  # Code-mixed
    ]
    
    predictions = fine_tuner.predict(sample_texts)
    logger.info(f"Sample predictions: {predictions}")
    
    # Predictions with confidence
    detailed_predictions = fine_tuner.predict_with_confidence(sample_texts)
    for i, result in enumerate(detailed_predictions):
        logger.info(f"Text {i+1}: {sample_texts[i]}")
        logger.info(f"Prediction: {result['prediction']}, Confidence: {result['confidence']:.3f}")

if __name__ == "__main__":
    main()
