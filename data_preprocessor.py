#!/usr/bin/env python3
"""
Data Preprocessor for DravidianCodeMix Dataset
Handles data cleaning, preprocessing, and preparation for Kannada fine-tuning
"""

import pandas as pd
import re
import json
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class DravidianDataPreprocessor:
    """Preprocessor for DravidianCodeMix dataset"""
    
    def __init__(self):
        self.kannada_pattern = re.compile(r'[\u0C80-\u0CFF]+')  # Kannada Unicode range
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove special characters but keep Kannada and English
        text = re.sub(r'[^\w\s\u0C80-\u0CFF]', ' ', text)
        
        return text.strip()
    
    def detect_language_mix(self, text: str) -> Dict[str, bool]:
        """Detect if text contains Kannada, English, or both"""
        has_kannada = bool(self.kannada_pattern.search(text))
        has_english = bool(self.english_pattern.search(text))
        
        return {
            'has_kannada': has_kannada,
            'has_english': has_english,
            'is_code_mixed': has_kannada and has_english
        }
    
    def filter_kannada_data(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """Filter dataset to include only Kannada-related data"""
        logger.info("Filtering for Kannada data...")
        
        # Create a copy
        filtered_df = df.copy()
        
        # Add language detection columns
        language_info = filtered_df[text_column].apply(self.detect_language_mix)
        filtered_df['has_kannada'] = [info['has_kannada'] for info in language_info]
        filtered_df['has_english'] = [info['has_english'] for info in language_info]
        filtered_df['is_code_mixed'] = [info['is_code_mixed'] for info in language_info]
        
        # Filter for Kannada data (either pure Kannada or code-mixed with Kannada)
        kannada_mask = filtered_df['has_kannada'] == True
        
        # If there's a language column, also filter by it
        if 'language' in filtered_df.columns:
            kannada_mask = kannada_mask | (filtered_df['language'].str.lower().isin(['kannada', 'kn']))
        
        filtered_df = filtered_df[kannada_mask].copy()
        
        logger.info(f"Filtered to {len(filtered_df)} Kannada samples")
        logger.info(f"Code-mixed samples: {filtered_df['is_code_mixed'].sum()}")
        
        return filtered_df
    
    def prepare_sentiment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for sentiment analysis task"""
        logger.info("Preparing sentiment analysis data...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Map sentiment labels
        sentiment_mapping = {
            'positive': 2,
            'negative': 0,
            'neutral': 1,
            'Positive': 2,
            'Negative': 0,
            'Neutral': 1
        }
        
        if 'sentiment' in df.columns:
            df['label'] = df['sentiment'].map(sentiment_mapping)
        elif 'label' in df.columns:
            df['label'] = df['label'].map(sentiment_mapping)
        else:
            logger.warning("No sentiment column found. Please check your dataset.")
            return df
        
        # Remove rows with missing labels
        df = df.dropna(subset=['label'])
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        logger.info(f"Sentiment distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def prepare_offensive_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for offensive language identification task"""
        logger.info("Preparing offensive language data...")
        
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Map offensive labels
        offensive_mapping = {
            'yes': 1,
            'no': 0,
            'Yes': 1,
            'No': 0,
            'OFF': 1,
            'NOT': 0
        }
        
        if 'offensive' in df.columns:
            df['label'] = df['offensive'].map(offensive_mapping)
        elif 'label' in df.columns:
            df['label'] = df['label'].map(offensive_mapping)
        else:
            logger.warning("No offensive column found. Please check your dataset.")
            return df
        
        # Remove rows with missing labels
        df = df.dropna(subset=['label'])
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        logger.info(f"Offensive distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    def create_train_val_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                             stratify_column: str = 'label') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-validation split"""
        from sklearn.model_selection import train_test_split
        
        train_df, val_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42, 
            stratify=df[stratify_column]
        )
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file"""
        if output_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif output_path.endswith('.json'):
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError("Unsupported output format. Use CSV or JSON.")
        
        logger.info(f"Processed data saved to {output_path}")

def main():
    """Example usage of the preprocessor"""
    preprocessor = DravidianDataPreprocessor()
    
    # Example: Load and process data
    # df = pd.read_csv('dravidian_dataset.csv')
    # kannada_df = preprocessor.filter_kannada_data(df)
    # processed_df = preprocessor.prepare_sentiment_data(kannada_df)
    # train_df, val_df = preprocessor.create_train_val_split(processed_df)
    
    print("Data preprocessor ready for use!")

if __name__ == "__main__":
    main()
