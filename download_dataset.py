#!/usr/bin/env python3
"""
Dataset Downloader for DravidianCodeMix Dataset
Downloads and prepares the dataset for Kannada fine-tuning
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetDownloader:
    """Downloads and prepares the DravidianCodeMix dataset"""
    
    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_from_github(self, repo_url, file_paths):
        """Download files from GitHub repository"""
        base_url = "https://raw.githubusercontent.com"
        repo_path = repo_url.replace("https://github.com/", "")
        
        for file_path in file_paths:
            url = f"{base_url}/{repo_path}/main/{file_path}"
            output_path = self.data_dir / Path(file_path).name
            
            logger.info(f"Downloading {file_path}...")
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"Downloaded to {output_path}")
                
            except requests.RequestException as e:
                logger.error(f"Failed to download {file_path}: {e}")
    
    def create_sample_dataset(self):
        """Create a sample dataset for testing purposes"""
        logger.info("Creating sample dataset...")
        
        sample_data = [
            {
                "text": "ನಾನು ಈ ಚಿತ್ರವನ್ನು ಇಷ್ಟಪಟ್ಟೆ. It was amazing!",
                "language": "kannada",
                "sentiment": "positive",
                "offensive": "no"
            },
            {
                "text": "ಇದು ಚೆನ್ನಾಗಿಲ್ಲ. Not good at all.",
                "language": "kannada", 
                "sentiment": "negative",
                "offensive": "no"
            },
            {
                "text": "Movie was okay. ಚಿತ್ರ ಸಾಮಾನ್ಯವಾಗಿತ್ತು.",
                "language": "kannada",
                "sentiment": "neutral", 
                "offensive": "no"
            },
            {
                "text": "This is stupid. ಇದು ಮೂರ್ಖತನ.",
                "language": "kannada",
                "sentiment": "negative",
                "offensive": "yes"
            },
            {
                "text": "Great movie! ಉತ್ತಮ ಚಿತ್ರ!",
                "language": "kannada",
                "sentiment": "positive",
                "offensive": "no"
            },
            {
                "text": "ನಾನು ಈ ಚಿತ್ರವನ್ನು ನೋಡಲು ಇಷ್ಟಪಡುವುದಿಲ್ಲ",
                "language": "kannada",
                "sentiment": "negative", 
                "offensive": "no"
            },
            {
                "text": "Movie was boring. ಚಿತ್ರ ಬೇಸರಿಕೆಯಾಗಿತ್ತು.",
                "language": "kannada",
                "sentiment": "negative",
                "offensive": "no"
            },
            {
                "text": "Excellent performance! ಅತ್ಯುತ್ತಮ ಅಭಿನಯ!",
                "language": "kannada",
                "sentiment": "positive",
                "offensive": "no"
            },
            {
                "text": "ನಾನು ಈ ಚಿತ್ರವನ್ನು ಮತ್ತೆ ನೋಡಲು ಬಯಸುತ್ತೇನೆ",
                "language": "kannada",
                "sentiment": "positive",
                "offensive": "no"
            },
            {
                "text": "Waste of time. ಸಮಯ ವ್ಯರ್ಥ.",
                "language": "kannada",
                "sentiment": "negative",
                "offensive": "no"
            }
        ]
        
        df = pd.DataFrame(sample_data)
        output_path = self.data_dir / "sample_kannada_dataset.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"Sample dataset created at {output_path}")
        return output_path
    
    def download_dravidian_dataset(self):
        """Download the actual DravidianCodeMix dataset"""
        logger.info("Downloading DravidianCodeMix dataset...")
        
        # GitHub repository URL
        repo_url = "https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset"
        
        # File paths to download
        file_paths = [
            "programs/train.csv",
            "programs/test.csv", 
            "programs/dev.csv"
        ]
        
        try:
            self.download_from_github(repo_url, file_paths)
            logger.info("Dataset download completed!")
            
            # Combine the files
            self.combine_datasets()
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Creating sample dataset instead...")
            return self.create_sample_dataset()
    
    def combine_datasets(self):
        """Combine train, test, and dev datasets"""
        logger.info("Combining datasets...")
        
        datasets = []
        for split in ['train', 'test', 'dev']:
            file_path = self.data_dir / f"{split}.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['split'] = split
                datasets.append(df)
                logger.info(f"Loaded {len(df)} samples from {split}.csv")
        
        if datasets:
            combined_df = pd.concat(datasets, ignore_index=True)
            output_path = self.data_dir / "dravidian_dataset.csv"
            combined_df.to_csv(output_path, index=False)
            logger.info(f"Combined dataset saved to {output_path}")
            return output_path
        else:
            logger.warning("No datasets found, creating sample dataset...")
            return self.create_sample_dataset()

def main():
    """Main function to download and prepare the dataset"""
    logging.basicConfig(level=logging.INFO)
    
    downloader = DatasetDownloader()
    
    print("DravidianCodeMix Dataset Downloader")
    print("=" * 40)
    
    choice = input("Choose an option:\n1. Download full dataset\n2. Create sample dataset\nEnter choice (1 or 2): ")
    
    if choice == "1":
        dataset_path = downloader.download_dravidian_dataset()
    else:
        dataset_path = downloader.create_sample_dataset()
    
    print(f"\nDataset ready at: {dataset_path}")
    print("\nYou can now run the fine-tuning script:")
    print(f"python train_mbert_kannada.py --data_path {dataset_path} --task sentiment")

if __name__ == "__main__":
    main()
