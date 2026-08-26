#!/usr/bin/env python
"""
Script to prepare and split dataset for training
"""

import os
import shutil
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(raw_dir: str = "data/raw", 
                   processed_dir: str = "data/processed",
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1):
    """
    Prepare and split dataset into train/val/test sets
    
    Args:
        raw_dir: Directory with raw data (cats/ and dogs/ subdirectories)
        processed_dir: Output directory for processed data
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
    """
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    
    # Validate input
    if not (raw_path / "cats").exists() or not (raw_path / "dogs").exists():
        logger.error(f"Expected 'cats' and 'dogs' directories in {raw_dir}")
        logger.error("Dataset structure should be:")
        logger.error("  data/raw/")
        logger.error("    ├── cats/")
        logger.error("    │   ├── cat_1.jpg")
        logger.error("    │   └── cat_2.jpg")
        logger.error("    └── dogs/")
        logger.error("        ├── dog_1.jpg")
        logger.error("        └── dog_2.jpg")
        return False
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['cats', 'dogs']:
            (processed_path / split / cls).mkdir(parents=True, exist_ok=True)
    
    logger.info("Created directory structure")
    
    # Split and copy datasets
    test_ratio = 1.0 - train_ratio - val_ratio
    
    for cls in ['cats', 'dogs']:
        src_dir = raw_path / cls
        files = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.jpeg'))
        
        if not files:
            logger.warning(f"No images found in {src_dir}")
            continue
        
        # First split: train+val vs test
        train_val_files, test_files = train_test_split(
            files, 
            test_size=test_ratio,
            random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_size_adjusted,
            random_state=42
        )
        
        # Copy files to respective directories
        for split, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            for file in split_files:
                dst = processed_path / split / cls / file.name
                shutil.copy2(file, dst)
            logger.info(f"{cls} {split}: {len(split_files)} images")
    
    logger.info("Dataset preparation complete!")
    
    # Print summary
    total_train = sum(1 for _ in (processed_path / 'train').rglob('*.jpg'))
    total_val = sum(1 for _ in (processed_path / 'val').rglob('*.jpg'))
    total_test = sum(1 for _ in (processed_path / 'test').rglob('*.jpg'))
    
    print("\n" + "="*50)
    print("Dataset Summary")
    print("="*50)
    print(f"Training samples: {total_train}")
    print(f"Validation samples: {total_val}")
    print(f"Test samples: {total_test}")
    print(f"Total: {total_train + total_val + total_test}")
    print("="*50)
    
    return True


if __name__ == "__main__":
    print("Preparing dataset...")
    success = prepare_dataset()
    if not success:
        exit(1)
