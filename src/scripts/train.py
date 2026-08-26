#!/usr/bin/env python
"""
Standalone training script for model training
"""

import argparse
import logging
import os
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.models import train_model


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='simple_cnn',
        choices=['simple_cnn', 'resnet18'],
        help='Model architecture to use'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='models/model.pkl',
        help='Path to save trained model'
    )
    
    args = parser.parse_args()
    
    logger.info("Starting model training...")
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    try:
        train_model(
            data_dir=args.data_dir,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            save_path=args.output_path
        )
        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
