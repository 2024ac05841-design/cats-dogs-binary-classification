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

from src.models.train import train_multiple_models


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs classifier")
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/processed',
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--best-model-dir',
        type=str,
        default='models/best_model',
        help='Directory to save the best trained model'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output path for models (alias for --best-model-dir, takes precedence if provided)'
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
    
    args = parser.parse_args()
    
    # Handle output-path as an alias for best-model-dir (--output-path takes precedence)
    best_model_dir = args.output_path if args.output_path else args.best_model_dir
    
    logger.info("Starting multi-model training...")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Best model directory: {best_model_dir}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.lr}")
    
    try:
        results = train_multiple_models(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            best_model_dir=best_model_dir
        )
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
