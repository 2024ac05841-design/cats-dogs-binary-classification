#!/usr/bin/env python
"""
Standalone training script for model training
Supports single model training or multi-model comparison
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

from src.models import train_model, train_multiple_models


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
        choices=['simple_cnn', 'resnet18', 'logistic_regression'],
        help='Model architecture to use (single model training)'
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
        help='Path to save trained model (single model training)'
    )
    
    parser.add_argument(
        '--compare-models',
        action='store_true',
        help='Train and compare multiple models (SimpleCNN, LogisticRegression, ResNet18)'
    )
    
    parser.add_argument(
        '--best-model-dir',
        type=str,
        default='models/best_model',
        help='Directory to save best model (when comparing models)'
    )
    
    args = parser.parse_args()
    
    if args.compare_models:
        # Train multiple models and compare
        logger.info("="*70)
        logger.info("Starting multi-model training and comparison...")
        logger.info("Training: simple_cnn, logistic_regression, and resnet18")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info("="*70)
        
        try:
            results, best_model_info = train_multiple_models(
                data_dir=args.data_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                best_model_dir=args.best_model_dir
            )
            logger.info("="*70)
            logger.info("Multi-model training completed successfully!")
            logger.info(f"Best model: {best_model_info['name']}")
            logger.info(f"Validation Accuracy: {best_model_info['val_acc']:.2f}%")
            logger.info(f"Validation Loss: {best_model_info['val_loss']:.4f}")
            logger.info(f"Best model saved to: {best_model_dir}")
            logger.info(f"Results saved to: {best_model_info.get('result_path', 'N/A')}")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Multi-model training failed: {e}")
            raise
    else:
        # Train single model
        logger.info("="*70)
        logger.info("Starting single model training...")
        logger.info(f"Model: {args.model}")
        logger.info(f"Epochs: {args.epochs}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr}")
        logger.info("="*70)
        
        try:
            train_model(
                data_dir=args.data_dir,
                model_name=args.model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_path=args.output_path
            )
            logger.info("="*70)
            logger.info("Training completed successfully!")
            logger.info(f"Model saved to: {args.output_path}")
            logger.info("="*70)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


if __name__ == "__main__":
    main()
