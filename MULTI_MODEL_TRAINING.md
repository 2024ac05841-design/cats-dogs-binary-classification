# Multi-Model Training & Selection Guide

## Overview

You can now train 3 different models and automatically select the best one:

1. **SimpleCNN** - Custom baseline CNN (2 conv blocks)
2. **Logistic Regression** - Simple linear model on flattened images
3. **ResNet18** - Transfer learning from ImageNet pretrained weights

## Quick Start

### Option 1: Train All 3 Models & Select Best (Recommended)

```bash
# From project root
python src/scripts/train.py --compare-models --epochs 20 --batch-size 32
```

**What happens:**
1. Trains SimpleCNN for 20 epochs
2. Trains Logistic Regression for 20 epochs
3. Trains ResNet18 for 20 epochs
4. Compares validation accuracy of all 3
5. Copies best model to `models/best_model/`
6. Saves comparison results

**Output:**
```
models/best_model/
├── best_model_simple_cnn.pkl          # (or logistic_regression, or resnet18)
└── model_comparison.json              # Shows all 3 models' performance
```

### Option 2: Train Single Model

```bash
# Train SimpleCNN
python src/scripts/train.py --model simple_cnn --epochs 20 --output-path models/simple_cnn.pkl

# Train Logistic Regression
python src/scripts/train.py --model logistic_regression --epochs 20 --output-path models/logreg.pkl

# Train ResNet18
python src/scripts/train.py --model resnet18 --epochs 20 --output-path models/resnet18.pkl
```

## Command Options

```bash
python src/scripts/train.py [OPTIONS]

Options:
  --data-dir PATH                Path to training data (default: data/processed)
  --model {simple_cnn, resnet18, logistic_regression}
                                Model to train in single mode (default: simple_cnn)
  --epochs N                    Number of training epochs (default: 20)
  --batch-size N                Batch size (default: 32)
  --lr RATE                     Learning rate (default: 0.001)
  --output-path PATH            Save path for single model (default: models/model.pkl)
  --compare-models              Train & compare all 3 models (flag)
  --best-model-dir PATH         Dir for best model (default: models/best_model)
```

## Examples

### Train all models for 30 epochs with batch size 16
```bash
python src/scripts/train.py --compare-models --epochs 30 --batch-size 16
```

### Train single model with custom output path
```bash
python src/scripts/train.py --model resnet18 --epochs 25 --output-path my_models/resnet.pkl
```

### Use different data directory
```bash
python src/scripts/train.py --compare-models --data-dir custom_data/
```

## Model Comparison Results

When using `--compare-models`, a JSON file is created at `models/best_model/model_comparison.json`:

```json
{
  "best_model": "resnet18",
  "best_model_val_acc": 94.32,
  "best_model_val_loss": 0.1245,
  "all_models": {
    "simple_cnn": {
      "val_loss": 0.2341,
      "val_acc": 89.12,
      "train_loss": 0.1523,
      "train_acc": 91.45
    },
    "logistic_regression": {
      "val_loss": 0.5432,
      "val_acc": 75.23,
      "train_loss": 0.4821,
      "train_acc": 78.90
    },
    "resnet18": {
      "val_loss": 0.1245,
      "val_acc": 94.32,
      "train_loss": 0.0892,
      "train_acc": 96.78
    }
  }
}
```

## Model Details

### SimpleCNN
- **Architecture**: 3 convolutional blocks with pooling and batch norm
- **Parameters**: ~1.2M
- **Speed**: Fast training
- **Performance**: Good baseline (80-90% accuracy expected)

### Logistic Regression
- **Architecture**: Flattens image → Linear layer
- **Parameters**: ~48M (224×224×3 = 150,528 input features)
- **Speed**: Fastest training
- **Performance**: Baseline (70-80% accuracy expected)
- **Best for**: Quick testing

### ResNet18
- **Architecture**: Pre-trained ImageNet weights, fine-tuned
- **Parameters**: ~11.2M
- **Speed**: Moderate training
- **Performance**: Best performance (92-97% accuracy expected)
- **Best for**: Production use

## File Structure After Training

```
models/
├── best_model/
│   ├── best_model_resnet18.pkl       # Best model (example)
│   └── model_comparison.json         # Comparison results
├── temp_simple_cnn.pkl               # (cleaned up after)
├── temp_logistic_regression.pkl      # (cleaned up after)
└── temp_resnet18.pkl                 # (cleaned up after)
```

## Using the Best Model

```python
import torch
from src.inference.model_utils import load_model, predict_from_path
from src.models import create_model

# Load best model
best_model = torch.load('models/best_model/best_model_resnet18.pkl')

# Or use the utility function
model = load_model('models/best_model/best_model_resnet18.pkl', create_model('resnet18'))

# Make predictions
class_name, confidence, probs = predict_from_path(model, 'test_image.jpg')
print(f"Prediction: {class_name} ({confidence:.2%})")
```

## Deployment

Copy the best model to your inference service:

```bash
# In Docker/deployment setup
cp models/best_model/best_model_*.pkl models/model.pkl
```

Or update your FastAPI app's MODEL_PATH to:
```bash
export MODEL_PATH=models/best_model/best_model_resnet18.pkl
```

## Tips & Best Practices

1. **Start with quick test**: Use `--epochs 5 --batch-size 64` to test setup
2. **Monitor each model**: Check MLflow UI to see training progress
3. **GPU usage**: ResNet18 needs more VRAM. Use smaller batch size if OOM
4. **Validation accuracy**: Usually best metric for model selection (used here)
5. **Fine-tuning**: After selection, can train best model longer with lower LR

## Troubleshooting

**Issue**: Logistic Regression runs out of memory
- **Solution**: It's fine - this is expected due to large flattened input size
- The memory usage indicates why CNNs are better for image tasks

**Issue**: ResNet18 training is slow
- **Solution**: Use smaller batch size or skip it: `--model simple_cnn`
- ResNet has more parameters so it's slower but gives better accuracy

**Issue**: Model comparison shows all models poorly
- **Solution**: Check your data is in `data/processed/train/` and `data/processed/val/`
- Run `python src/scripts/prepare_data.py` first if needed

## Next Steps

1. ✅ Prepare dataset
2. ✅ Train multiple models
3. ✅ Select best model
4. **Deploy best model** - See QUICKSTART.md for deployment steps
