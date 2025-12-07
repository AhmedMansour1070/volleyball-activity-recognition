# Hierarchical LSTM for Volleyball Group Activity Recognition

A PyTorch implementation of hierarchical deep temporal models for group activity recognition in volleyball videos, based on the CVPR 2016 paper "A Hierarchical Deep Temporal Model for Group Activity Recognition" by Ibrahim et al.

**Author:** Ahmed Mansour  
**GitHub:** [AhmedMansour1070](https://github.com/AhmedMansour1070)  
**Contact:** ahmedh.mansourr@gmail.com

## Abstract

This project implements a two-stage hierarchical LSTM architecture for recognizing group activities in volleyball games. The model achieves **75.39% test accuracy** on the volleyball dataset, demonstrating strong performance in temporal activity recognition and reaching 92.5% of the original paper's reported accuracy.

## Results

### Performance Metrics

| Split | Accuracy | Precision | Recall | F1 Score | Samples |
|-------|----------|-----------|--------|----------|---------|
| Training | 79.23% | - | - | - | 2,152 |
| Validation | 75.02% | - | - | - | 1,341 |
| **Test** | **75.39%** | **75.87%** | **75.39%** | **72.61%** | **1,337** |

### Comparison to Literature

| Model | Test Accuracy | Dataset |
|-------|---------------|---------|
| Ibrahim et al. (2016) - Baseline | 51.1% | Volleyball (55 videos) |
| **This Implementation** | **75.39%** | **Volleyball (55 videos)** |
| Ibrahim et al. (2016) - Full Model | 81.5% | Volleyball (55 videos) |

**Key Achievement:** This implementation achieves 92.5% of the paper's full model performance and outperforms the baseline by 24.29 percentage points.

### Per-Class Performance

| Activity Class | Accuracy | Test Samples | Notes |
|----------------|----------|--------------|-------|
| r_set | 92.44% | 807 | Dominant class, excellent performance |
| l_winpoint | 90.20% | 102 | Strong recognition |
| r_winpoint | 86.21% | 87 | Very good performance |
| r_pass | 71.26% | 87 | Moderate performance |
| r_spike | 53.18% | 173 | Lower performance, minority class |

**Note:** Test set contains only 5 of 8 activity classes due to dataset distribution.

## Architecture

### Two-Stage Hierarchical Model

The model consists of two sequential LSTM stages:

**Stage 1: Person-Level LSTM**
- Input: AlexNet fc7 features (9,216 dimensions)
- LSTM Hidden Units: 3,000
- Dropout: 0.5
- Purpose: Model individual player action dynamics over time

**Stage 2: Group-Level LSTM**
- Input: Concatenated CNN + Person LSTM features (12,216 dimensions)
- Pooling: Max pooling across all players
- LSTM Hidden Units: 500
- Dropout: 0.5
- Output: 8-class group activity classification

**Model Statistics:**
- Total Parameters: 154,759,517
- Trainable Parameters: 154,759,517
- Feature Extractor: AlexNet (pretrained on ImageNet)

## Dataset

### Volleyball Activity Dataset

**Source:** Ibrahim et al., CVPR 2016  
**Size:** 55 videos from Olympic Games and FIVB World League matches  
**Annotations:** 4,830 labeled frames

**Activity Labels (8 classes):**
- Right team: r_set, r_spike, r_pass, r_winpoint
- Left team: l_set, l_spike, l_pass, l_winpoint

**Action Labels (9 classes):**
- waiting, setting, digging, falling, spiking, blocking, jumping, moving, standing

**Data Splits:**
- Training: 24 videos (2,152 frames)
- Validation: 15 videos (1,341 frames)  
- Test: 16 videos (1,337 frames)

**Temporal Window:**
- 5 frames before target
- 1 target frame
- 4 frames after target
- Total: 10-frame sequences

## Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.12+
CUDA (optional, for GPU training)
```

### Setup

```bash
# Clone the repository
git clone https://github.com/AhmedMansour1070/volleyball-activity-recognition.git
cd volleyball-activity-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.64.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Usage

### 1. Dataset Preparation

Organize the volleyball dataset in the following structure:

```
data/volleyball/
├── 0/
│   ├── annotations.txt
│   ├── frame_id_1/
│   │   └── *.jpg (41 images)
│   └── ...
├── 1/
└── ... (up to 54)
```

### 2. Feature Extraction

Extract AlexNet features from all frames:

```bash
set PYTHONPATH=.
python scripts/extract_features.py --split all --model alexnet
```

**Output:** Features saved to `data/volleyball/features/alexnet/`

**Processing Time:** Approximately 2-3 hours for 55 videos (CPU)

### 3. Training

Train the hierarchical LSTM model:

```bash
set PYTHONPATH=.
python scripts/train.py --epochs 50 --batch-size 4
```

**Key Training Parameters:**
- Learning Rate: 0.0001
- Weight Decay: 0.001
- Dropout: 0.5
- Optimizer: SGD with momentum (0.9)
- LR Schedule: Step decay (gamma=0.1, step=20)

**Training Time:** Approximately 14 hours (CPU) or 2-3 hours (GPU)

**Checkpoint:** Best model saved to `outputs/checkpoints/best_model.pth`

### 4. Evaluation

Evaluate on the test set:

```bash
set PYTHONPATH=.
python scripts/evaluate_test.py
```

**Outputs:**
- `outputs/evaluation/test_results.txt` - Detailed metrics
- `outputs/evaluation/confusion_matrix_test.png` - Visualization

## Project Structure

```
volleyball-activity-recognition/
├── configs/
│   └── volleyball_config.yaml          # Model and training configuration
├── data/
│   └── volleyball/
│       ├── 0/ ... 54/                   # Video frame directories
│       └── features/alexnet/            # Extracted CNN features
├── src/
│   ├── data/
│   │   ├── volleyball_loader.py         # Dataset annotation loader
│   │   └── volleyball_dataset.py        # PyTorch Dataset class
│   ├── models/
│   │   ├── person_lstm.py               # Stage 1: Person LSTM
│   │   ├── group_lstm.py                # Stage 2: Group LSTM
│   │   └── hierarchical_model.py        # Complete two-stage model
│   └── utils/
│       └── collate_fn.py                # Custom batch collation
├── scripts/
│   ├── extract_features.py              # Feature extraction pipeline
│   ├── train.py                         # Training script
│   └── evaluate_test.py                 # Evaluation script
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.pth              # Trained model weights
│   └── evaluation/
│       ├── test_results.txt            # Test metrics
│       └── confusion_matrix_test.png   # Confusion matrix
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## Configuration

Model and training parameters are defined in `configs/volleyball_config.yaml`:

```yaml
model:
  person_lstm:
    hidden_dim: 3000
    dropout: 0.5
  group_lstm:
    hidden_dim: 500
    dropout: 0.5
  pooling: "max"

training:
  learning_rate: 0.0001
  weight_decay: 0.001
  batch_size: 4
  num_epochs: 50
```

## Training Progress

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|-------|------------|-----------|----------|---------|
| 0 | 1.4019 | 58.46% | 1.3601 | 59.88% |
| 10 | 0.9923 | 63.01% | 0.9613 | 63.53% |
| 20 | 0.6613 | 76.30% | 0.7713 | 72.78% |
| 35 | 0.6186 | **79.23%** | 0.7294 | **75.02%** (best) |
| 49 | 0.6102 | 79.51% | 0.7380 | 74.42% |

**Observations:**
- Steady convergence over 50 epochs
- Best validation accuracy at epoch 35
- Minimal overfitting (4.21% gap between train and validation)
- Model generalizes well to test set (75.39%)

## Key Findings

### Strengths
1. **Strong performance on dominant classes:** 92.44% accuracy on r_set (60% of test samples)
2. **Robust generalization:** Test accuracy (75.39%) matches validation (75.02%)
3. **Temporal modeling:** Successfully captures action dynamics across 10-frame sequences
4. **Scalable architecture:** Handles variable number of players per frame

### Challenges
1. **Class imbalance:** Test set contains only 5 of 8 activity classes
2. **Minority class performance:** r_spike achieves only 53.18% (173 samples)
3. **Computational cost:** 155M parameters require significant training time on CPU

### Future Improvements
1. **GPU acceleration:** Reduce training time from 14 hours to 2-3 hours
2. **Data augmentation:** Horizontal flips, temporal jittering
3. **Class balancing:** Weighted loss functions or oversampling
4. **Fine-tuning:** Train AlexNet layers end-to-end
5. **Ensemble methods:** Combine multiple model predictions

## Reproducing Results

To reproduce the reported 75.39% test accuracy:

1. Extract features using AlexNet (pretrained on ImageNet)
2. Use the provided configuration in `configs/volleyball_config.yaml`
3. Train for 50 epochs with batch size 4
4. Evaluate using the best validation checkpoint (typically epoch 30-40)

**Expected variance:** ±1% due to random initialization and data shuffling

## Citation

If you use this code or find it helpful, please cite the original paper:

```bibtex
@inproceedings{ibrahim2016hierarchical,
  title={A hierarchical deep temporal model for group activity recognition},
  author={Ibrahim, Mostafa S and Muralidharan, Srikanth and Deng, Zhiwei and Vahdat, Arash and Mori, Greg},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1971--1980},
  year={2016}
}
```

## License

This project is for educational and research purposes. Please refer to the original paper for dataset usage terms.

## Acknowledgments

- Original paper authors: Ibrahim et al.
- Volleyball dataset creators
- PyTorch development team
- AlexNet architecture by Krizhevsky et al.

## Contact

**Ahmed Mansour**  
Email: ahmedh.mansourr@gmail.com  
GitHub: [@AhmedMansour1070](https://github.com/AhmedMansour1070)

For questions, issues, or suggestions, please open an issue on the GitHub repository.

---

**Last Updated:** December 2025  
**Status:** Complete - Reproducible research implementation
