"""
Evaluate trained model on test set
"""

import sys
sys.path.insert(0, '.')

import torch
import yaml
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.volleyball_dataset import VolleyballDataset
from src.models.hierarchical_model import HierarchicalModel
from src.utils.collate_fn import volleyball_collate_fn


def evaluate_model(model, test_loader, device, config):
    """Evaluate model on test set"""
    
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            features = batch['features'].to(device)
            labels = batch['activity_label'].to(device)
            
            # Forward pass
            outputs = model(features)
            _, predicted = torch.max(outputs['activity_logits'], 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels
    }


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {save_path}")


def main():
    # Load config
    with open('configs/volleyball_config.yaml') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset
    print("Loading test dataset...")
    test_dataset = VolleyballDataset(
        video_ids=config['dataset']['test_videos'],
        features_dir=Path('data/volleyball/features/alexnet'),
        config=config['dataset'],
        split='test'
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=volleyball_collate_fn
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print("\nLoading best model...")
    model = HierarchicalModel(
        feature_dim=9216,
        person_hidden_dim=config['model']['person_lstm']['hidden_dim'],
        group_hidden_dim=config['model']['group_lstm']['hidden_dim'],
        num_action_classes=config['model']['num_action_classes'],
        num_activity_classes=config['model']['num_activity_classes'],
        pooling=config['model']['pooling'],
        dropout=config['model']['person_lstm']['dropout']
    )
    
    # Load checkpoint
    checkpoint = torch.load('outputs/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")
    
    # Evaluate
    results = evaluate_model(model, test_loader, device, config)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']*100:.2f}%")
    print(f"Recall:    {results['recall']*100:.2f}%")
    print(f"F1 Score:  {results['f1']*100:.2f}%")
    print("="*60)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    class_names = config['dataset']['activity_labels']
    
    for i, class_name in enumerate(class_names):
        mask = results['labels'] == i
        if mask.sum() > 0:
            class_acc = (results['predictions'][mask] == i).sum() / mask.sum()
            print(f"  {class_name:12s}: {class_acc*100:5.2f}% ({mask.sum()} samples)")
    
    # Save confusion matrix
    output_dir = Path('outputs/evaluation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_confusion_matrix(
        results['confusion_matrix'],
        class_names,
        output_dir / 'confusion_matrix_test.png'
    )
    
    # Save results
    results_file = output_dir / 'test_results.txt'
    with open(results_file, 'w') as f:
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Accuracy:  {results['accuracy']*100:.2f}%\n")
        f.write(f"Precision:      {results['precision']*100:.2f}%\n")
        f.write(f"Recall:         {results['recall']*100:.2f}%\n")
        f.write(f"F1 Score:       {results['f1']*100:.2f}%\n\n")
        f.write("Per-class Accuracy:\n")
        for i, class_name in enumerate(class_names):
            mask = results['labels'] == i
            if mask.sum() > 0:
                class_acc = (results['predictions'][mask] == i).sum() / mask.sum()
                f.write(f"  {class_name:12s}: {class_acc*100:5.2f}%\n")
    
    print(f"\n✓ Saved results to {results_file}")
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()