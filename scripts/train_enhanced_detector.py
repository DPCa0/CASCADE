#!/usr/bin/env python3
"""
Enhanced Unified Detector Training Script

Trains the enhanced model with:
1. Position-aware Layer 2 features
2. Attention-based pooling
3. Transition pattern encoding
"""

import sys
import json
import logging
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, f1_score,
    precision_recall_curve, accuracy_score, precision_score, recall_score
)
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Import enhanced model classes
from enhanced_detector_models import EnhancedFeatureExtractor, EnhancedIPIDetector

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(attack_path: str, benign_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Load training data."""
    with open(attack_path, 'r') as f:
        attack_data = json.load(f)

    with open(benign_path, 'r') as f:
        benign_data = json.load(f)

    logger.info(f"Loaded data: {len(attack_data)} attack, {len(benign_data)} benign samples")
    return attack_data, benign_data


def extract_sequence_from_tool_parameters(tool_params: List[Dict]) -> List[Dict]:
    """Extract function call sequence from tool_parameters."""
    sequence = []
    for param in tool_params:
        sequence.append({
            'function_name': param.get('function', ''),
            'parameters': param.get('args', {})
        })
    return sequence


def prepare_training_data(
    attack_data: List[Dict],
    benign_data: List[Dict],
    feature_extractor: EnhancedFeatureExtractor
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prepare training data with enhanced features."""

    logger.info("Building vocabularies...")

    # Build Layer 2 vocabulary first
    all_sequences = []
    attack_sequences = []
    benign_sequences = []

    for item in attack_data:
        if 'tool_parameters' in item and item['tool_parameters']:
            sequence = extract_sequence_from_tool_parameters(item['tool_parameters'])
            all_sequences.append(sequence)
            attack_sequences.append(sequence)

    for item in benign_data:
        if 'tool_parameters' in item and item['tool_parameters']:
            sequence = extract_sequence_from_tool_parameters(item['tool_parameters'])
            all_sequences.append(sequence)
            benign_sequences.append(sequence)

    feature_extractor.build_layer2_vocabulary(all_sequences)

    # Analyze suspicious patterns
    logger.info("Analyzing attack patterns...")
    feature_extractor.analyze_suspicious_patterns(attack_sequences, benign_sequences)

    # Extract features
    logger.info("Extracting features...")
    layer1_features = []
    layer2_features = []
    labels = []

    # Process attack data
    for item in tqdm(attack_data, desc="Processing attack data"):
        if 'tool_parameters' not in item or not item['tool_parameters']:
            continue

        tool_params = item['tool_parameters']
        sequence = extract_sequence_from_tool_parameters(tool_params)
        suite_name = item.get('suite_name', 'unknown')

        # Extract features from each string parameter
        for tool_call in tool_params:
            function_name = tool_call.get('function', '')
            args = tool_call.get('args', {})

            if not isinstance(args, dict):
                continue

            for param_name, param_value in args.items():
                if isinstance(param_value, str) and param_value.strip() and len(param_value.strip()) >= 3:
                    l1_feat = feature_extractor.extract_layer1_features(
                        function_name, param_value.strip(), suite_name
                    )
                    l2_feat = feature_extractor.extract_layer2_features(sequence)

                    layer1_features.append(l1_feat)
                    layer2_features.append(l2_feat)
                    labels.append(1)  # Attack

    # Process benign data
    for item in tqdm(benign_data, desc="Processing benign data"):
        if 'tool_parameters' not in item or not item['tool_parameters']:
            continue

        tool_params = item['tool_parameters']
        sequence = extract_sequence_from_tool_parameters(tool_params)
        suite_name = item.get('suite_name', 'unknown')

        for tool_call in tool_params:
            function_name = tool_call.get('function', '')
            args = tool_call.get('args', {})

            if not isinstance(args, dict):
                continue

            for param_name, param_value in args.items():
                if isinstance(param_value, str) and param_value.strip() and len(param_value.strip()) >= 3:
                    l1_feat = feature_extractor.extract_layer1_features(
                        function_name, param_value.strip(), suite_name
                    )
                    l2_feat = feature_extractor.extract_layer2_features(sequence)

                    layer1_features.append(l1_feat)
                    layer2_features.append(l2_feat)
                    labels.append(0)  # Benign

    # Convert to arrays
    layer1_features = np.array(layer1_features)
    layer2_features = np.array(layer2_features)
    labels = np.array(labels)

    if len(layer1_features) == 0:
        raise ValueError("No training samples extracted")

    # Scale features
    feature_extractor.layer1_scaler.fit(layer1_features)
    layer1_features = feature_extractor.layer1_scaler.transform(layer1_features)

    feature_extractor.layer2_scaler.fit(layer2_features)
    layer2_features = feature_extractor.layer2_scaler.transform(layer2_features)

    logger.info(f"Prepared {len(labels)} training samples")
    logger.info(f"Layer1 features shape: {layer1_features.shape}")
    logger.info(f"Layer2 features shape: {layer2_features.shape}")
    logger.info(f"Attack/Benign ratio: {np.mean(labels):.3f}")

    return layer1_features, layer2_features, labels


def train_model(
    model: EnhancedIPIDetector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001
) -> Dict[str, List[float]]:
    """Train the enhanced model."""

    device = torch.device('cpu')
    model = model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_auc': []}
    best_f1 = 0
    best_model_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0

        for batch_l1, batch_l2, batch_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            batch_l1 = batch_l1.to(device)
            batch_l2 = batch_l2.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_l1, batch_l2).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for batch_l1, batch_l2, batch_labels in val_loader:
                batch_l1 = batch_l1.to(device)
                batch_l2 = batch_l2.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_l1, batch_l2).squeeze()
                loss = criterion(outputs, batch_labels.float())
                val_loss += loss.item()

                val_preds.extend(outputs.cpu().numpy())
                val_true.extend(batch_labels.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_pred_binary = (val_preds > 0.5).astype(int)

        val_f1 = f1_score(val_true, val_pred_binary)
        val_auc = roc_auc_score(val_true, val_preds)

        scheduler.step(val_loss / len(val_loader))

        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            logger.info(f"  → New best model! F1: {best_f1:.4f}")

        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {history['train_loss'][-1]:.4f}, "
                       f"Val Loss: {history['val_loss'][-1]:.4f}, "
                       f"Val F1: {val_f1:.4f}, "
                       f"Val AUC: {val_auc:.4f}")

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Restored best model with F1: {best_f1:.4f}")

    return history


def evaluate_model(model: EnhancedIPIDetector, test_loader: DataLoader, device) -> Dict:
    """Evaluate the model on test data."""
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch_l1, batch_l2, batch_labels in test_loader:
            batch_l1 = batch_l1.to(device)
            batch_l2 = batch_l2.to(device)

            outputs = model(batch_l1, batch_l2).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_true.extend(batch_labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    test_pred_binary = (test_preds > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(test_true, test_pred_binary),
        'precision': precision_score(test_true, test_pred_binary),
        'recall': recall_score(test_true, test_pred_binary),
        'f1': f1_score(test_true, test_pred_binary),
        'auc_roc': roc_auc_score(test_true, test_preds),
    }

    logger.info("\nTest Set Performance:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")

    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(test_true, test_pred_binary)
    logger.info(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    logger.info(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced IPI Detector")
    parser.add_argument("--attack-dataset", default="/Users/yummy/IPI_Defense/dataset/attack_dataset.json")
    parser.add_argument("--benign-dataset", default="/Users/yummy/IPI_Defense/dataset/benign_dataset_fixed.json")
    parser.add_argument("--output-model", default="/Users/yummy/IPI_Defense/models/enhanced_detector.pkl")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--test-split", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.15)

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("ENHANCED UNIFIED DETECTOR TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Attack dataset: {args.attack_dataset}")
    logger.info(f"  Benign dataset: {args.benign_dataset}")
    logger.info(f"  Output model: {args.output_model}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info("="*80 + "\n")

    # Load data
    attack_data, benign_data = load_data(args.attack_dataset, args.benign_dataset)

    # Initialize enhanced feature extractor
    logger.info("Initializing enhanced feature extractor...")
    feature_extractor = EnhancedFeatureExtractor()

    # Prepare training data
    layer1_features, layer2_features, labels = prepare_training_data(
        attack_data, benign_data, feature_extractor
    )

    # Split data
    logger.info(f"Splitting data (test: {args.test_split}, val: {args.val_split})...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        np.arange(len(labels)), labels,
        test_size=args.test_split,
        random_state=42,
        stratify=labels
    )

    val_size = args.val_split / (1 - args.test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=42,
        stratify=y_temp
    )

    logger.info(f"  Train: {len(X_train)} samples")
    logger.info(f"  Val:   {len(X_val)} samples")
    logger.info(f"  Test:  {len(X_test)} samples")

    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(layer1_features[X_train]),
        torch.FloatTensor(layer2_features[X_train]),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(layer1_features[X_val]),
        torch.FloatTensor(layer2_features[X_val]),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(layer1_features[X_test]),
        torch.FloatTensor(layer2_features[X_test]),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Initialize enhanced model
    logger.info("\nInitializing enhanced model...")
    logger.info(f"  Layer 1 dim: {layer1_features.shape[1]}")
    logger.info(f"  Layer 2 dim: {layer2_features.shape[1]}")
    logger.info(f"  Vocab size: {feature_extractor.vocab_size}")

    model = EnhancedIPIDetector(
        layer1_dim=layer1_features.shape[1],
        layer2_dim=layer2_features.shape[1],
        vocab_size=feature_extractor.vocab_size,
        embedding_dim=32,
        hidden_dim=64
    )

    logger.info(f"\nModel architecture:")
    logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Train model
    logger.info("\n" + "="*80)
    logger.info("TRAINING")
    logger.info("="*80 + "\n")

    history = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )

    # Evaluate on test set
    logger.info("\n" + "="*80)
    logger.info("FINAL EVALUATION")
    logger.info("="*80)

    device = torch.device('cpu')
    test_metrics = evaluate_model(model, test_loader, device)

    # Save model
    logger.info("\n" + "="*80)
    logger.info("SAVING MODEL")
    logger.info("="*80)

    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'model_state_dict': model.state_dict(),
        'feature_extractor': feature_extractor,
        'model_config': {
            'layer1_dim': layer1_features.shape[1],
            'layer2_dim': layer2_features.shape[1],
            'vocab_size': feature_extractor.vocab_size,
            'embedding_dim': 32,
            'hidden_dim': 64
        },
        'training_history': history,
        'test_metrics': test_metrics,
        'model_type': 'enhanced'  # Mark as enhanced model
    }

    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)

    logger.info(f"✓ Model saved to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE")
    logger.info("="*80)
    logger.info(f"\nBest validation F1: {max(history['val_f1']):.4f}")
    logger.info(f"Test F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test AUC: {test_metrics['auc_roc']:.4f}")

    logger.info("\nTo use this model:")
    logger.info(f"  python scripts/test_enhanced_detector.py --model {output_path}")


if __name__ == "__main__":
    main()