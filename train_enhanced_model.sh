#!/bin/bash
# Quick start script to train the enhanced detector

echo "=========================================================================="
echo "ENHANCED UNIFIED DETECTOR - TRAINING"
echo "=========================================================================="
echo ""
echo "This script will train the enhanced detector with:"
echo "  ✓ Position-aware temporal features"
echo "  ✓ Attention-based pooling"
echo "  ✓ Transition pattern encoding"
echo "  ✓ Escalation detection"
echo ""
echo "=========================================================================="
echo ""

# Check if datasets exist
if [ ! -f "dataset/attack_dataset.json" ]; then
    echo "❌ Error: dataset/attack_dataset.json not found"
    exit 1
fi

if [ ! -f "dataset/benign_dataset_fixed.json" ]; then
    echo "❌ Error: dataset/benign_dataset_fixed.json not found"
    exit 1
fi

echo "✓ Training datasets found"
echo ""

# Create models directory if needed
mkdir -p models

# Ask for confirmation
read -p "Start training? (This may take 30-60 minutes) [y/N]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

echo ""
echo "=========================================================================="
echo "STARTING TRAINING"
echo "=========================================================================="
echo ""

# Train the model
python scripts/train_enhanced_detector.py \
    --attack-dataset dataset/attack_dataset.json \
    --benign-dataset dataset/benign_dataset_fixed.json \
    --output-model models/enhanced_detector.pkl \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================================================="
    echo "✓ TRAINING COMPLETE"
    echo "=========================================================================="
    echo ""
    echo "Model saved to: models/enhanced_detector.pkl"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model:"
    echo "     python enhanced_detector_wrapper.py"
    echo ""
    echo "  2. Compare with original model on failing tasks:"
    echo "     python diagnose_failing_tasks.py --detailed-analysis"
    echo ""
    echo "  3. Run full AgentDojo benchmark:"
    echo "     python threshold_sensitivity_benchmark.py --model enhanced_detector.pkl"
    echo ""
else
    echo ""
    echo "=========================================================================="
    echo "❌ TRAINING FAILED"
    echo "=========================================================================="
    echo ""
    echo "Check the error messages above for details"
    exit 1
fi