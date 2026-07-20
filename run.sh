#!/usr/bin/env bash
set -euo pipefail

echo "=== PMC-VQA Pipeline ==="

echo ""
echo "Installing dependencies..."
pip install -r requirements.txt -q

echo ""
echo "Step 1: Exploratory Data Analysis..."
python src/eda.py

echo ""
echo "Step 2: Training..."
python src/train.py

echo ""
echo "Step 3: Evaluation + Explainable AI..."
python src/eval.py --checkpoint checkpoints/best.pt --explain --num_explain 10

echo ""
echo "Done! All outputs in outputs/"
