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
echo "Step 2: Training (includes validation, test evaluation, and XAI)..."
python src/train.py

echo ""
echo "Done! All outputs in outputs/"
