#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "--- Shell Script Start ---"

if [ -d "venv" ]; then
    echo "--- Activate Virtual Environment ---"
    source venv/bin/activate
fi

echo "--- Run Inference ---"
python inference.py "$@" --output_path predictions.csv

if [ -d "venv" ]; then
    echo "--- Deactivate Virtual Environment ---"
    deactivate
fi

echo "--- Shell Script Stop ---"