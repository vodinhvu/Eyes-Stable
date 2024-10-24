#!/bin/bash

# Download the dataset
echo "Downloading dataset..."
# Replace with the actual URL of your dataset
DATASET_URL="http://example.com/path/to/your/dataset.zip"
DATASET_DIR="./data"

# Create data directory if it doesn't exist
mkdir -p $DATASET_DIR

# Download the dataset
curl -L $DATASET_URL -o "$DATASET_DIR/dataset.zip"

# Unzip the dataset
unzip "$DATASET_DIR/dataset.zip" -d $DATASET_DIR

# Navigate to the src directory
cd src

# Run main.py with arguments
echo "Running main.py..."
python main.py --image "$DATASET_DIR/hymenoptera_data" --epochs 3 --save "../models" --batch 4 --num_workers 2
