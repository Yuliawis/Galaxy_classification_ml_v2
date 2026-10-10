# Galaxy Morphological Classification (Elliptical vs Spiral)

Scientific research project for binary morphological classification of galaxies into **Elliptical** and **Spiral** types using convolutional neural networks and transfer learning.

## Overview

This repository contains a machine learning pipeline that classifies galaxy images from the Sloan Digital Sky Survey (SDSS) into two morphological classes:

- **ELLIPTICAL** (0)
- **SPIRAL** (1)

Labels are derived from citizen-science vote (GalaxyZoo) fractions (SPIRAL / ELLIPTICAL / UNCERTAIN). The model uses **MobileNetV2** pretrained on ImageNet with transfer learning and fine-tuning.

## Repository Structure

| File | Description |
|------|-------------|
| `try3.py` | Main training script (data matching, model training, evaluation) |
| `scriptCasJobs.txt` | SDSS CasJobs SQL query to attach spectroscopic redshifts |
| `script_mag.txt` | SDSS CasJobs SQL query to attach photometric magnitudes |
| `predicted_galaxy_types.csv` | Model predictions on unseen SDSS cutouts (OBJID, filename, class, confidence) |

## Model Architecture

- **Backbone**: MobileNetV2 (`include_top=False`, ImageNet weights)
- **Head**: GlobalAveragePooling2D → Dropout(0.3) → Dense(1, sigmoid)
- **Input size**: 224×224 RGB
- **Loss**: Binary cross-entropy
- **Metrics**: Binary Accuracy, AUC

### Training Procedure

1. Freeze the MobileNetV2 base and train the head (15 epochs, Adam lr=1e-3)
2. Unfreeze the top ~40 layers of the backbone and fine-tune (8 epochs, Adam lr=1e-4)
3. Callbacks: ModelCheckpoint (best val AUC), EarlyStopping, ReduceLROnPlateau
4. Data augmentation: RandomFlip, RandomRotation, RandomZoom, RandomContrast

Train/val/test split is stratified (≈65% / 15% / 20%).

## Data Pipeline

The script matches galaxy images to labels by **OBJID**:

1. Recursively indexes image files under `training_images/ELLIPTICAL/` and `training_images/SPIRAL/`
2. Extracts long numeric IDs (≥8 digits) from filenames
3. Reads `MyCoords_training.csv` containing columns: `OBJID`, `SPIRAL`, `ELLIPTICAL`, `UNCERTAIN`
4. Assigns the class with the highest vote fraction; drops uncertain cases and ties
