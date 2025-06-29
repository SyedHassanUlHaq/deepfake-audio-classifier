# Audio Deepfake Detection Pipeline

## Overview
This project provides a robust pipeline for detecting deepfake (synthetic) audio using a deep learning approach. The current system leverages raw audio waveform input and a 1D Convolutional Neural Network (1D CNN) to classify audio as either fake or real. The pipeline is designed for strict data separation, reproducibility, and extensibility.

## Features
- **Raw waveform input**: No hand-crafted features; the model learns directly from the audio signal.
- **1D CNN architecture**: Deep convolutional network for effective feature extraction from raw audio.
- **Strict train/validation/test separation**: Prevents data leakage by checking both file paths and audio content hashes.
- **Balanced sampling**: Ensures equal representation of fake and real samples in both training and testing.
- **Flexible data loading**: Supports multiple dataset types and common audio formats (`.wav`, `.mp3`, `.flac`, `.m4a`).
- **Model checkpointing and logging**: Saves models at each epoch and logs training progress.
- **Comprehensive model testing script**: Compare all saved models on any directory of audio files.

## Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd audio-deepfake-detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation
- Download and extract the Fake-or-Real dataset (or your own dataset) into the appropriate directory structure:
  ```
  dataset_root/
    for-2sec/
      training/fake/
      training/real/
      testing/fake/
      testing/real/
    for-norm/...
    for-original/...
    for-rerec/...
  ```
- The pipeline will automatically find and balance samples from each type and split.

## Training
Run the main training script:
```bash
python3 train_new_dataset.py
```
- The script will:
  - Load and preprocess all audio files (2 seconds, 16kHz, mono, normalized).
  - Assign labels based on parent directory (`fake`=0, `real`=1).
  - Strictly separate training and validation/testing sets (no overlap by path or content).
  - Train a 1D CNN on the raw waveform input.
  - Save model checkpoints after each epoch (e.g., `model_epoch_01_valacc_0.7240.h5`).
  - Save the final model as `fake_or_real_classifier.h5`.
  - Log training history and save accuracy/loss plots.

## Model Architecture
- **Input**: Raw audio waveform, 2 seconds at 16kHz (shape: `(32000, 1)`).
- **Layers**:
  - Multiple Conv1D + MaxPooling1D layers (32, 64, 128 filters).
  - Flatten, Dense(128), Dropout, Dense(64), Dropout.
  - Output: Dense(2, softmax) for binary classification.
- **Loss**: Categorical cross-entropy.
- **Optimizer**: Adam.

## Model Testing and Comparison
After training, you can evaluate all saved models on any directory of audio files using the provided script:
```bash
python3 test_all_models.py --audio_dir /path/to/test_audio --output_dir model_test_results
```
- The script will:
  - Automatically find all epoch and final models.
  - Preprocess each audio file as during training.
  - Output a summary of predictions for each model (fake/real counts, average confidence).
  - Highlight files where models disagree.
  - Save detailed results to a timestamped CSV file for further analysis.

## Best Practices
- **Use a truly independent test set** to evaluate generalization.
- **Inspect files with inconsistent predictions** to identify ambiguous or challenging samples.
- **Tune data augmentation and model hyperparameters** for improved robustness.
- **Extend the pipeline** with new architectures or ensembling for further gains.

## References
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Keras Documentation](https://keras.io/)
- [Fake-or-Real Dataset on Kaggle](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

---
For a detailed step-by-step description, see `pipeline_overview.txt`.
