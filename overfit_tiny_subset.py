import os
import numpy as np
import random
import librosa
import tensorflow as tf
from improved_model import create_improved_model, train_model, preprocess_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten

# Parameters
SAMPLE_RATE = 16000
DURATION = 2  # Back to 2 seconds for raw waveform
NUM_CLASSES = 2
BATCH_SIZE = 4
EPOCHS = 50

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Tiny subset size
N_PER_CLASS = 10

# Add this flag to toggle between toy and real data
USE_TOY_DATA = False  # Set to False to use real audio data

def extract_raw_audio_features(file_path):
    try:
        # Load audio and resample to 16kHz if needed
        audio, sr = librosa.load(file_path, sr=None)
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Pad or crop to target duration
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio
    except Exception as e:
        print(f"Error extracting raw audio from {file_path}: {e}")
        return None

def create_1d_cnn_classifier(input_length, num_classes):
    """Create a 1D CNN classifier for raw audio"""
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_length, 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

def get_tiny_subset(dataset_paths, n_per_class=10):
    fake_files, real_files = [], []
    for path in dataset_paths:
        for split in ['training', 'testing']:
            for label in ['fake', 'real']:
                dir_path = os.path.join(path, split, label)
                for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                    files = glob.glob(os.path.join(dir_path, '**', ext), recursive=True)
                    files += glob.glob(os.path.join(dir_path, '**', ext.upper()), recursive=True)
                    if label == 'fake':
                        fake_files.extend(files)
                    else:
                        real_files.extend(files)
    random.shuffle(fake_files)
    random.shuffle(real_files)
    return fake_files[:n_per_class], real_files[:n_per_class]

if __name__ == "__main__":
    if USE_TOY_DATA:
        print("Running toy dataset experiment...")
        # Toy dataset parameters
        N_SAMPLES = 20
        AUDIO_LENGTH = 32000  # 2 seconds at 16kHz
        X = np.zeros((N_SAMPLES, AUDIO_LENGTH, 1))
        y = np.zeros((N_SAMPLES,))
        # Class 0: random normal centered at -0.5
        for i in range(N_SAMPLES // 2):
            X[i, :, 0] = np.random.normal(loc=-0.5, scale=0.3, size=AUDIO_LENGTH)
            y[i] = 0
        # Class 1: random normal centered at +0.5
        for i in range(N_SAMPLES // 2, N_SAMPLES):
            X[i, :, 0] = np.random.normal(loc=0.5, scale=0.3, size=AUDIO_LENGTH)
            y[i] = 1
        y = to_categorical(y, NUM_CLASSES)
        # Shuffle
        idx = np.arange(N_SAMPLES)
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        # Split 80/20
        split = int(0.8 * N_SAMPLES)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        print(f"Toy Train: {X_train.shape}, Toy Val: {X_val.shape}")
        # Model
        input_length = AUDIO_LENGTH
        print("Building 1D CNN classifier for toy data...")
        model = create_1d_cnn_classifier(input_length, NUM_CLASSES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        print("Training on toy dataset...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=4,
            verbose=2
        )
        print("Final toy train acc:", history.history['accuracy'][-1])
        print("Final toy val acc:", history.history['val_accuracy'][-1])
        preds = model.predict(X_val)
        print("Toy predictions (softmax):", preds)
        print("Toy predicted classes:", np.argmax(preds, axis=1))
        print("Toy true classes:", np.argmax(y_val, axis=1))
    else:
        # Use the same dataset_paths as train_new_dataset.py
        dataset_paths = [
            "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-2sec/for-2seconds",
            "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-norm/for-norm",
            "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-rerec/for-rerecorded",
            "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-original/for-original"
        ]
        import glob
        print("Collecting tiny subset...")
        fake_files, real_files = get_tiny_subset(dataset_paths, N_PER_CLASS)
        files = fake_files + real_files
        labels = [0]*len(fake_files) + [1]*len(real_files)
        print(f"Fake: {len(fake_files)}, Real: {len(real_files)}")
        print("Extracting raw audio features...")
        X, y = [], []
        for file_path, label in zip(files, labels):
            audio = extract_raw_audio_features(file_path)
            if audio is not None:
                print(f"Extracted audio shape: {audio.shape} for {os.path.basename(file_path)}")
                X.append(audio)
                y.append(label)
            else:
                print(f"Failed to extract audio for {os.path.basename(file_path)}")
        X = np.array(X)
        y = np.array(y)
        X = np.expand_dims(X, axis=-1)  # Add channel dimension for 1D CNN
        y = to_categorical(y, NUM_CLASSES)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        if len(X) == 0:
            print("No audio features extracted successfully. Exiting.")
            import sys
            sys.exit(1)
        # Shuffle
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X, y = X[idx], y[idx]
        # Split 80/20
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        # Model
        input_length = X_train.shape[1]  # Raw audio features are 1D
        print("Building 1D CNN classifier for raw audio...")
        model = create_1d_cnn_classifier(input_length, NUM_CLASSES)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()
        print("Training on tiny subset...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )
        print("Final train acc:", history.history['accuracy'][-1])
        print("Final val acc:", history.history['val_accuracy'][-1])
        # Print predictions
        preds = model.predict(X_val)
        print("Predictions (softmax):", preds)
        print("Predicted classes:", np.argmax(preds, axis=1))
        print("True classes:", np.argmax(y_val, axis=1)) 