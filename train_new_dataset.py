import subprocess
import sys
import os
import numpy as np
import librosa
import tensorflow as tf
from improved_model import create_improved_model, train_model, preprocess_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from collections import Counter
import glob
import gc
import random
import psutil  # For memory monitoring
import hashlib

# Ensure kagglehub is installed
try:
    import kagglehub
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub'])
    import kagglehub

# Ensure psutil is installed for memory monitoring
try:
    import psutil
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'psutil'])
    import psutil

# Use all four dataset types
# dataset_path = "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-original/for-original/training"
dataset_paths = [
    "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-2sec/for-2seconds",
    "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-norm/for-norm",
    "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-rerec/for-rerecorded",
    "/home/syed-hassan-ul-haq/.cache/kagglehub/datasets/mohammedabdeldayem/the-fake-or-real-dataset/versions/2/for-original/for-original"
]
print(f"Using dataset paths: {dataset_paths}")

# Define parameters
NUM_CLASSES = 2  # fake and real
SAMPLE_RATE = 16000
DURATION = 2  # 2 seconds for raw audio
BATCH_SIZE = 32  # Reduced batch size for better generalization and memory management

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Custom callback to log epoch info to a txt file
default_log_file = 'epoch_log.txt'
class EpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_file=default_log_file):
        super().__init__()
        self.log_file = log_file
        # Clear the log file at the start
        with open(self.log_file, 'w') as f:
            f.write('Epoch log\n')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(self.log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}: " + ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()]) + "\n")

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / 1024 / 1024 / 1024
    return memory_gb

def check_memory_safety():
    """Check if we have enough memory to continue processing."""
    memory_gb = get_memory_usage()
    print(f"Current memory usage: {memory_gb:.2f} GB")
    
    # If we're using more than 14GB, we're getting close to the limit
    if memory_gb > 14:
        print("WARNING: High memory usage detected!")
        return False
    return True

def extract_audio_features(file_path):
    """
    Extract raw audio features from a 2-second segment of an audio file.
    Returns normalized raw audio waveform.
    """
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

def train_model_with_generator(model, train_generator, val_generator, epochs=50, patience=10):
    """
    Train model using data generators for memory efficiency.
    """
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint: save every epoch with epoch number and val_accuracy in filename
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.h5',
        monitor='val_accuracy',
        save_best_only=False,  # Save every epoch
        save_freq='epoch',
        verbose=1
    )
    
    # Epoch logger callback
    epoch_logger = EpochLogger('epoch_log.txt')
    
    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        callbacks=[early_stopping, lr_scheduler, checkpoint, epoch_logger],
        verbose=1
    )
    
    return history

def load_and_preprocess_new_dataset(dataset_paths):
    """
    Load and preprocess the fake-or-real dataset with memory-efficient batch processing from multiple dataset paths.
    """
    print("Loading and preprocessing data from all dataset types...")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    
    # Look for audio files in all dataset paths
    audio_files = []
    for path in dataset_paths:
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            audio_files.extend(glob.glob(os.path.join(path, '**', ext.upper()), recursive=True))
    
    if not audio_files:
        print("No audio files found. Checking dataset structure...")
        for path in dataset_paths:
            for root, dirs, files in os.walk(path):
                print(f"Directory: {root}")
                print(f"Files: {files[:10]}...")
                break
    
    print(f"Found {len(audio_files)} audio files across all dataset types")
    
    # Process files in batches
    X = []
    y = []
    max_time_steps = 109
    processed_count = 0
    error_count = 0
    
    for batch_start in range(0, len(audio_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(audio_files))
        batch_files = audio_files[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(len(audio_files) + BATCH_SIZE - 1)//BATCH_SIZE} "
              f"({batch_start + 1}-{batch_end}/{len(audio_files)})")
        print(f"Memory usage: {get_memory_usage():.2f} GB")
        
        # Check memory safety
        if not check_memory_safety():
            print("Memory usage too high, stopping processing to prevent SIGKILL")
            break
        
        batch_X = []
        batch_y = []
        
        for file_path in batch_files:
            try:
                label = determine_label_from_path(file_path)
                try:
                    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
                except Exception as audio_error:
                    print(f"Audio loading error for {file_path}: {audio_error}")
                    error_count += 1
                    continue
                mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
                mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
                if mel_spectrogram.shape[1] < max_time_steps:
                    mel_spectrogram = np.pad(mel_spectrogram, 
                                           ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), 
                                           mode='constant')
                else:
                    mel_spectrogram = mel_spectrogram[:, :max_time_steps]
                batch_X.append(mel_spectrogram)
                batch_y.append(label)
                processed_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                error_count += 1
                continue
        X.extend(batch_X)
        y.extend(batch_y)
        del batch_X, batch_y
        gc.collect()
        print(f"Processed: {processed_count}, Errors: {error_count}")
        if (batch_start // BATCH_SIZE) % 5 == 0:
            gc.collect()
            print(f"Memory after GC: {get_memory_usage():.2f} GB")
    if not X:
        print("No valid audio files processed!")
        return None, None, None, None
    print(f"Final memory usage before conversion: {get_memory_usage():.2f} GB")
    X = np.array(X)
    y = np.array(y)
    print(f"Successfully processed {len(X)} audio files")
    print(f"Total errors: {error_count}")
    print(f"Label distribution: {Counter(y)}")
    print(f"Memory usage after numpy conversion: {get_memory_usage():.2f} GB")
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    split_index = int(0.8 * len(X))
    print('Total samples:', len(X))
    print('Train/Val split:', split_index, len(X) - split_index)
    print('Train label distribution:', Counter(y[:split_index]))
    print('Val label distribution:', Counter(y[split_index:]))
    print("Preprocessing data...")
    X = preprocess_data(X)
    y_encoded = to_categorical(y, NUM_CLASSES)
    print(f"Memory usage after preprocessing: {get_memory_usage():.2f} GB")
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y_encoded[:split_index], y_encoded[split_index:]
    
    # Debug: Check data distribution
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    # Check class distribution in training and validation
    train_class_counts = np.sum(y_train, axis=0)
    val_class_counts = np.sum(y_val, axis=0)
    print(f"Training class distribution: {train_class_counts}")
    print(f"Validation class distribution: {val_class_counts}")
    
    # Check for any NaN or infinite values
    print(f"Training data - NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"Validation data - NaN: {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
    
    return X_train, X_val, y_train, y_val

def determine_label_from_path(file_path):
    """
    Determine the label (fake/real) from the immediate parent directory.
    """
    parent_dir = os.path.basename(os.path.dirname(file_path)).lower()
    if parent_dir == 'fake':
        return 0
    elif parent_dir == 'real':
        return 1
    else:
        print(f"Warning: Could not determine label for {file_path}, parent dir: {parent_dir}. Defaulting to fake (0)")
        return 0

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig('training_history_new_dataset.png')
    plt.close()

def prepare_data_for_generators(dataset_paths):
    """
    Prepare file paths and labels for data generators without loading audio into memory, from multiple dataset paths.
    """
    print("Preparing data for generators from all dataset types...")
    print(f"Initial memory usage: {get_memory_usage():.2f} GB")
    audio_files = []
    for path in dataset_paths:
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
            audio_files.extend(glob.glob(os.path.join(path, '**', ext), recursive=True))
            audio_files.extend(glob.glob(os.path.join(path, '**', ext.upper()), recursive=True))
    if not audio_files:
        print("No audio files found!")
        return None, None
    print(f"Found {len(audio_files)} audio files across all dataset types")
    labels = []
    valid_files = []
    for i, file_path in enumerate(audio_files):
        try:
            label = determine_label_from_path(file_path)
            labels.append(label)
            valid_files.append(file_path)
        except Exception as e:
            print(f"Error determining label for {file_path}: {e}")
            continue
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(audio_files)} files for labels")
    print(f"Successfully prepared {len(valid_files)} files")
    print(f"Label distribution: {Counter(labels)}")
    print(f"Memory usage after label preparation: {get_memory_usage():.2f} GB")
    combined = list(zip(valid_files, labels))
    random.shuffle(combined)
    valid_files, labels = zip(*combined)
    print("\nSample file paths and their assigned labels:")
    for i in range(min(20, len(valid_files))):
        print(f"File: {valid_files[i]}, Label: {labels[i]}")
    print("\nLabel distribution (all data):", Counter(labels))
    split_index = int(0.8 * len(valid_files))
    train_files = valid_files[:split_index]
    train_labels = labels[:split_index]
    val_files = valid_files[split_index:]
    val_labels = labels[split_index:]
    print("Label distribution (train):", Counter(train_labels))
    print("Label distribution (val):", Counter(val_labels))
    # Overlap check by file path
    train_set = set(train_files)
    val_set = set(val_files)
    overlap = train_set & val_set
    print(f"Number of overlapping file paths between train and val: {len(overlap)}")
    if overlap:
        print("Sample overlapping files:", list(overlap)[:10])
    # Overlap check by audio content (hash)
    print("Computing audio hashes for train and val sets (this may take a while)...")
    train_hashes = set(audio_hash(f) for f in train_files)
    val_hashes = set(audio_hash(f) for f in val_files)
    hash_overlap = train_hashes & val_hashes
    print(f"Number of overlapping audio hashes between train and val: {len(hash_overlap)}")
    if hash_overlap:
        print("Sample overlapping audio hashes:", list(hash_overlap)[:5])
    return list(valid_files), list(labels)

def audio_hash(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        return hashlib.md5(audio.tobytes()).hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return None

def prepare_balanced_train_test_sets(dataset_base_paths, train_per_type=3000, test_per_type=750):
    """
    Prepare balanced training and testing sets from each dataset type.
    Ensures perfect balance between 'fake' and 'real' samples for each type and split.
    Selects train_per_type from each type's training set and test_per_type from each type's testing set.
    """
    print("Preparing balanced train/test sets from each dataset type (with perfect fake/real balance)...")
    type_names = [
        "for-2sec",
        "for-norm",
        "for-original",
        "for-rerec"
    ]
    train_files, train_labels = [], []
    test_files, test_labels = [], []
    for type_name in type_names:
        # Find the base path for this type
        type_path = None
        for base_path in dataset_base_paths:
            if type_name in base_path:
                type_path = base_path
                break
        if not type_path:
            print(f"Warning: Could not find base path for type {type_name}")
            continue
        for split, per_type, files_list, labels_list in [
            ("training", train_per_type, train_files, train_labels),
            ("testing", test_per_type, test_files, test_labels)
        ]:
            split_dir = os.path.join(type_path, split)
            fake_dir = os.path.join(split_dir, 'fake')
            real_dir = os.path.join(split_dir, 'real')
            print(f"Looking in: {fake_dir}")
            print(f"Looking in: {real_dir}")
            fake_files, real_files = [], []
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
                fake_files_found = glob.glob(os.path.join(fake_dir, '**', ext), recursive=True)
                fake_files_found += glob.glob(os.path.join(fake_dir, '**', ext.upper()), recursive=True)
                real_files_found = glob.glob(os.path.join(real_dir, '**', ext), recursive=True)
                real_files_found += glob.glob(os.path.join(real_dir, '**', ext.upper()), recursive=True)
                fake_files.extend(fake_files_found)
                real_files.extend(real_files_found)
            print(f"Found {len(fake_files)} fake and {len(real_files)} real files in {split_dir}")
            n_each = per_type // 2
            if len(fake_files) < n_each:
                print(f"Warning: Not enough fake files for {type_name}/{split} (needed {n_each}, found {len(fake_files)})")
                sampled_fake = fake_files
            else:
                sampled_fake = random.sample(fake_files, n_each)
            if len(real_files) < n_each:
                print(f"Warning: Not enough real files for {type_name}/{split} (needed {n_each}, found {len(real_files)})")
                sampled_real = real_files
            else:
                sampled_real = random.sample(real_files, n_each)
            files = sampled_fake + sampled_real
            labels = [0]*len(sampled_fake) + [1]*len(sampled_real)
            combined = list(zip(files, labels))
            random.shuffle(combined)
            files, labels = zip(*combined) if combined else ([],[])
            files_list.extend(files)
            labels_list.extend(labels)
    # Shuffle final lists
    train_combined = list(zip(train_files, train_labels))
    random.shuffle(train_combined)
    train_files, train_labels = zip(*train_combined) if train_combined else ([],[])
    test_combined = list(zip(test_files, test_labels))
    random.shuffle(test_combined)
    test_files, test_labels = zip(*test_combined) if test_combined else ([],[])
    print(f"Total training files: {len(train_files)}")
    print(f"Total testing files: {len(test_files)}")
    print(f"Train label distribution: {Counter(train_labels)}")
    print(f"Test label distribution: {Counter(test_labels)}")
    print("Sample training files and labels:")
    for i in range(min(10, len(train_files))):
        print(f"{train_files[i]} -> {train_labels[i]}")
    print("Sample testing files and labels:")
    for i in range(min(10, len(test_files))):
        print(f"{test_files[i]} -> {test_labels[i]}")
    return list(train_files), list(train_labels), list(test_files), list(test_labels)

def preprocess_data(X):
    """
    Preprocess data with proper normalization and data augmentation.
    """
    # Normalize each sample independently to prevent data leakage
    X_normalized = np.zeros_like(X)
    for i in range(X.shape[0]):
        sample = X[i]
        # Normalize to zero mean and unit variance
        sample_mean = np.mean(sample)
        sample_std = np.std(sample)
        if sample_std > 0:
            X_normalized[i] = (sample - sample_mean) / sample_std
        else:
            X_normalized[i] = sample - sample_mean
    
    return X_normalized

def augment_data(X, y, augmentation_factor=0.3):
    """
    Apply data augmentation to reduce overfitting.
    augmentation_factor: fraction of samples to augment
    """
    if augmentation_factor <= 0:
        return X, y
    
    n_samples = X.shape[0]
    n_augment = int(n_samples * augmentation_factor)
    
    # Randomly select samples to augment
    indices = np.random.choice(n_samples, n_augment, replace=False)
    
    X_augmented = []
    y_augmented = []
    
    for idx in indices:
        sample = X[idx]
        label = y[idx]
        
        # Apply random noise
        noise_factor = 0.01
        noise = np.random.normal(0, noise_factor, sample.shape)
        augmented_sample = sample + noise
        
        # Apply random time shift (horizontal shift) - handle 3D arrays
        shift = np.random.randint(-5, 6)  # Shift by -5 to +5 time steps
        if shift > 0:
            # For 3D array: pad along the time dimension (axis=1)
            augmented_sample = np.pad(augmented_sample, 
                                    ((0, 0), (0, shift), (0, 0)), 
                                    mode='edge')[:, shift:, :]
        elif shift < 0:
            # For 3D array: pad along the time dimension (axis=1)
            augmented_sample = np.pad(augmented_sample, 
                                    ((0, 0), (-shift, 0), (0, 0)), 
                                    mode='edge')[:, :shift, :]
        
        X_augmented.append(augmented_sample)
        y_augmented.append(label)
    
    # Combine original and augmented data
    X_combined = np.concatenate([X, np.array(X_augmented)], axis=0)
    y_combined = np.concatenate([y, np.array(y_augmented)], axis=0)
    
    # Shuffle the combined data
    indices = np.arange(len(X_combined))
    np.random.shuffle(indices)
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    print(f"Data augmentation: {n_samples} -> {len(X_combined)} samples")
    return X_combined, y_combined

def main():
    print("Standard training with strict train/test split and perfect balance...")
    # Remove .npy/.csv feature loading and saving logic
    train_files, train_labels, test_files, test_labels = prepare_balanced_train_test_sets(dataset_paths, train_per_type=3000, test_per_type=750)
    if not train_files or not test_files:
        print("Failed to load dataset. Please check the dataset structure.")
        return
    
    # Load all training data into memory
    print("Loading and preprocessing all training data into memory...")
    X_train = []
    train_extraction_failures = 0
    for file_path, label in zip(train_files, train_labels):
        audio = extract_audio_features(file_path)
        if audio is not None:
            X_train.append(audio)
        else:
            train_extraction_failures += 1
    X_train = np.array(X_train)
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension for 1D CNN
    y_train = to_categorical(train_labels, NUM_CLASSES)
    print(f"Loaded {len(X_train)} training samples. Feature extraction failures: {train_extraction_failures}")

    print("Loading and preprocessing all validation/testing data into memory...")
    X_val = []
    val_extraction_failures = 0
    for file_path, label in zip(test_files, test_labels):
        audio = extract_audio_features(file_path)
        if audio is not None:
            X_val.append(audio)
        else:
            val_extraction_failures += 1
    X_val = np.array(X_val)
    X_val = np.expand_dims(X_val, axis=-1)  # Add channel dimension for 1D CNN
    y_val = to_categorical(test_labels, NUM_CLASSES)
    print(f"Loaded {len(X_val)} validation/testing samples. Feature extraction failures: {val_extraction_failures}")
    
    # Debug: Check data distribution
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Validation labels shape: {y_val.shape}")
    
    # Check class distribution in training and validation
    train_class_counts = np.sum(y_train, axis=0)
    val_class_counts = np.sum(y_val, axis=0)
    print(f"Training class distribution: {train_class_counts}")
    print(f"Validation class distribution: {val_class_counts}")
    
    # Check for any NaN or infinite values
    print(f"Training data - NaN: {np.isnan(X_train).sum()}, Inf: {np.isinf(X_train).sum()}")
    print(f"Validation data - NaN: {np.isnan(X_val).sum()}, Inf: {np.isinf(X_val).sum()}")
    
    # Apply data augmentation to training data only
    print("Applying data augmentation to training data...")
    X_train, y_train = augment_data(X_train, y_train, augmentation_factor=0)  # Reduced from 0.3
    
    # Create and train 1D CNN model
    input_length = X_train.shape[1]  # Raw audio length
    print(f"Building 1D CNN classifier for raw audio (input length: {input_length})...")
    model = create_1d_cnn_classifier(input_length, NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    print("Training model...")
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning rate scheduler
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    
    # Model checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'model_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.h5',
        monitor='val_accuracy',
        save_best_only=False,
        save_freq='epoch',
        verbose=1
    )
    
    # Epoch logger callback
    epoch_logger = EpochLogger('epoch_log.txt')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, lr_scheduler, checkpoint, epoch_logger],
        verbose=1
    )
    
    plot_training_history(history)
    model.save('fake_or_real_classifier.h5')
    print("Model saved as 'fake_or_real_classifier.h5'")
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_accuracy:.4f}")

    # Print first 10 predictions and true labels for validation set
    preds = model.predict(X_val[:10])
    print("First 10 model predictions (softmax outputs):", preds)
    print("First 10 predicted classes:", np.argmax(preds, axis=1))
    print("First 10 true classes:", np.argmax(y_val[:10], axis=1))

if __name__ == "__main__":
    main() 