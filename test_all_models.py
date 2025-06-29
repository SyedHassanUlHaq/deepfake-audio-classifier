import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import argparse
from pathlib import Path
import glob
import pandas as pd
from datetime import datetime

def preprocess_audio(file_path, sample_rate=16000, duration=2):
    """
    Preprocess a single audio file for prediction using raw audio waveform.
    Matches the preprocessing used in training.
    """
    try:
        # Load audio and resample to 16kHz if needed
        audio, sr = librosa.load(file_path, sr=None)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        
        # Ensure audio is mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Pad or crop to target duration
        target_length = int(sample_rate * duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            audio = audio[:target_length]
        
        # Normalize audio (same as training)
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Apply the same normalization as in training (zero mean, unit variance)
        sample_mean = np.mean(audio)
        sample_std = np.std(audio)
        if sample_std > 0:
            audio = (audio - sample_mean) / sample_std
        else:
            audio = audio - sample_mean
        
        # Add channel dimension for 1D CNN
        audio = audio.reshape(1, -1, 1)
        
        return audio
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def predict_audio(model, file_path, class_names=['spoof', 'bonafide']):
    """
    Make prediction on a single audio file.
    """
    # Preprocess the audio
    processed_audio = preprocess_audio(file_path)
    
    if processed_audio is None:
        return None
    
    # Make prediction
    prediction = model.predict(processed_audio, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    return {
        'file': file_path,
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            'spoof': prediction[0][0],
            'bonafide': prediction[0][1]
        }
    }

def test_directory_with_model(model, model_name, directory_path, supported_formats=['.wav', '.mp3', '.flac', '.m4a']):
    """
    Test all audio files in a directory with a specific model.
    """
    results = []
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Directory {directory_path} does not exist!")
        return results
    
    # Find all audio files
    audio_files = []
    for format_ext in supported_formats:
        audio_files.extend(directory.glob(f"*{format_ext}"))
        audio_files.extend(directory.glob(f"*{format_ext.upper()}"))
    
    if not audio_files:
        print(f"No audio files found in {directory_path}")
        print(f"Supported formats: {supported_formats}")
        return results
    
    print(f"Testing {model_name} on {len(audio_files)} audio files...")
    
    # Process each file
    for i, audio_file in enumerate(audio_files, 1):
        result = predict_audio(model, str(audio_file))
        if result:
            result['model_name'] = model_name
            results.append(result)
    
    return results

def find_model_files():
    """
    Find all epoch models and the final model.
    """
    models = {}
    
    # Find epoch models
    epoch_models = glob.glob("model_epoch_*.h5")
    for model_path in epoch_models:
        # Extract epoch number and validation accuracy from filename
        filename = os.path.basename(model_path)
        parts = filename.replace('.h5', '').split('_')
        epoch_num = int(parts[2])
        val_acc = float(parts[4])
        models[model_path] = {
            'type': 'epoch',
            'epoch': epoch_num,
            'val_acc': val_acc,
            'name': f"Epoch {epoch_num} (val_acc: {val_acc:.4f})"
        }
    
    # Find final model
    final_models = ['fake_or_real_classifier.h5']
    for final_model in final_models:
        if os.path.exists(final_model):
            models[final_model] = {
                'type': 'final',
                'epoch': 'final',
                'val_acc': 'N/A',
                'name': f"Final Model ({final_model})"
            }
    
    return models

def analyze_results(all_results, audio_dir):
    """
    Analyze and compare results across all models.
    """
    if not all_results:
        print("No results to analyze.")
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("="*80)
    
    # Create summary dataframe
    summary_data = []
    
    for model_name, results in all_results.items():
        if not results:
            continue
            
        # Count predictions
        spoof_count = sum(1 for r in results if r['predicted_class'] == 'spoof')
        bonafide_count = sum(1 for r in results if r['predicted_class'] == 'bonafide')
        total_files = len(results)
        
        # Calculate statistics
        spoof_percentage = (spoof_count / total_files) * 100 if total_files > 0 else 0
        bonafide_percentage = (bonafide_count / total_files) * 100 if total_files > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in results]) if results else 0
        
        summary_data.append({
            'Model': model_name,
            'Total Files': total_files,
            'Spoof Count': spoof_count,
            'Bonafide Count': bonafide_count,
            'Spoof %': f"{spoof_percentage:.1f}%",
            'Bonafide %': f"{bonafide_percentage:.1f}%",
            'Avg Confidence': f"{avg_confidence:.3f}"
        })
    
    # Create and display summary table
    df = pd.DataFrame(summary_data)
    print("\nModel Performance Summary:")
    print("-" * 80)
    print(df.to_string(index=False))
    
    # Find models with most different predictions
    print("\n" + "="*80)
    print("MODEL PREDICTION CONSISTENCY ANALYSIS")
    print("="*80)
    
    if len(all_results) > 1:
        # Get all unique files
        all_files = set()
        for results in all_results.values():
            all_files.update([r['file'] for r in results])
        
        # Check prediction consistency for each file
        inconsistent_files = []
        for file_path in all_files:
            predictions = []
            for model_name, results in all_results.items():
                for result in results:
                    if result['file'] == file_path:
                        predictions.append((model_name, result['predicted_class'], result['confidence']))
                        break
            
            if len(predictions) > 1:
                # Check if all predictions are the same
                first_pred = predictions[0][1]
                if not all(pred[1] == first_pred for pred in predictions):
                    inconsistent_files.append((file_path, predictions))
        
        print(f"\nFiles with inconsistent predictions across models: {len(inconsistent_files)}")
        if inconsistent_files:
            print("\nSample of inconsistent predictions:")
            for i, (file_path, predictions) in enumerate(inconsistent_files[:10]):  # Show first 10
                filename = Path(file_path).name
                print(f"\n{filename}:")
                for model_name, pred_class, confidence in predictions:
                    print(f"  {model_name}: {pred_class} (conf: {confidence:.3f})")
            
            if len(inconsistent_files) > 10:
                print(f"\n... and {len(inconsistent_files) - 10} more files with inconsistent predictions")
    
    # Save detailed results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"model_comparison_results_{timestamp}.csv"
    
    # Flatten results for CSV
    csv_data = []
    for model_name, results in all_results.items():
        for result in results:
            csv_data.append({
                'Model': model_name,
                'File': Path(result['file']).name,
                'Predicted_Class': result['predicted_class'],
                'Confidence': result['confidence'],
                'Spoof_Probability': result['probabilities']['spoof'],
                'Bonafide_Probability': result['probabilities']['bonafide']
            })
    
    df_detailed = pd.DataFrame(csv_data)
    df_detailed.to_csv(csv_filename, index=False)
    print(f"\nDetailed results saved to: {csv_filename}")

def main():
    parser = argparse.ArgumentParser(description='Test all epoch models and final model on audio directory')
    parser.add_argument('--audio_dir', type=str, required=True,
                       help='Directory containing audio files to test')
    parser.add_argument('--output_dir', type=str, default='model_test_results',
                       help='Directory to save detailed results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all models
    print("Finding model files...")
    models = find_model_files()
    
    if not models:
        print("No model files found!")
        return
    
    print(f"Found {len(models)} models:")
    for model_path, info in models.items():
        print(f"  - {model_path}: {info['name']}")
    
    # Test each model
    all_results = {}
    
    for model_path, model_info in models.items():
        print(f"\n{'='*60}")
        print(f"Testing {model_info['name']}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print(f"Loading model: {model_path}")
            model = load_model(model_path)
            print("Model loaded successfully!")
            
            # Test on directory
            results = test_directory_with_model(model, model_info['name'], args.audio_dir)
            all_results[model_info['name']] = results
            
            # Print quick summary for this model
            if results:
                spoof_count = sum(1 for r in results if r['predicted_class'] == 'spoof')
                bonafide_count = sum(1 for r in results if r['predicted_class'] == 'bonafide')
                avg_conf = np.mean([r['confidence'] for r in results])
                
                print(f"Results: {spoof_count} spoof, {bonafide_count} bonafide")
                print(f"Average confidence: {avg_conf:.3f}")
            
        except Exception as e:
            print(f"Error testing {model_path}: {e}")
            continue
    
    # Analyze and compare all results
    analyze_results(all_results, args.audio_dir)
    
    print(f"\nAll results have been saved to the '{args.output_dir}' directory.")

if __name__ == "__main__":
    main() 