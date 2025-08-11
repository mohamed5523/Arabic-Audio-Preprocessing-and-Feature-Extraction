"""
Utility functions for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import glob
import librosa
import matplotlib.pyplot as plt
import librosa.display
import warnings
from config.config import Config

warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  # Use non-interactive backend for plots

def find_audio_files(input_folder):
    """Find all audio files in the folder structure"""
    audio_files = []
    
    for ext in Config.AUDIO_EXTENSIONS:
        pattern = os.path.join(input_folder, '**', ext)
        found_files = glob.glob(pattern, recursive=True)
        audio_files.extend(found_files)
    
    # Filter out empty or corrupted files
    valid_audio_files = []
    for audio_file in audio_files:
        try:
            # Quick check if file can be loaded
            y, sr = librosa.load(audio_file, sr=None, duration=0.1)
            if len(y) > 0:
                valid_audio_files.append(audio_file)
            else:
                print(f"⚠️ Skipping empty audio file: {os.path.basename(audio_file)}")
        except Exception as e:
            print(f"⚠️ Skipping corrupted audio file: {os.path.basename(audio_file)} - {str(e)}")
    
    return valid_audio_files

def create_output_structure(audio_path, input_folder, output_folder):
    """Create output folder structure matching input structure"""
    # Get relative path from input folder
    rel_path = os.path.relpath(audio_path, input_folder)
    rel_dir = os.path.dirname(rel_path)
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    # Create output folder maintaining the same structure
    if rel_dir and rel_dir != '.':
        output_folder_path = os.path.join(output_folder, rel_dir, audio_name)
    else:
        output_folder_path = os.path.join(output_folder, audio_name)
    
    os.makedirs(output_folder_path, exist_ok=True)
    return output_folder_path

def save_waveform_plot(y, sr, title, output_path, duration_limit=None):
    """Save waveform plot as PNG"""
    if duration_limit is None:
        duration_limit = Config.PLOT_DURATION_LIMIT
        
    plt.figure(figsize=Config.PLOT_FIGSIZE)
    
    # Limit duration for better visualization
    if len(y) > duration_limit * sr:
        y_plot = y[:int(duration_limit * sr)]
        title += f" (First {duration_limit}s)"
    else:
        y_plot = y
    
    librosa.display.waveshow(y_plot, sr=sr, alpha=0.8)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=Config.PLOT_DPI, bbox_inches='tight')
    plt.close()  # Important to free memory

def validate_audio_file(audio_path):
    """Validate if audio file exists and can be loaded"""
    if not os.path.exists(audio_path):
        return False, f"Audio file not found: {audio_path}"
    
    try:
        y_test, sr_test = librosa.load(audio_path, sr=None, duration=0.1)
        if len(y_test) == 0:
            return False, f"Empty audio file: {audio_path}"
        return True, "Valid audio file"
    except Exception as e:
        return False, f"Cannot load audio file {audio_path}: {str(e)}"

def get_audio_name(audio_path):
    """Get audio name without extension"""
    return os.path.splitext(os.path.basename(audio_path))[0]

def print_processing_summary(successful, failed, output_folder, milvus_host, milvus_port):
    """Print processing summary"""
    print(f"\n📊 Processing Summary:")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed to process: {failed} files")
    print(f"📁 Output folder: {output_folder}")
    print(f"🗄️ Milvus host: {milvus_host}:{milvus_port}")

def print_collection_stats(embedding_count, logmel_count):
    """Print Milvus collection statistics"""
    print(f"\n📈 Milvus Collection Statistics:")
    print(f"   🎤 Speaker Embeddings: {embedding_count} records")
    print(f"   📊 Log-Mel Features: {logmel_count} records")