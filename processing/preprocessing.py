"""
Audio preprocessing functions for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import librosa
import noisereduce as nr
import soundfile as sf
from utils.utils import save_waveform_plot
from config.config import Config

def preprocess_audio(audio_path, output_folder):
    """Step 1: Audio preprocessing (denoising + resampling)"""
    # Load audio
    y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
    
    # Save original waveform plot
    original_plot_path = os.path.join(output_folder, Config.ORIGINAL_PLOT_FILENAME)
    save_waveform_plot(y, sr, "Original Audio Waveform", original_plot_path)
    
    # Apply noise reduction
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # Save denoised waveform plot
    denoised_plot_path = os.path.join(output_folder, Config.DENOISED_PLOT_FILENAME)
    save_waveform_plot(y_denoised, sr, "Denoised Audio Waveform", denoised_plot_path)
    
    # Save denoised audio
    denoised_path = os.path.join(output_folder, Config.DENOISED_AUDIO_FILENAME)
    sf.write(denoised_path, y_denoised, sr)
    
    # Save original audio copy
    original_path = os.path.join(output_folder, Config.ORIGINAL_AUDIO_FILENAME)
    sf.write(original_path, y, sr)
    
    return denoised_path, y_denoised, sr