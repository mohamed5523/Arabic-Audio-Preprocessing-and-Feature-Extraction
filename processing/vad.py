"""
Voice Activity Detection (VAD) processing for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import librosa
import soundfile as sf
import numpy as np
import torch
import torch.nn.functional as F
from utils.utils import save_waveform_plot
from config.config import Config

def apply_vad(audio_path, output_folder, vad_model, device, 
              threshold=None, min_speech_duration=None):
    """Step 2: Voice Activity Detection"""
    if threshold is None:
        threshold = Config.VAD_THRESHOLD
    if min_speech_duration is None:
        min_speech_duration = Config.MIN_SPEECH_DURATION
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE, mono=True)
    
    # Run VAD
    signal = torch.tensor(y).unsqueeze(0).to(device)
    length = torch.tensor([signal.shape[1]]).to(device)
    
    with torch.no_grad():
        logits = vad_model(input_signal=signal, input_signal_length=length)
        probs = F.softmax(logits, dim=2).cpu().numpy()[0]
        speech_probs = probs[:, 1]
    
    # Detect speech segments
    is_speech = (speech_probs > threshold).astype(bool).tolist()
    
    segments = []
    start = None
    
    for i, val in enumerate(is_speech):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i
            if (end - start) * Config.FRAME_DURATION >= min_speech_duration:
                segments.append((start, end))
            start = None
    
    if start is not None:
        end = len(is_speech)
        if (end - start) * Config.FRAME_DURATION >= min_speech_duration:
            segments.append((start, end))
    
    # Extract speech segments
    final_audio = []
    
    for start_f, end_f in segments:
        start_t = max(0, int((start_f * Config.FRAME_DURATION - Config.AUDIO_PADDING) * sr))
        end_t = min(len(y), int((end_f * Config.FRAME_DURATION + Config.AUDIO_PADDING) * sr))
        final_audio.extend(y[start_t:end_t])
    
    final_audio = np.array(final_audio)
    
    # Save VAD waveform plot
    if len(final_audio) > 0:
        vad_plot_path = os.path.join(output_folder, Config.VAD_PLOT_FILENAME)
        save_waveform_plot(final_audio, sr, "VAD Processed Audio Waveform", vad_plot_path)
        
        # Save VAD output
        vad_path = os.path.join(output_folder, Config.VAD_AUDIO_FILENAME)
        sf.write(vad_path, final_audio, sr)
    else:
        print(f"⚠️ Warning: No speech detected in audio, using original audio")
        vad_path = audio_path
        # Create a plot showing no speech detected
        vad_plot_path = os.path.join(output_folder, Config.VAD_PLOT_FILENAME)
        save_waveform_plot(y, sr, "VAD: No Speech Detected (Original Audio)", vad_plot_path)
    
    return vad_path