"""
Feature extraction functions for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import numpy as np
import torch
import torchaudio
import uuid
from datetime import datetime
import traceback
from config.config import Config

def extract_speaker_embedding(audio_path, audio_name, embedding_inference, speaker_id=None):
    """Step 4A: Extract speaker embeddings using native pyannote"""
    try:
        # Load and preprocess audio for pyannote
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample to 16kHz if needed (pyannote typically expects 16kHz)
        if sample_rate != Config.SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=Config.SAMPLE_RATE)
            waveform = resampler(waveform)
            sample_rate = Config.SAMPLE_RATE
        
        # Create audio dictionary for pyannote
        audio_dict = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }
        
        # Extract embeddings using pyannote inference
        with torch.no_grad():
            embedding = embedding_inference(audio_dict)
        
        # Convert to numpy array
        if isinstance(embedding, torch.Tensor):
            embedding_vector = embedding.cpu().numpy()
        else:
            embedding_vector = np.array(embedding)
        
        # Ensure embedding is 1D
        if len(embedding_vector.shape) > 1:
            embedding_vector = embedding_vector.flatten()
        
        print(f"✅ Extracted pyannote embedding with shape: {embedding_vector.shape}")
        
        # Prepare embedding data
        embedding_data = {
            "id": str(uuid.uuid4()),
            "audio_name": audio_name,
            "speaker_id": speaker_id if speaker_id else "combined",
            "audio_path": audio_path,
            "embedding_vector": embedding_vector.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return embedding_data
        
    except Exception as e:
        print(f"❌ Error extracting pyannote embeddings for {audio_name}: {str(e)}")
        traceback.print_exc()
        return None

def extract_logmel_features(audio_path, audio_name, speaker_id=None):
    """Step 4B: Extract Log-Mel features (kept for comparison)"""
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.to(torch.float32)
        
        # Convert stereo to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != Config.SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=Config.SAMPLE_RATE)(waveform)
            sr = Config.SAMPLE_RATE
        
        # Extract Log-Mel features (192D)
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.SAMPLE_RATE,
            n_fft=Config.N_FFT,
            win_length=int(Config.WIN_LENGTH_RATIO * Config.SAMPLE_RATE),
            hop_length=int(Config.HOP_LENGTH_RATIO * Config.SAMPLE_RATE),
            n_mels=Config.N_MELS,
            f_min=Config.F_MIN,
            f_max=Config.F_MAX,
            power=Config.POWER,
        )
        
        with torch.no_grad():
            mel_spec = mel_spec_transform(waveform)
            log_mel_spec = torch.log(mel_spec + 1e-9).squeeze(0)
            delta = torchaudio.functional.compute_deltas(log_mel_spec)
            delta2 = torchaudio.functional.compute_deltas(delta)
            features_all = torch.cat([log_mel_spec, delta, delta2], dim=0)
            
            # Normalize
            mean = features_all.mean(dim=1, keepdim=True)
            std = features_all.std(dim=1, keepdim=True)
            features_norm = (features_all - mean) / (std + 1e-9)
        
        # Mean over time (192D)
        mel_mean = features_norm[0:64].mean(dim=1).cpu().numpy()
        delta_mean = features_norm[64:128].mean(dim=1).cpu().numpy()
        delta2_mean = features_norm[128:192].mean(dim=1).cpu().numpy()
        
        # Combine all features into single vector
        logmel_vector = np.concatenate([mel_mean, delta_mean, delta2_mean])
        
        # Prepare Log-Mel data
        logmel_data = {
            "id": str(uuid.uuid4()),
            "audio_name": audio_name,
            "speaker_id": speaker_id if speaker_id else "combined",
            "audio_path": audio_path,
            "logmel_vector": logmel_vector.tolist(),
            "timestamp": datetime.now().isoformat()
        }
        
        return logmel_data
        
    except Exception as e:
        print(f"❌ Error extracting log-mel features for {audio_name}: {str(e)}")
        return None