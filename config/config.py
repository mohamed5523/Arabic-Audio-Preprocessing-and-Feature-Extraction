"""
Configuration file for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os

class Config:
    """Configuration class containing all settings for the Arabic-Audio-Preprocessing-and-Feature-Extraction"""
    
    # Audio Processing Parameters
    SAMPLE_RATE = 16000
    VAD_THRESHOLD = 0.99
    MIN_SPEECH_DURATION = 0.15
    FRAME_DURATION = 0.02
    AUDIO_PADDING = 0.1
    
    # Plot Settings
    PLOT_DURATION_LIMIT = 30
    PLOT_DPI = 300
    PLOT_FIGSIZE = (14, 4)
    
    # Audio File Extensions
    AUDIO_EXTENSIONS = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.aac']
    
    # Model Settings
    VAD_MODEL_NAME = "nvidia/frame_vad_multilingual_marblenet_v2.0"
    DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization@2.1"
    EMBEDDING_MODEL_NAME = "pyannote/embedding"
    
    # Feature Extraction Parameters
    N_FFT = 512
    WIN_LENGTH_RATIO = 0.025  # 25ms window
    HOP_LENGTH_RATIO = 0.01   # 10ms hop
    N_MELS = 64
    F_MIN = 0.0
    F_MAX = 8000.0
    POWER = 2.0
    
    # Embedding Dimensions
    EMBEDDING_DIM = 512  # Pyannote embedding dimension
    LOGMEL_DIM = 192     # Log-mel feature dimension (64*3)
    
    # Milvus Settings
    DEFAULT_MILVUS_HOST = "localhost"
    DEFAULT_MILVUS_PORT = "19530"
    
    # Index Parameters for Milvus
    INDEX_PARAMS = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    
    # Search Parameters for Milvus
    SEARCH_PARAMS = {
        "metric_type": "COSINE", 
        "params": {"nprobe": 10}
    }
    
    # Collection Names
    EMBEDDING_COLLECTION_NAME = "speaker_embeddings"
    LOGMEL_COLLECTION_NAME = "logmel_features"
    
    # File Names
    ORIGINAL_AUDIO_FILENAME = "original_audio.wav"
    DENOISED_AUDIO_FILENAME = "denoised_audio.wav"
    VAD_AUDIO_FILENAME = "vad_audio.wav"
    DIARIZATION_RTTM_FILENAME = "diarization.rttm"
    FEATURES_JSON_FILENAME = "features.json"
    ALL_FEATURES_JSON_FILENAME = "all_audio_features.json"
    
    # Plot Filenames
    ORIGINAL_PLOT_FILENAME = "01_original_waveform.png"
    DENOISED_PLOT_FILENAME = "02_denoised_waveform.png"
    VAD_PLOT_FILENAME = "03_vad_waveform.png"
    
    @staticmethod
    def get_huggingface_token():
        """Get Hugging Face token from environment variable or return default"""
        return os.getenv('HUGGINGFACE_TOKEN', 'Your Tocken') # ADD YOUR HUGGINGFACE_TOKEN HERE
    
    @staticmethod
    def validate_paths(*paths):
        """Validate that all provided paths exist"""
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
    
    @staticmethod
    def create_output_dirs(*dirs):
        """Create output directories if they don't exist"""
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)