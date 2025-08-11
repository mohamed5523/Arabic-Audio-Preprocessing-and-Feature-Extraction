"""
Model initialization and management for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import torch
import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline, Model, Inference
from config.config import Config

class ModelManager:
    """Manages all ML models used in the pipeline"""
    
    def __init__(self, auth_token):
        self.auth_token = auth_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model instances
        self.vad_model = None
        self.diarization_pipeline = None
        self.embedding_model = None
        self.embedding_inference = None
        
        # Initialize all models
        self.setup_models()
    
    def setup_models(self):
        """Initialize VAD, Diarization, and Embedding models"""
        print("🔄 Loading models...")
        
        # Load VAD model
        self._load_vad_model()
        
        # Load Diarization pipeline
        self._load_diarization_pipeline()
        
        # Load native pyannote Speaker Embedding model
        self._load_embedding_model()
        
        print(f"✅ Models loaded on device: {self.device}")
        print("✅ Using native pyannote embedding model")
    
    def _load_vad_model(self):
        """Load Voice Activity Detection model"""
        try:
            self.vad_model = nemo_asr.models.EncDecFrameClassificationModel.from_pretrained(
                model_name=Config.VAD_MODEL_NAME
            )
            self.vad_model.eval()
            self.vad_model = self.vad_model.to(self.device)
            print("✅ VAD model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading VAD model: {str(e)}")
            raise
    
    def _load_diarization_pipeline(self):
        """Load Speaker Diarization pipeline"""
        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                Config.DIARIZATION_MODEL_NAME,
                use_auth_token=self.auth_token
            )
            print("✅ Diarization pipeline loaded successfully")
        except Exception as e:
            print(f"❌ Error loading diarization pipeline: {str(e)}")
            raise
    
    def _load_embedding_model(self):
        """Load Speaker Embedding model"""
        try:
            # Load native pyannote Speaker Embedding model
            self.embedding_model = Model.from_pretrained(
                Config.EMBEDDING_MODEL_NAME,
                use_auth_token=self.auth_token
            )
            
            # Create inference object for embeddings
            self.embedding_inference = Inference(
                self.embedding_model,
                window="whole",  # Use whole audio segment
                device=torch.device(self.device)
            )
            print("✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {str(e)}")
            raise
    
    def get_vad_model(self):
        """Get VAD model instance"""
        return self.vad_model
    
    def get_diarization_pipeline(self):
        """Get diarization pipeline instance"""
        return self.diarization_pipeline
    
    def get_embedding_inference(self):
        """Get embedding inference instance"""
        return self.embedding_inference
    
    def get_device(self):
        """Get the device being used"""
        return self.device