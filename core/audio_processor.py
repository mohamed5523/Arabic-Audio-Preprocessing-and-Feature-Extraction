"""
Main Audio Processor class that orchestrates the entire pipeline
"""

import os
import json
from datetime import datetime
from tqdm import tqdm

from config.config import Config
from models.models import ModelManager
from database.milvus_handler import MilvusHandler
from processing.preprocessing import preprocess_audio
from processing.vad import apply_vad
from processing.diarization import perform_diarization
from processing.feature_extraction import extract_speaker_embedding, extract_logmel_features
from utils.utils import (
    find_audio_files, create_output_structure, validate_audio_file, 
    get_audio_name, print_processing_summary, print_collection_stats
)

class AudioProcessor:
    """Main Arabic-Audio-Preprocessing-and-Feature-Extraction orchestrator"""
    
    def __init__(self, input_folder, output_folder, auth_token=None, 
                 milvus_host=None, milvus_port=None):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.auth_token = auth_token or Config.get_huggingface_token()
        
        # Create output folder
        Config.create_output_dirs(output_folder)
        
        # Initialize components
        self.model_manager = ModelManager(self.auth_token)
        self.milvus_handler = MilvusHandler(milvus_host, milvus_port)
        
        # Initialize lists for tracking processed data
        self.all_embeddings = []
        self.all_logmel_features = []
    
    def process_single_audio(self, audio_path):
        """Process a single audio file through the complete pipeline"""
        audio_name = get_audio_name(audio_path)
        
        # Create output folder maintaining input structure
        audio_output_folder = create_output_structure(
            audio_path, self.input_folder, self.output_folder
        )
        
        try:
            # Validate audio file
            is_valid, message = validate_audio_file(audio_path)
            if not is_valid:
                return False, f"❌ {message}"
            
            # Step 1: Preprocessing
            denoised_path, _, _ = preprocess_audio(audio_path, audio_output_folder)
            
            # Step 2: VAD
            vad_path = apply_vad(
                denoised_path, 
                audio_output_folder, 
                self.model_manager.get_vad_model(),
                self.model_manager.get_device()
            )
            
            # Step 3: Diarization
            speaker_files, rttm_path = perform_diarization(
                vad_path, 
                audio_output_folder, 
                self.model_manager.get_diarization_pipeline()
            )
            
            # Step 4: Feature extraction and Milvus insertion for each speaker
            audio_features = {"embeddings": [], "logmel": []}
            
            for speaker_id, speaker_file in speaker_files.items():
                if os.path.exists(speaker_file) and os.path.getsize(speaker_file) > 0:
                    # Extract speaker embeddings using native pyannote
                    embedding_data = extract_speaker_embedding(
                        speaker_file, 
                        audio_name, 
                        self.model_manager.get_embedding_inference(),
                        speaker_id
                    )
                    
                    # Extract log-mel features
                    logmel_data = extract_logmel_features(
                        speaker_file, audio_name, speaker_id
                    )
                    
                    # Insert to Milvus
                    if self.milvus_handler.insert_data(embedding_data, logmel_data):
                        if embedding_data:
                            audio_features["embeddings"].append(embedding_data)
                            self.all_embeddings.append(embedding_data)
                        if logmel_data:
                            audio_features["logmel"].append(logmel_data)
                            self.all_logmel_features.append(logmel_data)
                else:
                    print(f"⚠️ Warning: Speaker file {speaker_id} is empty or missing for {audio_name}")
            
            # Save individual audio features to JSON
            if audio_features["embeddings"] or audio_features["logmel"]:
                json_filename = f"{audio_name}_{Config.FEATURES_JSON_FILENAME}"
                json_path = os.path.join(audio_output_folder, json_filename)
                with open(json_path, 'w') as f:
                    json.dump(audio_features, f, indent=2)
            
            return True, f"✅ Successfully processed: {audio_name}"
            
        except Exception as e:
            return False, f"❌ Error processing {audio_name}: {str(e)}"
    
    def process_all_audios(self):
        """Process all audio files in the input folder"""
        # Find all audio files
        audio_files = find_audio_files(self.input_folder)
        
        if not audio_files:
            print("❌ No audio files found in the input folder!")
            return
        
        print(f"🎵 Found {len(audio_files)} audio files to process")
        
        # Process each audio file
        successful = 0
        failed = 0
        
        for audio_path in tqdm(audio_files, desc="Processing audio files"):
            audio_name = get_audio_name(audio_path)
            tqdm.write(f"🔄 Processing: {audio_name}")
            
            success, message = self.process_single_audio(audio_path)
            tqdm.write(message)
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Flush data to Milvus
        self.milvus_handler.flush_collections()
        
        # Save combined features to JSON
        self._save_combined_features()
        
        # Print summary
        print_processing_summary(
            successful, failed, self.output_folder, 
            self.milvus_handler.host, self.milvus_handler.port
        )
    
    def _save_combined_features(self):
        """Save all combined features to a single JSON file"""
        if self.all_embeddings or self.all_logmel_features:
            print("\n💾 Saving combined features...")
            
            combined_data = {
                "embeddings": self.all_embeddings,
                "logmel_features": self.all_logmel_features,
                "metadata": {
                    "total_files": len(self.all_embeddings),
                    "embedding_dimension": len(self.all_embeddings[0]["embedding_vector"]) if self.all_embeddings else 0,
                    "logmel_dimension": len(self.all_logmel_features[0]["logmel_vector"]) if self.all_logmel_features else 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            combined_json_path = os.path.join(self.output_folder, Config.ALL_FEATURES_JSON_FILENAME)
            with open(combined_json_path, 'w') as f:
                json.dump(combined_data, f, indent=2)
            
            print(f"✅ Combined features saved to: {combined_json_path}")
            print(f"✅ Milvus collections: {Config.EMBEDDING_COLLECTION_NAME}, {Config.LOGMEL_COLLECTION_NAME}")
    
    def demo_similarity_search(self, query_audio_path, top_k=5):
        """Demo function to search for similar speakers"""
        print(f"\n🔍 Searching for speakers similar to: {query_audio_path}")
        
        # Extract embedding from query audio
        audio_name = get_audio_name(query_audio_path)
        embedding_data = extract_speaker_embedding(
            query_audio_path, 
            audio_name, 
            self.model_manager.get_embedding_inference()
        )
        
        if embedding_data:
            # Search for similar speakers
            results = self.milvus_handler.search_similar_speakers(
                embedding_data["embedding_vector"], top_k
            )
            
            if results:
                print(f"🎯 Found {len(results[0])} similar speakers:")
                for i, result in enumerate(results[0]):
                    print(f"  {i+1}. Audio: {result.entity.get('audio_name')}")
                    print(f"     Speaker: {result.entity.get('speaker_id')}")
                    print(f"     Distance: {result.distance:.4f}")
                    print(f"     Path: {result.entity.get('audio_path')}")
                    print()
    
    def get_collection_stats(self):
        """Get statistics about Milvus collections"""
        embedding_count, logmel_count = self.milvus_handler.get_collection_stats()
        print_collection_stats(embedding_count, logmel_count)
        return embedding_count, logmel_count