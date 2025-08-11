"""
Main entry point for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import sys
from core.audio_processor import AudioProcessor
from config.config import Config

def main():
    """Main function to run the Arabic-Audio-Preprocessing-and-Feature-Extraction"""
    # Configuration - Update these paths according to your setup
    INPUT_FOLDER = r"C:\Users\EW\Desktop\AudioFeature310\Test_Audios"  # Change this to your input folder path
    OUTPUT_FOLDER = r"C:\Users\EW\Desktop\AudioFeature310\justOutTest"  # Change this to your output folder path
    
    # Optional: Get from environment variables
    AUTH_TOKEN = os.getenv('HUGGINGFACE_TOKEN', Config.get_huggingface_token())
    MILVUS_HOST = os.getenv('MILVUS_HOST', Config.DEFAULT_MILVUS_HOST)
    MILVUS_PORT = os.getenv('MILVUS_PORT', Config.DEFAULT_MILVUS_PORT)
    
    print("🎵 Starting Arabic-Audio-Preprocessing-and-Feature-Extraction with Native Pyannote Embeddings and Milvus")
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"🗄️ Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    
    # Validate input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Error: Input folder does not exist: {INPUT_FOLDER}")
        sys.exit(1)
    
    try:
        # Create processor instance
        processor = AudioProcessor(INPUT_FOLDER, OUTPUT_FOLDER, AUTH_TOKEN, MILVUS_HOST, MILVUS_PORT)
        
        # Process all audio files
        processor.process_all_audios()
        
        # Get collection statistics
        processor.get_collection_stats()
        
        print("\n🎉 Arabic-Audio-Preprocessing-and-Feature-Extraction completed!")
        
        # Optional: Demo similarity search (uncomment to test)
        # if processor.all_embeddings:
        #     first_audio_path = processor.all_embeddings[0]["audio_path"]
        #     processor.demo_similarity_search(first_audio_path, top_k=3)
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()