"""
Example usage of the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.audio_processor import AudioProcessor

def basic_example():
    """Basic example of processing audio files"""
    print("🎵 Basic Audio Processing Example")
    
    # Configuration
    input_folder = "path/to/your/audio/files"
    output_folder = "path/to/output/folder"
    
    # Initialize processor
    processor = AudioProcessor(
        input_folder=input_folder,
        output_folder=output_folder,
        auth_token="your_huggingface_token",  # Optional
        milvus_host="localhost",
        milvus_port="19530"
    )
    
    # Process all audio files
    processor.process_all_audios()
    
    # Get statistics
    processor.get_collection_stats()

def single_file_example():
    """Example of processing a single audio file"""
    print("🎵 Single File Processing Example")
    
    # Configuration
    input_folder = "path/to/folder/containing/audio"
    output_folder = "path/to/output"
    audio_file_path = "path/to/specific/audio/file.wav"
    
    # Initialize processor
    processor = AudioProcessor(input_folder, output_folder)
    
    # Process single file
    success, message = processor.process_single_audio(audio_file_path)
    print(message)

def similarity_search_example():
    """Example of performing similarity search"""
    print("🔍 Similarity Search Example")
    
    # Initialize processor (assumes you have already processed some files)
    processor = AudioProcessor("input", "output")
    
    # Query audio file
    query_audio = "path/to/query/audio.wav"
    
    # Search for similar speakers
    processor.demo_similarity_search(query_audio, top_k=5)

def batch_processing_example():
    """Example of processing multiple folders"""
    print("📁 Batch Processing Example")
    
    folders_to_process = [
        "path/to/folder1",
        "path/to/folder2",
        "path/to/folder3"
    ]
    
    base_output = "output"
    
    for i, folder in enumerate(folders_to_process):
        print(f"Processing folder {i+1}/{len(folders_to_process)}: {folder}")
        
        output_folder = os.path.join(base_output, f"batch_{i+1}")
        
        processor = AudioProcessor(folder, output_folder)
        processor.process_all_audios()
        
        # Get stats for this batch
        embedding_count, logmel_count = processor.get_collection_stats()
        print(f"Batch {i+1} completed: {embedding_count} embeddings, {logmel_count} log-mel features")

def custom_config_example():
    """Example with custom configuration"""
    print("⚙️ Custom Configuration Example")
    
    from config.config import Config
    
    # Modify configuration
    Config.VAD_THRESHOLD = 0.95  # More strict VAD
    Config.MIN_SPEECH_DURATION = 0.2  # Longer minimum speech segments
    Config.SAMPLE_RATE = 22050  # Different sample rate
    
    processor = AudioProcessor("input", "output")
    processor.process_all_audios()

def docker_example():
    """Example of running with Docker"""
    print("🐳 Docker Example")
    print("To run with Docker:")
    print("1. Build the image:")
    print("   docker build -t Arabic-Audio-Preprocessing-and-Feature-Extraction .")
    print("")
    print("2. Run with volume mounts:")
    print("   docker run -it --rm \\")
    print("     -v /path/to/input:/app/input \\")
    print("     -v /path/to/output:/app/output \\")
    print("     -e HUGGINGFACE_TOKEN=your_token \\")
    print("     --network host \\")
    print("     Arabic-Audio-Preprocessing-and-Feature-Extraction")
    print("")
    print("3. Or use docker-compose to start Milvus:")
    print("   docker-compose up -d")

def environment_setup_example():
    """Example of setting up environment variables"""
    print("🌍 Environment Setup Example")
    
    # Set environment variables
    os.environ['HUGGINGFACE_TOKEN'] = 'your_token_here'
    os.environ['MILVUS_HOST'] = 'localhost'
    os.environ['MILVUS_PORT'] = '19530'
    
    # These will be automatically picked up by the Config class
    processor = AudioProcessor("input", "output")
    processor.process_all_audios()

def error_handling_example():
    """Example with error handling"""
    print("🚨 Error Handling Example")
    
    try:
        processor = AudioProcessor("input", "output")
        processor.process_all_audios()
    except FileNotFoundError as e:
        print(f"Input folder not found: {e}")
    except ConnectionError as e:
        print(f"Cannot connect to Milvus: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run examples
    print("🎵 Arabic-Audio-Preprocessing-and-Feature-Extraction Examples\n")
    
    # Uncomment the example you want to run
    # basic_example()
    # single_file_example()
    # similarity_search_example()
    # batch_processing_example()
    # custom_config_example()
    docker_example()
    # environment_setup_example()
    # error_handling_example()