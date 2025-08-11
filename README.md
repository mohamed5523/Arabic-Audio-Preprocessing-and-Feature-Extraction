# Arabic-Audio-Preprocessing-and-Feature-Extraction

A comprehensive audio processing pipeline that performs speaker diarization, voice activity detection, and feature extraction with vector storage in Milvus for similarity search.

## Features

- **Audio Preprocessing**: Noise reduction and audio normalization
- **Voice Activity Detection (VAD)**: Using NVIDIA NeMo models
- **Speaker Diarization**: Using Pyannote.audio for speaker separation
- **Feature Extraction**: 
  - Speaker embeddings using native Pyannote models (512D)
  - Log-Mel features (192D)
- **Vector Database**: Milvus integration for similarity search
- **Visualization**: Waveform plots for each processing step

## Project Structure

```
Arabic-Audio-Preprocessing-and-Feature-Extraction/
├── 📁 config/
│   ├── __init__.py
│   └── config.py                    # Configuration settings and parameters
│
├── 📁 models/
│   ├── __init__.py
│   └── models.py                    # Model initialization and management
│
├── 📁 processing/
│   ├── __init__.py
│   ├── preprocessing.py             # Audio preprocessing (denoising)
│   ├── vad.py                      # Voice Activity Detection
│   ├── diarization.py              # Speaker Diarization
│   └── feature_extraction.py       # Feature extraction (embeddings, log-mel)
│
├── 📁 database/
│   ├── __init__.py
│   └── milvus_handler.py           # Milvus database operations
│
├── 📁 utils/
│   ├── __init__.py
│   └── utils.py                    # Utility functions
│
├── 📁 core/
│   ├── __init__.py
│   └── audio_processor.py          # Main processor orchestrator
│
├── 📁 examples/
│   └── example_usage.py            # Usage examples
│
├── 📁 tests/
│   └── test_audio_processor.py     # Unit tests
│
├── 📄 main.py                      # Entry point
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                     # Package setup
├── 📄 docker-compose.yml           # Docker Compose for Milvus
├── 📄 Makefile                     # Build and deployment commands
├── 📄 .gitignore                   # Git ignore rules
└── 📄 README.md                    # Documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mohamed5523/Arabic-Audio-Preprocessing-and-Feature-Extraction
cd Arabic-Audio-Preprocessing-and-Feature-Extraction
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate   # On Linux:source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install and start Milvus:
   - Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - Or use the provided docker-compose: `make milvus-up`
   - Default connection: `localhost:19530`
   - Web UI (Attu): `localhost:18080`

## Configuration

### Environment Variables

Set the following environment variables (optional):

```bash
export HUGGINGFACE_TOKEN="your_huggingface_token_here"
export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
```

### Config File

Modify `config/config.py` to adjust processing parameters:

- Audio processing settings (sample rate, thresholds)
- Model names and parameters
- Milvus connection settings
- Feature extraction parameters

## Usage

### Basic Usage

1. Update the paths in `main.py`:
```python
INPUT_FOLDER = "path/to/your/audio/files"
OUTPUT_FOLDER = "path/to/output/folder"
```

2. Run the pipeline:
```bash
python main.py
```

### Programmatic Usage

```python
from core.audio_processor import AudioProcessor

# Initialize processor
processor = AudioProcessor(
    input_folder="path/to/audio/files",
    output_folder="path/to/output",
    auth_token="your_hf_token",
    milvus_host="localhost",
    milvus_port="19530"
)

# Process all audio files
processor.process_all_audios()

# Get statistics
processor.get_collection_stats()

# Demo similarity search
processor.demo_similarity_search("path/to/query/audio.wav", top_k=5)
```

## Output Structure

For each processed audio file, the pipeline creates:

```
output/
└── audio_name/
    ├── 01_original_waveform.png      # Original audio visualization
    ├── 02_denoised_waveform.png      # Denoised audio visualization
    ├── 03_vad_waveform.png           # VAD processed audio
    ├── original_audio.wav            # Original audio copy
    ├── denoised_audio.wav           # Noise-reduced audio
    ├── vad_audio.wav                # VAD processed audio
    ├── diarization.rttm             # Speaker diarization results
    ├── speaker_SPEAKER_00.wav       # Separated speaker audio
    ├── speaker_SPEAKER_01.wav       # Separated speaker audio
    └── audio_name_features.json     # Extracted features
```

## Pipeline Steps

1. **Preprocessing**: Load audio, apply noise reduction, generate visualizations
2. **Voice Activity Detection**: Remove non-speech segments using NVIDIA NeMo
3. **Speaker Diarization**: Separate speakers using Pyannote.audio
4. **Feature Extraction**: 
   - Extract speaker embeddings (512D) using native Pyannote
   - Extract log-mel features (192D) for comparison
5. **Storage**: Store features in Milvus vector database for similarity search

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- M4A (.m4a)
- AAC (.aac)

## Dependencies

Key dependencies include:
- `torch` and `torchaudio` for PyTorch operations
- `librosa` for audio processing
- `nemo-toolkit` for VAD
- `pyannote.audio` for diarization and embeddings
- `pymilvus` for vector database operations
- `matplotlib` for visualizations

See `requirements.txt` for complete list.

## Troubleshooting

### Common Issues

1. **Milvus Connection Error**: Ensure Milvus is running on the specified host/port
2. **CUDA Out of Memory**: Reduce batch size or use CPU processing
3. **Hugging Face Token Error**: Set your token in environment variables
4. **Audio Loading Error**: Check audio file format and integrity

### Model Downloads

Models are automatically downloaded on first use:
- NVIDIA NeMo VAD model (~100MB)
- Pyannote speaker diarization model (~50MB)
- Pyannote speaker embedding model (~100MB)

## Performance Notes

- Processing time depends on audio length and complexity
- GPU acceleration recommended for large datasets
- Milvus indexing improves search performance for large collections

## License

This project is licensed under the MIT License.

