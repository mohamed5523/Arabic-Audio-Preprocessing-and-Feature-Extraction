# Arabic-Audio-Preprocessing-and-Feature-Extraction
Arabic Audio Preprocessing: Denoising, VAD, Diarization, and Feature Extraction 


This project provides a complete pipeline for processing Arabic audio data, including denoising, silence trimming, speaker diarization, and per-speaker feature extraction such as MFCC and Log-Mel spectrograms.

---

## 🧠 Project Overview

The main goal of this project is to prepare noisy Arabic audio for downstream tasks like transcription or speaker analysis by applying:

1. **Denoising**
2. **Voice Activity Detection (VAD)**
3. **Speaker Diarization**
4. **Feature Extraction (MFCC, Log-Mel, etc.)**
---


## 📁 Project Structure

```text
ArabicAudioProcessing/
├── README.md                # Project overview and usage
├── LICENSE                  # Project license (e.g., MIT)
├── requirements.txt         # List of required Python packages
├── main.py                  # Main pipeline script
├── audio_utils/             # Core audio processing modules
│   ├── __init__.py
│   ├── denoise.py           # Handles audio denoising
│   ├── vad.py               # Applies Voice Activity Detection
│   ├── diarization.py       # Performs speaker diarization
│   ├── feature_extraction.py # Extracts MFCCs, Log-Mel, etc.
├── data/
│   └── Audio/               # Raw input audio files (nested folders)
├── output/
│   ├── diarized_audio/      # Speaker-separated audio segments
│   └── features/            # Extracted audio features (per speaker)
└── .gitignore               # Files/folders to be ignored by Git
---

```
## 🔧 Features

- ✅ **Denoising**: Traditional filters and advanced tools (e.g., NVIDIA Nemo , noisereduce , etc..)
- ✅ **VAD**: Remove non-speech regions using (MarblNet from Nemo)
- ✅ **Speaker Diarization**: Split audio by speaker using pretrained models (Pyannote Speaker Diarization)
- ✅ **Per-Speaker Feature Extraction**:
  - MFCC
  - Log-Mel Spectrogram
  - (Optional) Spectral Centroid, Pitch
  

## 🛠️ Installation
```
git clone https://github.com/YOUR_USERNAME/ArabicAudioProcessing.git
cd ArabicAudioProcessing
pip install -r requirements.txt
🚀 Usage
Run the complete pipeline with:

python main.py
Make sure your input audio files are inside data/Audio/, following the expected nested structure.
```
## 📦 Dependencies

To run this project, you need the following Python libraries:

- [librosa](https://librosa.org/) — audio loading and feature extraction
- [noisereduce](https://github.com/timsainb/noisereduce) — traditional audio denoising
- [matplotlib](https://matplotlib.org/) — visualization of spectrograms and features
- [soundfile](https://pysoundfile.readthedocs.io/) — audio file I/O
- [numpy](https://numpy.org/) — array and signal processing
- [torch](https://pytorch.org/) — core deep learning framework
- [torchaudio](https://pytorch.org/audio/) — audio preprocessing and I/O for PyTorch
- [nemo_toolkit[asr]](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/index.html) — NVIDIA NeMo for ASR and diarization
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) — pretrained speaker diarization models
- [pydub](https://github.com/jiaaro/pydub) — audio manipulation
- [pandas](https://pandas.pydata.org/) — tabular data manipulation
- [tqdm](https://tqdm.github.io/) — progress bars for loops

> Python built-in modules used: `os`, `glob`, `shutil`, `warnings`


All dependencies are listed in requirements.txt.

## 📊 Output
✅ Denoised and cleaned audio — raw audio is processed to remove noise

✅ Silence-trimmed segments — non-speech regions removed using VAD

✅ Diarized audio files → Separate Speakers in each audio by using Pyannote (one file per speaker)

✅ Extracted features:

   * MFCCs, Log-Mel spectrograms, pitch, and more

   * Saved as .xlsx files for (MFCC , LogMel and more)
  

## 📜 License
This project is licensed under the MIT License.

