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

ArabicAudioProcessing/
├── README.md
├── LICENSE
├── requirements.txt
├── main.py  ← your main pipeline script
├── audio_utils/
│   ├── __init__.py
│   ├── denoise.py
│   ├── vad.py
│   ├── diarization.py
│   ├── feature_extraction.py
├── data/
│   └── Audio/   ← your raw input dataset
├── output/
│   ├── diarized_audio/
│   ├── features/
└── .gitignore



---

## 🔧 Features

- ✅ **Denoising**: Traditional filters and advanced tools (e.g., NVIDIA Maxine)
- ✅ **VAD**: Remove non-speech regions using energy or model-based VAD
- ✅ **Speaker Diarization**: Split audio by speaker using pretrained models
- ✅ **Per-Speaker Feature Extraction**:
  - MFCC
  - Log-Mel Spectrogram
  - (Optional) Spectral Centroid, Pitch

---

## 🛠️ Installation

```
git clone https://github.com/YOUR_USERNAME/ArabicAudioProcessing.git
cd ArabicAudioProcessing
pip install -r requirements.txt
🚀 Usage
Run the complete pipeline with:

python main.py
Make sure your input audio files are inside data/Audio/, following the expected nested structure.

💼 Dependencies
librosa

pydub

noisereduce

SpeechBrain

NVIDIA Maxine SDK (optional, for high-end denoising)

torchaudio, scipy, matplotlib (for feature extraction/visualization)

All dependencies are listed in requirements.txt.

📊 Output
Cleaned and diarized audio files → output/diarized_audio/

Feature CSVs and spectrogram images → output/features/

📜 License
This project is licensed under the MIT License.

🤝 Contributing
Pull requests and suggestions are welcome. For major changes, please open an issue first to discuss.
