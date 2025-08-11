"""
Speaker Diarization processing for the Arabic-Audio-Preprocessing-and-Feature-Extraction
"""

import os
from pydub import AudioSegment
from config.config import Config

def perform_diarization(audio_path, output_folder, diarization_pipeline):
    """Step 3: Speaker Diarization"""
    # Apply diarization
    diarization = diarization_pipeline(audio_path, num_speakers=2)
    
    # Save RTTM file
    rttm_path = os.path.join(output_folder, Config.DIARIZATION_RTTM_FILENAME)
    with open(rttm_path, "w") as rttm:
        diarization.write_rttm(rttm)
    
    # Separate speakers
    audio = AudioSegment.from_wav(audio_path)
    speaker_segments = {}
    
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = turn.start * 1000  # convert to ms
        end = turn.end * 1000
        
        segment_audio = audio[start:end]
        
        if speaker not in speaker_segments:
            speaker_segments[speaker] = segment_audio
        else:
            speaker_segments[speaker] += segment_audio
    
    # Export separated speaker audio
    speaker_files = {}
    for speaker, combined_audio in speaker_segments.items():
        speaker_file = os.path.join(output_folder, f"speaker_{speaker}.wav")
        combined_audio.export(speaker_file, format="wav")
        speaker_files[speaker] = speaker_file
    
    return speaker_files, rttm_path