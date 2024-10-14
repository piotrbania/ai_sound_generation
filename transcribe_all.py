# prepares dataset for Coqui TTS training 
# input_directory - put you *.wav files here with the voice you would like to use 
# each file will be converted to MONO and later transcribed using Whisper, verify and fix the transcription later if necessary
# Coqui TTS requires following dataset structure:
#
# dataset/
#  + wavs/ (dir with split audio)  
#  + metadata.csv
# 
# - piotr bania 


#
# to install Whisper use:
# pip uninstall whisper
# pip install git+https://github.com/openai/whisper.git 


import os
import sys
import subprocess

import whisper
from pathlib import Path

cur_dir = os.getcwd()
cur_dir = os.path.join(cur_dir, "dataset")


#lang                        = "pl"

lang = "en"

input_directory             = os.path.join(cur_dir, 'input_directory')
output_directory            = os.path.join(cur_dir, 'wavs')
transcriptions_directory    = os.path.join(cur_dir, 'transcriptions')
csv_file                    = os.path.join(cur_dir, 'metadata.csv')
segment_time                = 15  # Time in seconds


print("+ Prepare and transcribe dataset \n")



if os.path.exists(input_directory) == False:
    print("! Input directory does not exit: \"%s\" \r\n" % str(input_directory))
    sys.exit(0)

os.makedirs(output_directory, exist_ok=True)
os.makedirs(transcriptions_directory, exist_ok=True)

model = whisper.load_model("base")

dataset = open(csv_file, "w", encoding='utf-8')
    
count = 0

for wav_file in Path(input_directory).rglob('*.[wm][ap][v3]'):
    print(f"Processing file: {wav_file.name}")
    count = count + 1
    
    # split the audio file using FFmpeg
    output_pattern = os.path.join(output_directory, f"{wav_file.stem}_output_{count}_%03d.wav")
    
    # command to split an audio file (wav_file) into segments based on detected silence, converting the audio to mono (1 channel) with a sample rate of 22050 Hz
    subprocess.run(['ffmpeg', '-i', str(wav_file), 
    '-af', 'silencedetect=noise=-30dB:d=0.5',  # Adjust silence detection parameters as needed
    '-f', 'segment', 
    '-ac', '1', 
    '-ar', '22050', 
    '-segment_time', str(segment_time), 
    '-c:a', 'pcm_s16le', 
    output_pattern
    ], check=True)


    # transcribe each split audio chunk
    for split_file in Path(output_directory).glob(f'{wav_file.stem}_output_*.wav'):
        print(f"Transcribing {split_file.name}...")

        # perform the transcription using Whisper
        result = model.transcribe(str(split_file), language=lang)
        transcription = result['text'].strip()
    
        dataset.write(f"{split_file.name}|{transcription}\n")

print(f"Transcriptions saved to {csv_file}.")
