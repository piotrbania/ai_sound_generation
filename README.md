# ai_sound_generation
Text-to-speach python scripts to automatically create samples and teach / tune the coqui-ai TTS


# transcribe_all.py


# Dataset Preparation Script for Coqui TTS Training

This script prepares a dataset for training a Text-to-Speech (TTS) model using [Coqui TTS](https://github.com/coqui-ai/TTS). It processes audio files by splitting them into segments, transcribing them using OpenAI's Whisper model, and organizing the dataset in the required format for Coqui TTS.

## Input and Output Structure

- **Input Directory**: Place your audio files (`*.wav`, `*.mp3`, etc.) here. The audio should be of the voice you want to use for TTS training.
- **Output Directory**: The script will convert each audio file to MONO and segment it based on silence detection.
- **Transcription Directory**: The script will save the transcriptions here.
- **Metadata File**: A `metadata.csv` file will be generated, containing the filenames and corresponding transcriptions.

## Script Workflow

    Preparation: Ensures the required directories exist.
    Audio Processing:
        Splits the audio files from the input_directory into smaller segments using FFmpeg.
        Each segment is converted to MONO with a sample rate of 22050 Hz.
        Silence detection is used to determine segment boundaries.
    Transcription:
        Each audio segment is transcribed using the Whisper model.
        Transcriptions are saved in metadata.csv in the format: filename|transcription.




# teach_voice2.py

# TTS Training Preparation and Execution Script

This script sets up and runs training for a Text-to-Speech (TTS) model using the [Coqui TTS](https://github.com/coqui-ai/TTS) framework. It supports both Tacotron2 and VITS models and performs dataset preparation, configuration, and training execution.

## Features

1. **Dataset Preparation**:
   - Cleans and processes the metadata for training.
   - Filters out unsuitable audio files based on their size and language.
   - Supports cleaning text for Polish language using a regex-based approach.

2. **Configuration**:
   - Sets up the training configurations, including audio processing, learning rate, batch sizes, epochs, and phoneme processing.
   - Supports setting up configurations for VITS models with customizable audio parameters (e.g., sample rate, mel-spectrogram properties).
   - Uses the VITS or Tacotron2 model configuration depending on your choice.

3. **Training Execution**:
   - Automatically detects if a GPU is available for training.
   - Configures the trainer with the provided dataset and starts the training loop.
   - Supports mixed precision training and phoneme-based text-to-speech synthesis.
   - Uses Coqui's built-in `Trainer` class to manage training, evaluation, and logging.

## Prerequisites

- **GPU**: The script requires a CUDA-capable GPU for training.
- **Coqui TTS Installation**: Make sure to have [Coqui TTS](https://github.com/coqui-ai/TTS) installed.
- **Python Dependencies**: Required packages include `torch`, `os`, `re`, and Coqui's TTS-specific modules.

## Usage

1. **Dataset Setup**:
   - Place your dataset inside the `dataset/wavs` directory.
   - Prepare a `metadata.csv` file in the `dataset` directory with appropriate formatting.

2. **Script Configuration**:
   - Modify the language, batch size, learning rate, and other parameters to fit your needs.

3. **Run the Script**:
   - Execute the script to preprocess the dataset, set up configurations, and start training the TTS model.

4. **Output**:
   - The trained model, logs, and other outputs will be saved in the `output` directory.

## Customization

- Change the target language by modifying `OPT_LANGUAGE`.
- Adjust training parameters like `OPT_MAX_EPOCH`, `OPT_BATCH_SIZE`, and `OPT_LEARNING_RATE` to fit your requirements.
- Modify audio processing parameters for different sample rates and mel-spectrogram configurations.

## Notes

- The script checks for GPU support before starting training.
- It uses Espeak for phoneme generation when phoneme-based synthesis is enabled.
- Make sure to have an appropriate environment set up with required dependencies and a supported GPU.

- You need to tweak those params otherwise the output will sound like a drunk chipmunk 

## Author

- Piotr Bania
