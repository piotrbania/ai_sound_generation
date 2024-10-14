import os
import re
import sys
import pprint 

import torch
import multiprocessing


from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


#OPT_LANGUAGE = "pl"  # change to other language if necessary ie. "en-us"

OPT_LANGUAGE = "en-us"

# google collab settings (tweak this)
OPT_MAX_EPOCH          = 1000
OPT_BATCH_SIZE         = 32
OPT_EVAL_BATCH_SIZE    = 16
OPT_LEARNING_RATE      = 0.001





if torch.cuda.is_available() != True:
    print("! Error: no GPU support found")
    sys.exit(0)
     

# this is for polish lang 
def clean_polish_text(text):
    # Remove unsupported characters
    cleaned_text = re.sub(r'[^a-zA-Z0-9ąćęłńóśźżĄĆĘŁŃÓŚŹŻ|\s,.!?\'"-]', '', text)
    cleaned_text = " ".join(cleaned_text.split()) 
    return cleaned_text


def is_file_less_than_1mb(file_path):
    # Get the size of the file in bytes
    file_size = os.path.getsize(file_path)
    # 1 MB = 1,048,576 bytes
    return file_size < (1 * 1024 * 1024)




def fix_metadata(root_path, manifest_path, out_manifest_path):

    out = open(out_manifest_path, "w", encoding="utf-8")
    with open(manifest_path, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0])
            text = cols[1]
            
            if OPT_LANGUAGE == "pl":
                text = clean_polish_text(text)

            if len(text) < 3:
                continue 
            
            
            if is_file_less_than_1mb(os.path.join(wavs_path, wav_file)) == True:
                #print("Skipping file: \"%s\" \r\n" % wav_file)
                continue
            
            wav_file = cols[0]
            wav_file = wav_file.replace(".wav", "")
            text = text.replace("\n", "")
            out.write("%s|%s|test\n" % (wav_file, text))
    out.close()






multiprocessing.freeze_support()


if __name__ == '__main__':


    print("+ Using GPU: %s | device count: %d" % (torch.cuda.get_device_name(0), torch.cuda.device_count()))

    cdir        = os.path.dirname(os.path.realpath(__file__))

    output_path = os.path.join(cdir, "output")
    input_path  = os.path.join(cdir, "dataset")
    wavs_path   = os.path.join(input_path, "wavs")

    file_count = len([f for f in os.listdir(wavs_path) if os.path.isfile(os.path.join(wavs_path, f))])

    print("+ Dataset size: %d " % file_count)


    os.makedirs(output_path, exist_ok=True)


    # Assuming you know your dataset size
    
    dataset_size    = file_count  # example dataset size
    
    '''
    batch_size      = 32
    max_steps       = 50000

    steps_per_epoch = dataset_size // batch_size 
    max_epochs = max_steps // steps_per_epoch
    '''
    
 

    print("+ batch size: %d | max epochs: %d | learning rate: %d \n" % (OPT_BATCH_SIZE, OPT_MAX_EPOCH, OPT_LEARNING_RATE))


    fix_metadata(input_path, os.path.join(input_path, "metadata.csv"), os.path.join(input_path, "metadataCLEAN.csv"))
    
    dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="metadataCLEAN.csv", path=input_path)


    # tweak this if necessary 
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ljspeech",
        batch_size=OPT_BATCH_SIZE,
        eval_batch_size=OPT_EVAL_BATCH_SIZE,
        batch_group_size=5,
        num_loader_workers=8,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=OPT_MAX_EPOCH,
        use_phonemes=True,
        text_cleaner="phoneme_cleaners",
        phoneme_language=OPT_LANGUAGE, 
        phonemizer="espeak", # added 
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=False, # in case of error TypeError: Got unsupported ScalarType BFloat16 type this to False
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=False,
        target_loss="loss_1",
        lr=OPT_LEARNING_RATE,
        save_step=200,
    )


    ap = AudioProcessor.init_from_config(config)


    tokenizer, config = TTSTokenizer.init_from_config(config)

    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and go
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()