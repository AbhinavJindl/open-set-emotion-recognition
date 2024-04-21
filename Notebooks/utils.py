import os
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
from scipy import signal
from scipy.io import wavfile
from torch_audiomentations import Compose, Gain, AddColoredNoise, PitchShift, PeakNormalization
import torch
import soundfile as sf

def create_spectrogram_helper(filename, audio_file_path, output_file_path):
    y, sr = librosa.load(audio_file_path)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, fmax=8000)
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()

def create_spectrograms(iemocap_spectrogram_dir, audio_files):
    log_dir = os.path.join(os.path.dirname(os.getcwd()),'iemocap','log_dir')
    output_dir = os.path.join(os.path.dirname(os.getcwd()),iemocap_spectrogram_dir)
    log_file_path = os.path.join(log_dir,'processed_files_spectrogram.log')
    error_log_path = os.path.join(log_dir,'error_files_spectrogram.log')

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_files = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            processed_files = set(file.read().splitlines())

    processed_files_count = 0
    throttle_delay = 1 

    for filenum in tqdm(range(len(audio_files))):
        filename = audio_files[filenum]
        if filename.endswith(".wav") and filename not in processed_files:

            audio_file_path = os.path.join(filename)
            output_file_path = os.path.join(output_dir, os.path.splitext(os.path.basename(filename))[0])
            try:
                create_spectrogram_helper(filename, audio_file_path, output_file_path)
                processed_files.add(filename)
                processed_files_count += 1
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{filename}\n")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                with open(error_log_path, 'a') as error_log:
                    error_log.write(f"{filename}: {e}\n")
            finally:
                time.sleep(throttle_delay)

    print(f"Batch conversion completed for spectrograms. Processed {processed_files_count} files.")

def log_specgram(audio, sample_rate, window_size=20,
                        step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio, fs=sample_rate, window='hann', 
                                            nperseg=nperseg, noverlap=noverlap, detrend=False)
    return freqs, np.log(spec.T.astype(np.float32) + eps)

def process_audio_file(filepath, output_dir):
    sample_rate, audio = wavfile.read(filepath)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    _, spectrogram = log_specgram(audio, sample_rate)
    plt.figure(figsize=(10, 4))  
    plt.xticks([])
    plt.yticks([])
    plt.imshow(spectrogram.T, aspect='auto', origin='lower')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, os.path.splitext(os.path.basename(filepath))[0]+".png"))
    plt.close()  

def create_log_spectrograms(iemocap_log_spectrogram_dir, audio_files):

    log_dir = os.path.join(os.path.dirname(os.getcwd()),'iemocap','log_dir')
    output_dir = os.path.join(os.path.dirname(os.getcwd()),iemocap_log_spectrogram_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, 'processed_files_log_spectrogram.log')
    error_log_path = os.path.join(log_dir, 'error_files_log_spectrogram.log')

    throttle_delay = 1 

    processed_files = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as file:
            processed_files = set(file.read().splitlines())

    processed_files_count = 0
    for filenum in tqdm(range(len(audio_files))):
        filepath = audio_files[filenum]
        if filepath.endswith(".wav") and filepath not in processed_files:
            try:
                process_audio_file(filepath, output_dir)
                processed_files.add(filepath)
                processed_files_count += 1
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"{filepath}\n")
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                with open(error_log_path, 'a') as error_log:
                    error_log.write(f"{filepath}: {e}\n")
            finally:
                time.sleep(throttle_delay)

    print(f"Batch conversion completed for log spectrograms. Processed {processed_files_count} files.")

def augment_audio_and_save(input_audio_path, augmented_audio_path):
    audio, sr = librosa.load(input_audio_path)
    audio = np.expand_dims(audio,axis=(0,1))
    audio = torch.tensor(audio, device="cuda")
    apply_augmentation = Compose(
        transforms=[
            Gain(
                min_gain_in_db=-15.0,
                max_gain_in_db=5.0,
                p=0.35,
            ),
            AddColoredNoise(p=0.25),
            PitchShift(p=0.5, sample_rate=sr),
        ]
    )
    perturbed_audio_samples = apply_augmentation(audio, sample_rate=sr)
    perturbed_audio_samples = np.squeeze(np.array(perturbed_audio_samples.cpu()))
    sf.write(augmented_audio_path,perturbed_audio_samples,samplerate=sr)
