import torch
import numpy as np
import librosa
import json
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm
from audiomentations import (
    AddBackgroundNoise, AddGaussianNoise, Compose, Gain,
    OneOf, PitchShift, PolarityInversion, TimeStretch,
    Mp3Compression, ClippingDistortion, LowPassFilter
)
from tqdm import tqdm
import torch.nn.functional as F
from loguru import logger
from utils.scheduler import WarmupLR
from enhanced_constants import *

class EnhancedAudioDataset(Dataset):
    def __init__(self,
                 manifest_files,
                 tokenizer_model_path,
                 bg_noise_path=[],
                 shuffle=False,
                 augment=False,
                 max_duration=15.1,
                 min_duration=0.3,  # Giảm từ 0.9 xuống 0.3
                 min_text_len=1,    # Giảm từ 3 xuống 1
                 max_text_len=99999,
                 short_sentence_boost=3.0,  # Boost factor cho short sentences
                 short_threshold=2.0):      # Threshold cho short sentences (giây)

        self.samples = []
        self.short_samples = []
        self.normal_samples = []

        throw_away = 0

        # Load và categorize samples
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as f:
                for line in tqdm(f, desc=f'Loading {manifest_file}'):
                    sample = json.loads(line)

                    # Filter theo duration và text length
                    if (sample['duration'] > max_duration or
                        sample['duration'] < min_duration or
                        len(sample['text'].strip()) < min_text_len or
                        len(sample['text'].strip()) > max_text_len):
                        throw_away += 1
                        continue

                    # Categorize short vs normal sentences
                    if sample['duration'] <= short_threshold:
                        self.short_samples.append(sample)
                    else:
                        self.normal_samples.append(sample)

        logger.info(f"Short samples: {len(self.short_samples)}")
        logger.info(f"Normal samples: {len(self.normal_samples)}")
        logger.info(f"Thrown away: {throw_away} samples")

        # Boost short sentences
        boosted_short_samples = self.short_samples * int(short_sentence_boost)
        self.samples = boosted_short_samples + self.normal_samples

        logger.info(f"Total samples after boosting: {len(self.samples)}")

        if shuffle:
            np.random.shuffle(self.samples)

        self.tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_model_path)
        self.device = 'cpu'
        self.short_threshold = short_threshold

        # Enhanced augmentation với special handling cho short sentences
        if augment:
            # Light augmentation cho short sentences
            self.light_augmentation = Compose([
                Gain(min_gain_db=-10, max_gain_db=5, p=0.7),
                PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.005, p=0.5),
            ])

            # Stronger augmentation cho normal sentences
            self.strong_augmentation = Compose([
                Gain(min_gain_db=-25, max_gain_db=10, p=0.9),
                PitchShift(min_semitones=-4, max_semitones=4, p=0.4),
                TimeStretch(min_rate=0.95, max_rate=1.05, p=0.3),
                OneOf([
                    AddBackgroundNoise(sounds_path=bg_noise_path, min_snr_db=1.0, max_snr_db=8.0, p=0.8),
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.8),
                    ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=10, p=0.3),
                ]),
                Mp3Compression(p=0.2),
            ])
        else:
            self.light_augmentation = lambda samples, sample_rate: samples
            self.strong_augmentation = lambda samples, sample_rate: samples

    def mel_filters(self, device, n_mels: int) -> torch.Tensor:
        """Load mel filterbank matrix"""
        assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

        with np.load("./weights/mel_filters.npz", allow_pickle=False) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

    def log_mel_spectrogram(self, audio, n_mels, padding, device):
        """Enhanced mel spectrogram với special processing cho short audio"""
        if device is not None:
            audio = audio.to(device)

        # Adaptive padding cho short audio
        if len(audio) < SAMPLE_RATE * 0.5:  # < 0.5 seconds
            # Pad to minimum 0.5 seconds cho stability
            min_length = int(SAMPLE_RATE * 0.5)
            audio = F.pad(audio, (0, max(0, min_length - len(audio))))

        if padding > 0:
            audio = F.pad(audio, (0, padding))

        window = torch.hann_window(N_FFT).to(audio.device)
        stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        # Adaptive normalization cho short sentences
        if mel_spec.shape[1] < 50:  # Very short audio
            log_spec = (log_spec + 4.0) / 4.0
        else:
            log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def enhance_short_audio(self, waveform, duration):
        """Special processing cho short audio"""
        if duration <= self.short_threshold:
            # Aggressive silence trimming
            waveform, _ = librosa.effects.trim(
                waveform,
                top_db=25,  # More aggressive trimming
                frame_length=256,
                hop_length=64
            )

            # Ensure minimum length
            min_samples = int(0.3 * SAMPLE_RATE)  # 0.3 seconds minimum
            if len(waveform) < min_samples:
                # Repeat the audio thay vì zero padding
                repeat_times = int(np.ceil(min_samples / len(waveform)))
                waveform = np.tile(waveform, repeat_times)[:min_samples]

        return waveform

    def tokenize_with_handling(self, text):
        """Enhanced tokenization với special handling cho short text"""
        # Normalize text
        text = text.strip().lower()

        # Special tokens cho common short phrases
        short_phrases = {
            'ừ': [100],  # Special token cho "ừ"
            'à': [101],   # Special token cho "à"
            'ờ': [102],   # Special token cho "ờ"
            'ư': [103],   # Special token cho "ư"
        }

        if text in short_phrases:
            return short_phrases[text]

        # Standard tokenization
        return self.tokenizer.encode_as_ids(text)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio
        waveform, sample_rate = librosa.load(
            sample['audio_filepath'],
            sr=SAMPLE_RATE,
            offset=sample['offset'],
            duration=sample['duration']
        )

        # Enhance short audio
        waveform = self.enhance_short_audio(waveform, sample['duration'])

        # Apply appropriate augmentation
        if sample['duration'] <= self.short_threshold:
            waveform = self.light_augmentation(samples=waveform, sample_rate=sample_rate)
        else:
            waveform = self.strong_augmentation(samples=waveform, sample_rate=sample_rate)

        # Enhanced tokenization
        transcript_ids = self.tokenize_with_handling(sample['text'])

        waveform, transcript_ids = torch.from_numpy(waveform), torch.tensor(transcript_ids)

        # Generate mel spectrogram
        melspec = self.log_mel_spectrogram(waveform, N_MELS, 0, self.device)

        return melspec, transcript_ids, sample['duration']  # Return duration cho analysis

def enhanced_collate_fn(batch):
    """Enhanced collate function với special handling cho short sentences"""
    # mel, text_ids, durations = zip(*batch)

    # # Standard collation
    # max_len = max(x.shape[-1] for x in mel)
    # mel_input_lengths = torch.tensor([x.shape[-1] for x in mel])
    # text_input_lengths = torch.tensor([len(x) for x in text_ids])

    # # Pad mel spectrograms
    # mel_padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in mel]

    # # Pad text sequences
    # text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=PAD)

    # # Additional info for analysis
    # durations_tensor = torch.tensor(durations)

    # return (torch.stack(mel_padded),
    #         mel_input_lengths.int(),
    #         text_ids_padded,
    #         text_input_lengths.int(),
    #         durations_tensor)  # Extra info
    if len(batch[0]) == 3:  # If batch has duration info
        mel, text_ids, durations = zip(*batch)
    else:  # Original format
        mel, text_ids = zip(*batch)
        durations = [0.0] * len(mel)  # Dummy durations

    max_len = max(x.shape[-1] for x in mel)
    mel_input_lengths = torch.tensor([x.shape[-1] for x in mel])
    text_input_lengths = torch.tensor([len(x) for x in text_ids])
    mel_padded = [torch.nn.functional.pad(x, (0, max_len - x.shape[-1])) for x in mel]
    text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=PAD)

    # Always return 5 values for consistency
    durations_tensor = torch.tensor(durations)
    return (torch.stack(mel_padded),
            mel_input_lengths.int(),
            text_ids_padded,
            text_input_lengths.int(),
            durations_tensor)
