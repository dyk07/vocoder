import os
import traceback
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import auraloss
from pesq import pesq
import torchcrepe
from librosa.util import normalize

from PeriodWave import utils
from PeriodWave.meldataset_prior_length import mel_spectrogram, load_wav, MAX_WAV_VALUE
from PeriodWave.pitch_periodicity import from_audio, p_p_F

# ==============================================================================
# General Configuration (from assess.py interface)
# ==============================================================================
model_name = "periodwave_midpoint_16_0.667"
index_files = ["dev-clean.txt", "dev-other.txt"]
list_dir = "C:\\Users\\Kelvin\\Documents\\GitHub\\BigVGAN\\filelists\\LibriTTS"  # Path to index files
libri_tts_dir = "LibriTTS"
synthesized_dir = f"synthesized_{model_name}" 
output_csv = f"evaluation_scores_{model_name}.csv"
config_path = "configs/periodwave_24000hz.json"

MAX_WAV_VALUE = 32768.0
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load hps config for Mel Spectrogram parameters
if os.path.exists(config_path):
    hps = utils.get_hparams_from_file(config_path)
else:
    # Fallback default values if config is not located
    class HparamsData:
        sampling_rate = 24000
        filter_length = 1024
        n_mel_channels = 100
        hop_length = 256
        win_length = 1024
        mel_fmin = 0.0
        mel_fmax = 12000.0
    class Hparams:
        data = HparamsData()
    hps = Hparams()

# Initialize static evaluation tools from inference_with_evaluation.py
pesq_resampler = torchaudio.transforms.Resample(hps.data.sampling_rate, 16000).to(device)
loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)
threshold_fn = torchcrepe.threshold.Hysteresis()

UTMOS_PREDICTOR = None

def get_utmos():
    """Lazy-load UTMOS predictor as done in inference_with_evaluation.py."""
    global UTMOS_PREDICTOR
    if UTMOS_PREDICTOR is None:
        print("Loading UTMOS model onto device...")
        UTMOS_PREDICTOR = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", 
            "utmos22_strong", 
            trust_repo=True,
            skip_validation=True
        ).to(device)
        UTMOS_PREDICTOR.eval()
    return UTMOS_PREDICTOR


def parse_index_file(index_path):
    audio_bases = []
    if not os.path.exists(index_path):
        print(f"Warning: Index file {index_path} not found.")
        return audio_bases
    with open(index_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '|' not in line:
                continue
            base_path = line.split('|')[0] 
            audio_bases.append(base_path)
    return audio_bases


def evaluate(gt_path, synth_path):
    """
    Evaluates ground truth vs synthesized audio pairs using metric calculations
    from inference_with_evaluation.py.
    """
    with torch.no_grad():
        # 1. Load and preprocess Ground Truth audio (from inference_with_evaluation.py)
        audio_np, _ = load_wav(gt_path, hps.data.sampling_rate)
        audio_np = audio_np / MAX_WAV_VALUE
        audio_np = normalize(audio_np) * 0.95

        audio = torch.FloatTensor(audio_np).unsqueeze(0).to(device)
        if (audio.size(1) % hps.data.hop_length) != 0:
            audio = audio[:, :-(audio.size(1) % hps.data.hop_length)]

        # 2. Load and preprocess Synthesized audio (from inference_with_evaluation.py)
        resynthesis_audio_np, _ = load_wav(synth_path, hps.data.sampling_rate)
        resynthesis_audio_np = resynthesis_audio_np / MAX_WAV_VALUE
        resynthesis_audio = torch.FloatTensor(resynthesis_audio_np).unsqueeze(0).to(device)

        if torch.abs(resynthesis_audio).max() >= 0.95:
            resynthesis_audio = (resynthesis_audio / (torch.abs(resynthesis_audio).max())) * 0.95

        # Align length to prevent tensor shape mismatches
        min_len = min(audio.shape[-1], resynthesis_audio.shape[-1])
        audio = audio[:, :min_len]
        resynthesis_audio = resynthesis_audio[:, :min_len]

        # 3. Mel Spectrogram & Mel L1 Loss
        mel = mel_spectrogram(
            audio, hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax, center=False
        )
        mel_hat = mel_spectrogram(
            resynthesis_audio, hps.data.filter_length, hps.data.n_mel_channels,
            hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
            hps.data.mel_fmin, hps.data.mel_fmax, center=False
        )
        mel_l1 = F.l1_loss(mel, mel_hat).item()

        # 4. Multi-Resolution STFT Loss
        mrstft_val = loss_mrstft(resynthesis_audio.unsqueeze(1), audio.unsqueeze(1)).item()

        # 5. Resampling to 16kHz for PESQ, Pitch & UTMOS
        y_16k = pesq_resampler(audio)
        y_g_hat_16k = pesq_resampler(resynthesis_audio)

        # 6. Pitch, Periodicity & V/UV F1
        hopsize = int(256 * (torchcrepe.SAMPLE_RATE / 24000))
        padding = int((1024 - hopsize) // 2)

        audio_for_pitch = F.pad(y_16k, (padding, padding), mode='reflect').squeeze(0)
        gen_audio_for_pitch = F.pad(y_g_hat_16k, (padding, padding), mode='reflect').squeeze(0)

        ori_audio_len = audio.shape[-1] // 256
        true_pitch, true_periodicity = from_audio(audio_for_pitch.squeeze(), ori_audio_len, hopsize)
        fake_pitch, fake_periodicity = from_audio(gen_audio_for_pitch.squeeze(), ori_audio_len, hopsize)

        pitch, periodicity, f1 = p_p_F(threshold_fn, true_pitch, true_periodicity, fake_pitch, fake_periodicity)

        # 7. PESQ Wide-Band
        y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        pesq_wb = pesq(16000, y_int_16k, y_g_hat_int_16k, 'wb')

        # 8. UTMOS Score
        utmos_predictor = get_utmos()
        utmos_val = utmos_predictor(y_g_hat_16k, 16000).item()

    return {
        'Mel L1': mel_l1,
        'MR-STFT': mrstft_val,
        'PESQ WB': pesq_wb,
        'Pitch': pitch,
        'Periodicity': periodicity,
        'V/UV F1': f1,
        'UTMOS': utmos_val
    }


def main():
    audio_bases = []
    for index_file in index_files:
        index_path = os.path.join(list_dir, index_file) if os.path.exists(list_dir) else index_file
        audio_bases.extend(parse_index_file(index_path))
    
    results = []
    totals = {
        'Mel L1': 0.0,
        'MR-STFT': 0.0,
        'PESQ WB': 0.0,
        'Pitch': 0.0,
        'Periodicity': 0.0,
        'V/UV F1': 0.0,
        'UTMOS': 0.0
    }
    valid_count = 0

    for idx, base in enumerate(audio_bases):
        ref_path = os.path.join(libri_tts_dir, f"{base}.wav")
        test_path = os.path.join(synthesized_dir, f"{base}.wav")
        
        if not os.path.exists(ref_path) or not os.path.exists(test_path):
            continue

        print(f"\n[{idx + 1}/{len(audio_bases)}] Processing: {base}")

        try:
            results_dict = {"Audio_ID": base}
            eval_metrics = evaluate(ref_path, test_path)
            results_dict.update(eval_metrics)

            results.append(results_dict)
            valid_count += 1

            for key in totals:
                totals[key] += eval_metrics[key]

            print("[SUCCESS]")

        except Exception as e:
            print(f"Error processing {base}: {e}")
            traceback.print_exc()

    # Calculate and append Average row
    if valid_count > 0:
        avg_dict = {"Audio_ID": "Average"}
        for key in totals:
            avg_dict[key] = totals[key] / valid_count
        results.append(avg_dict)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Saved results to {output_csv}")


if __name__ == "__main__":
    main()