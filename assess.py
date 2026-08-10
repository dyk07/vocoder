import functools
import os
import traceback
import numpy as np
import pandas as pd
from pesq import pesq
from scipy.spatial.distance import euclidean
import torch
import torchaudio
from fastdtw import fastdtw
import auraloss

from evaluate import load_wav, readmgc
from cargan.evaluate.objective.metrics import Pitch
from cargan.preprocess.pitch import from_audio

# General configuration
model_name = "bigvgan_base_24khz_100band"
index_files = ["dev-clean.txt", "dev-other.txt"]
libri_tts_dir = "LibriTTS"
synthesized_dir = f"synthesized_{model_name}" 
output_csv = f"evaluation_scores_{model_name}.csv"
target_sr = 16000  # PESQ WB requirements
SR_TARGET = 24000  # Native vocoder sample rate
MAX_WAV_VALUE = 32768.0

# Global device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
UTMOS_PREDICTOR = None

def get_utmos():
    """Lazy-load UTMOS predictor."""
    global UTMOS_PREDICTOR
    if UTMOS_PREDICTOR is None:
        print("Loading UTMOS model onto device...")
        # CRITICAL: Disable online checking to prevent hanging on GitHub's connection check
        UTMOS_PREDICTOR = torch.hub.load(
            "tarepan/SpeechMOS", 
            "utmos22_strong", 
            trust_repo=True, 
            force_reload=False,
            skip_validation=True
        )
        UTMOS_PREDICTOR = UTMOS_PREDICTOR.to(device)
        UTMOS_PREDICTOR.eval()
    return UTMOS_PREDICTOR


def utmos_score(audio, sr):
    """Calculate UTMOS score utilizing device acceleration."""
    predictor = get_utmos()
    with torch.inference_mode():
        wav_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        score = predictor(wav_tensor, sr)
    return score.mean().item()


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
    """Perform objective evaluation for a single audio file pair"""
    gpu = 0 if torch.cuda.is_available() else None
    eval_device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    torch.cuda.empty_cache()

    resampler_16k = torchaudio.transforms.Resample(SR_TARGET, 16000).to(eval_device)
    resampler_22k = torchaudio.transforms.Resample(SR_TARGET, 22050).to(eval_device)

    # Modules for evaluation metrics
    loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss().to(eval_device)
    batch_metrics_periodicity = Pitch()
    periodicity_fn = functools.partial(from_audio, gpu=gpu)

    with torch.no_grad():
        # Load individual files
        y = load_wav(gt_path).to(eval_device)
        y_g_hat = load_wav(synth_path).to(eval_device)

        min_len = min(y.shape[-1], y_g_hat.shape[-1])
        y = y[:, :min_len]
        y_g_hat = y_g_hat[:, :min_len]

        # Resample
        y_16k = resampler_16k(y)
        y_g_hat_16k = resampler_16k(y_g_hat)

        y_22k = resampler_22k(y)
        y_g_hat_22k = resampler_22k(y_g_hat)

        # MRSTFT calculation
        mrstft_val = loss_mrstft(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()

        # PESQ calculation
        y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        pesq_val = pesq(16000, y_int_16k, y_g_hat_int_16k, 'wb')

        # MCD calculation
        y_double_22k = (y_22k[0] * MAX_WAV_VALUE).double().cpu().numpy()
        y_g_hat_double_22k = (y_g_hat_22k[0] * MAX_WAV_VALUE).double().cpu().numpy()

        y_mgc = readmgc(y_double_22k)
        y_g_hat_mgc = readmgc(y_g_hat_double_22k)

        _, path_dtw = fastdtw(y_mgc, y_g_hat_mgc, dist=euclidean)

        y_path = list(map(lambda l: l[0], path_dtw))
        y_g_hat_path = list(map(lambda l: l[1], path_dtw))
        y_mgc = y_mgc[y_path]
        y_g_hat_mgc = y_g_hat_mgc[y_g_hat_path]

        frames = y_mgc.shape[0]

        z = y_mgc - y_g_hat_mgc
        s = np.sqrt((z * z).sum(-1)).sum()
        mcd_val = 10.0 / np.log(10.0) * np.sqrt(2.0) * float(s) / float(frames)

        # Periodicity calculation
        true_pitch, true_periodicity = periodicity_fn(y_22k)
        pred_pitch, pred_periodicity = periodicity_fn(y_g_hat_22k)
        batch_metrics_periodicity.update(true_pitch, true_periodicity, pred_pitch, pred_periodicity)

    results = batch_metrics_periodicity()

    return {
        'M-STFT': mrstft_val,
        'PESQ': pesq_val,
        'MCD': mcd_val,
        'Periodicity': results['periodicity'],
        'V/UV F1': results['f1'],
    }


def main():
    audio_bases = []
    for index_file in index_files:
        audio_bases.extend(parse_index_file(index_file))
    
    results = []
    pesq_scores_sum = 0
    mstft_scores_sum = 0
    mcd_scores_sum = 0
    periodicity_sum = 0
    vuv_f1_sum = 0
    utmos_scores_sum = 0
    valid_count = 0
    pesq_valid_count = 0

    for idx, base in enumerate(audio_bases):
        ref_path = os.path.join(libri_tts_dir, f"{base}.wav")
        test_path = os.path.join(synthesized_dir, f"{base}.wav")
        
        if not os.path.exists(ref_path) or not os.path.exists(test_path):
            continue

        print(f"\n[{idx + 1}/{len(audio_bases)}] Processing: {base}")

        try:
            # Main evaluations
            results_dict = evaluate(ref_path, test_path)
            results_dict["Audio_ID"] = base

            # UTMOS calculation
            print("  --> Calculating UTMOS...")
            synth_wav = load_wav(test_path).squeeze().numpy()
            utmos_sc = utmos_score(synth_wav, SR_TARGET)
            results_dict["UTMOS"] = utmos_sc

            results.append(results_dict)
            print("[SUCCESS]")
            
            valid_count += 1
            if results_dict.get("PESQ") is not None:
                pesq_scores_sum += results_dict["PESQ"]
                pesq_valid_count += 1
            
            mstft_scores_sum += results_dict["M-STFT"]
            mcd_scores_sum += results_dict["MCD"]
            periodicity_sum += results_dict["Periodicity"]
            vuv_f1_sum += results_dict["V/UV F1"]
            utmos_scores_sum += results_dict["UTMOS"]

        except Exception as e:
            print(f"Error processing {base}: {e}")
            traceback.print_exc()

    # Append average metrics summary row
    results.append({
        "Audio_ID": "Average",
        "PESQ": pesq_scores_sum / pesq_valid_count if pesq_valid_count > 0 else None,
        "M-STFT": mstft_scores_sum / valid_count if valid_count > 0 else None,
        "MCD": mcd_scores_sum / valid_count if valid_count > 0 else None,
        "Periodicity": periodicity_sum / valid_count if valid_count > 0 else None,
        "V/UV F1": vuv_f1_sum / valid_count if valid_count > 0 else None,
        "UTMOS": utmos_scores_sum / valid_count if valid_count > 0 else None
    })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Saved to {output_csv}")


if __name__ == "__main__":
    main()