import functools
import os
import sys
import types
import traceback
import numpy as np
import pandas as pd
import auraloss
import pysptk
import librosa

import torch
import torchaudio
import torchcrepe

from pesq import pesq
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.io.wavfile import read

from visqol import visqol_lib_py
from visqol.pb2 import similarity_result_pb2
from visqol.pb2 import visqol_config_pb2

# The native binding imports this generated module using the repository's
# source-tree name, while the installed package exposes it under visqol.pb2.
src_module = types.ModuleType("src")
proto_module = types.ModuleType("src.proto")
src_module.proto = proto_module
proto_module.similarity_result_pb2 = similarity_result_pb2
sys.modules.setdefault("src", src_module)
sys.modules.setdefault("src.proto", proto_module)
sys.modules.setdefault("src.proto.similarity_result_pb2", similarity_result_pb2)

# General configuration
model_name = "flow2gan_4step"  # Change this to the model you want to evaluate
index_files = ["dev-clean.txt", "dev-other.txt"]
libri_tts_dir = "LibriTTS"
synthesized_dir = os.path.join("synthesized", f"synthesized_{model_name}")
output_csv = os.path.join("results", f"evaluation_scores_{model_name}.csv")
target_sr = 16000  # PESQ WB & CREPE requirement
SR_TARGET = 24000  # Native vocoder sample rate
MAX_WAV_VALUE = 32768.0
GT_CLAMPING = False  # Enable/Disable amplitude clamping for ground truth audio

# Global device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
UTMOS_PREDICTOR = None

# Global metric initialization (from assess_periodwave.py)
loss_mrstft = auraloss.freq.MultiResolutionSTFTLoss(device=device)

# ==============================================================================
# Pitch, Periodicity & V/UV F1 Metric Functions (Integrated from pitch_periodicity.py)
# ==============================================================================

def from_audio(audio, hopsize=160, target_length=None):
    """Preprocess pitch and periodicity from audio using torchcrepe."""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    # Estimate pitch and periodicity via torchcrepe at 16kHz
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        fmin=50,
        fmax=550,
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device=audio.device,
        pad=False
    )

    # Set low-energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False
    )

    # Resize tensors if target_length is provided and mismatch occurs
    if target_length is not None and pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity


def p_p_F(threshold, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
    """Calculate Pitch RMSE (in cents), Periodicity RMSE, and Voiced/Unvoiced F1 Score."""
    true_threshold = threshold(true_pitch, true_periodicity)
    pred_threshold = threshold(pred_pitch, pred_periodicity)
    true_voiced = ~torch.isnan(true_threshold)
    pred_voiced = ~torch.isnan(pred_threshold)

    # Update periodicity RMSE
    count = true_pitch.shape[1]
    periodicity_total = (true_periodicity - pred_periodicity).pow(2).sum()

    # Update pitch RMSE (calculated on voiced frames in cents)
    voiced = true_voiced & pred_voiced
    voiced_sum = voiced.sum()

    if voiced_sum > 0:
        difference_cents = 1200 * (torch.log2(true_pitch[voiced]) - torch.log2(pred_pitch[voiced]))
        pitch_total = difference_cents.pow(2).sum()
        pitch_rmse = torch.sqrt(pitch_total / voiced_sum)
    else:
        pitch_rmse = torch.tensor(0.0)

    # Update voiced/unvoiced precision, recall, and F1
    true_positives = (true_voiced & pred_voiced).sum()
    false_positives = (~true_voiced & pred_voiced).sum()
    false_negatives = (true_voiced & ~pred_voiced).sum()

    periodicity_rmse = torch.sqrt(periodicity_total / count)
    
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return pitch_rmse.nan_to_num().item(), periodicity_rmse.item(), f1.nan_to_num().item()


# ==============================================================================
# ViSQOL Function (Integrated from compute_pesq_visqol.py)
# ==============================================================================

def visqol(
    estimate: np.ndarray,
    reference: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "speech",
):  

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        assert sample_rate == target_sr
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        assert sample_rate == target_sr
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    ).encode("utf-8")

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    _visqol = api.Measure(reference.astype(np.float64), estimate.astype(np.float64))
    return _visqol.moslqo


# ==============================================================================
# Helper Functions & Evaluation Logic
# ==============================================================================

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


def load_wav(full_path):
    sampling_rate, audio = read(full_path)
    if sampling_rate != SR_TARGET:
        raise IOError(
            f'Sampling rate of the file {full_path} is {sampling_rate} Hz, but the model requires {SR_TARGET} Hz'
        )

    audio = audio / MAX_WAV_VALUE

    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    return audio


def readmgc(x):
    frame_length = 1024
    hop_length = 256
    # Windowing
    frames = librosa.util.frame(x, frame_length=frame_length, hop_length=hop_length).astype(np.float64).T
    frames *= pysptk.blackman(frame_length)
    assert frames.shape[1] == frame_length
    # Order of mel-cepstrum
    order = 25
    alpha = 0.41
    stage = 5
    gamma = -1.0 / stage

    mgc = pysptk.mgcep(frames, order, alpha, gamma)
    mgc = mgc.reshape(-1, order + 1)
    return mgc


def evaluate(gt_path, synth_path):
    """Perform objective evaluation for a single audio file pair."""
    gpu = 0 if torch.cuda.is_available() else None
    eval_device = torch.device('cpu' if gpu is None else f'cuda:{gpu}')
    torch.cuda.empty_cache()

    resampler_16k = torchaudio.transforms.Resample(SR_TARGET, 16000).to(eval_device)
    resampler_22k = torchaudio.transforms.Resample(SR_TARGET, 22050).to(eval_device)

    with torch.no_grad():
        # Load audio signals
        y = load_wav(gt_path).to(eval_device)
        y_g_hat = load_wav(synth_path).to(eval_device)

        # ==============================================================================
        # Added Amplitude Capping and Length Truncation
        # ==============================================================================
        # 1. Normalize Ground Truth to exactly 0.95 peak
        if GT_CLAMPING == True:
            y_max = torch.abs(y).max()
            if y_max > 0:
                y = (y / y_max) * 0.95
  
            # 3. Cap Synthesized Audio at 0.95 peak (Directly affects UTMOS)
            if torch.abs(y_g_hat).max() >= 0.95:
                y_g_hat = (y_g_hat / torch.abs(y_g_hat).max()) * 0.95
            # ==============================================================================

        
        min_len = min(y.shape[-1], y_g_hat.shape[-1])
        y = y[:, :min_len]
        y_g_hat = y_g_hat[:, :min_len]

        # Resample for various metric calculations
        y_16k = resampler_16k(y)
        y_g_hat_16k = resampler_16k(y_g_hat)

        y_22k = resampler_22k(y)
        y_g_hat_22k = resampler_22k(y_g_hat)

        # 1. Multi-Resolution STFT Loss (using global initialization logic from assess_periodwave.py)
        mrstft_val = loss_mrstft(y_g_hat.unsqueeze(1), y.unsqueeze(1)).item()

        # 2. PESQ (16kHz)
        y_int_16k = (y_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        y_g_hat_int_16k = (y_g_hat_16k[0] * MAX_WAV_VALUE).short().cpu().numpy()
        pesq_val = pesq(16000, y_int_16k, y_g_hat_int_16k, 'wb')
        
        # 3. UTMOS Score (using assess_periodwave.py logic directly on 16kHz tensor)
        utmos_predictor = get_utmos()
        utmos_val = utmos_predictor(y_g_hat_16k, 16000).item()

        # 4. MCD (22.05kHz)
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

        # 5. Pitch, Periodicity & V/UV F1
        hopsize = int(256 * (torchcrepe.SAMPLE_RATE / SR_TARGET))
        target_len = y.shape[-1] // 256
        padding = (1024 - hopsize) // 2

        y_16k_pad = torch.nn.functional.pad(y_16k, (padding, padding), mode='reflect')
        y_g_hat_16k_pad = torch.nn.functional.pad(y_g_hat_16k, (padding, padding), mode='reflect')

        true_pitch, true_periodicity = from_audio(y_16k_pad.squeeze(0), hopsize=hopsize, target_length=target_len)
        pred_pitch, pred_periodicity = from_audio(y_g_hat_16k_pad.squeeze(0), hopsize=hopsize, target_length=target_len)

        threshold_fn = torchcrepe.threshold.Hysteresis()
        pitch_rmse, periodicity_rmse, vuv_f1 = p_p_F(
            threshold_fn, true_pitch, true_periodicity, pred_pitch, pred_periodicity
        )

        # 6. ViSQOL (Evaluated using 16kHz float representations)
        if y_16k.shape[-1] < 16000:
            pad_len = 16000 - y_16k.shape[-1]
            y_g_hat_16k_visqol = torch.nn.functional.pad(y_g_hat_16k[0], (0, pad_len), value=0.0)
            y_16k_visqol = torch.nn.functional.pad(y_16k[0], (0, pad_len), value=0.0)
        else:
            y_g_hat_16k_visqol = y_g_hat_16k[0]
            y_16k_visqol = y_16k[0]

        try:
            visqol_val = visqol(
                estimate=y_g_hat_16k_visqol.cpu().numpy(),
                reference=y_16k_visqol.cpu().numpy(),
                mode="speech",
                sample_rate=16000,
            )
        except Exception as e:
            print(f"  --> ViSQOL evaluation error: {e}")
            visqol_val = None

    return {
        'M-STFT': mrstft_val,
        'PESQ': pesq_val,
        'UTMOS': utmos_val,
        'ViSQOL': float(visqol_val) if visqol_val is not None else None,
        'MCD': mcd_val,
        'Pitch': pitch_rmse,
        'Periodicity': periodicity_rmse,
        'V/UV F1': vuv_f1,
    }


def main():
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    audio_bases = []
    for index_file in index_files:
        audio_bases.extend(parse_index_file(index_file))
    
    results = []
    pesq_scores_sum = 0
    visqol_scores_sum = 0
    mstft_scores_sum = 0
    mcd_scores_sum = 0
    pitch_scores_sum = 0
    periodicity_sum = 0
    vuv_f1_sum = 0
    utmos_scores_sum = 0
    
    valid_count = 0
    pesq_valid_count = 0
    visqol_valid_count = 0

    for idx, base in enumerate(audio_bases):
        ref_path = os.path.join(libri_tts_dir, f"{base}.wav")
        test_path = os.path.join(synthesized_dir, f"{base}.wav")
        
        if not os.path.exists(ref_path) or not os.path.exists(test_path):
            continue

        print(f"\n[{idx + 1}/{len(audio_bases)}] Processing: {base}")

        try:
            # Main evaluation metrics
            results_dict = {"Audio_ID": base}
            results_dict.update(evaluate(ref_path, test_path))

            results.append(results_dict)
            print("[SUCCESS]")
            
            valid_count += 1
            if results_dict.get("PESQ") is not None:
                pesq_scores_sum += results_dict["PESQ"]
                pesq_valid_count += 1

            if results_dict.get("ViSQOL") is not None:
                visqol_scores_sum += results_dict["ViSQOL"]
                visqol_valid_count += 1
            
            mstft_scores_sum += results_dict["M-STFT"]
            mcd_scores_sum += results_dict["MCD"]
            pitch_scores_sum += results_dict["Pitch"]
            periodicity_sum += results_dict["Periodicity"]
            vuv_f1_sum += results_dict["V/UV F1"]
            utmos_scores_sum += results_dict["UTMOS"]

        except Exception as e:  
            print(f"Error processing {base}: {e}")
            traceback.print_exc()

    # Append summary row with average scores
    results.append({
        "Audio_ID": "Average",
        "PESQ": pesq_scores_sum / pesq_valid_count if pesq_valid_count > 0 else None,
        "ViSQOL": visqol_scores_sum / visqol_valid_count if visqol_valid_count > 0 else None,
        "M-STFT": mstft_scores_sum / valid_count if valid_count > 0 else None,
        "MCD": mcd_scores_sum / valid_count if valid_count > 0 else None,
        "Pitch": pitch_scores_sum / valid_count if valid_count > 0 else None,
        "Periodicity": periodicity_sum / valid_count if valid_count > 0 else None,
        "V/UV F1": vuv_f1_sum / valid_count if valid_count > 0 else None,
        "UTMOS": utmos_scores_sum / valid_count if valid_count > 0 else None
    })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Saved to {output_csv}")


if __name__ == "__main__":
    main()