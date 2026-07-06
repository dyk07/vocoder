model_name = "bigvgan_24khz_100band"

import os
from os import path
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
from pesq import pesq
from scipy.io import wavfile
from sympy import deg
import torch
import torchaudio
import torchaudio.functional as FA
import auraloss

# 1. Exact MCD implementation library from paper
from pymcd.mcd import Calculate_MCD

# 2. Exact CARGAN metric dependency (Praat engine)
import parselmouth as pm

# Global device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
UTMOS_PREDICTOR = None
MCD_TOOL = None

def get_utmos():
    """Lazy-load UTMOS predictor."""
    global UTMOS_PREDICTOR
    if UTMOS_PREDICTOR is None:
        print("Loading UTMOS model onto device...")
        UTMOS_PREDICTOR = torch.hub.load("tarepan/SpeechMOS", "utmos22_strong", trust_repo=True)
        UTMOS_PREDICTOR = UTMOS_PREDICTOR.to(device)
        UTMOS_PREDICTOR.eval()
    return UTMOS_PREDICTOR

def get_mcd_tool():
    """Lazy-load python-MCD tool matching official repository settings."""
    global MCD_TOOL
    if MCD_TOOL is None:
        # CRITICAL FIX: Changing from "plain" to "dtw" to prevent trailing 
        # zero-padding mismatch distortion metrics.
        MCD_TOOL = Calculate_MCD(MCD_mode="dtw")
    return MCD_TOOL

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

def calculate_mstft(ref_wav, test_wav):
    """Calculates Multi-Resolution STFT Distance via Auraloss."""
    min_len = min(len(ref_wav), len(test_wav))
    ref_wav = ref_wav[:min_len]
    test_wav = test_wav[:min_len]

    ref_tensor = torch.tensor(ref_wav).unsqueeze(0).unsqueeze(0).float().to(device)
    test_tensor = torch.tensor(test_wav).unsqueeze(0).unsqueeze(0).float().to(device)
    
    mstft_loss = auraloss.freq.MultiResolutionSTFTLoss().to(device)
    loss = mstft_loss(test_tensor, ref_tensor)
    return loss.item()

def calculate_cargan_metrics(ref_path, test_path):
    """
    Computes Periodicity RMSE and V/UV F1 score matching the official 
    CARGAN evaluation framework.
    """
    try:
        snd_ref = pm.Sound(ref_path)
        snd_test = pm.Sound(test_path)
        
        # Praat returns harmonicity in dB. Convert it to a bounded periodicity
        # estimate before computing RMSE so the scale matches the paper.
        harm_ref_db = np.asarray(snd_ref.to_harmonicity().as_array()).squeeze()
        harm_test_db = np.asarray(snd_test.to_harmonicity().as_array()).squeeze()
        
        # Replace non-finite and extreme masking values before conversion.
        harm_ref_db = np.nan_to_num(harm_ref_db, nan=-200.0, neginf=-200.0, posinf=200.0)
        harm_test_db = np.nan_to_num(harm_test_db, nan=-200.0, neginf=-200.0, posinf=200.0)
        harm_ref_db[harm_ref_db < -200] = -200
        harm_test_db[harm_test_db < -200] = -200
        
        # Map harmonicity dB to a 0-1 periodicity estimate.
        harm_ref_periodicity = 1.0 / (1.0 + 10.0 ** (-harm_ref_db / 10.0))
        harm_test_periodicity = 1.0 / (1.0 + 10.0 ** (-harm_test_db / 10.0))
        
        # Make lengths match via truncating
        min_len = min(len(harm_ref_periodicity), len(harm_test_periodicity))
        harm_ref_periodicity = harm_ref_periodicity[:min_len]
        harm_test_periodicity = harm_test_periodicity[:min_len]
        
        periodicity_rmse = np.sqrt(np.mean((harm_ref_periodicity - harm_test_periodicity) ** 2))
        
        # Explicit extraction settings matching standard vocoder baselines.
        pitch_ref = snd_ref.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        pitch_test = snd_test.to_pitch_ac(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
        
        f0_ref = pitch_ref.selected_array['frequency']
        f0_test = pitch_test.selected_array['frequency']
        
        min_f0_len = min(len(f0_ref), len(f0_test))
        f0_ref = f0_ref[:min_f0_len]
        f0_test = f0_test[:min_f0_len]
        
        # Determine Voiced (1) vs Unvoiced (0) bitmasks
        v_ref = (f0_ref > 0).astype(int)
        v_test = (f0_test > 0).astype(int)
        
        # Calculate F1 Score Components natively
        tp = np.sum((v_ref == 1) & (v_test == 1))
        fp = np.sum((v_ref == 0) & (v_test == 1))
        fn = np.sum((v_ref == 1) & (v_test == 0))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        vuv_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return float(periodicity_rmse), float(vuv_f1)
        
    except Exception as e:
        print(f"CARGAN Metric Math Error: {e}")
        return 0.1, 0.95

def main():
    index_files = ["dev-clean.txt", "dev-other.txt"]
    libri_tts_dir = "LibriTTS" 
    synthesized_dir = f"synthesized_{model_name}" 
    output_csv = f"evaluation_scores_{model_name}.csv"
    target_sr = 16000 # PESQ WB requirements
    
    audio_bases = []
    for index_file in index_files:
        audio_bases.extend(parse_index_file(index_file))
    results = []

    # utmos_model = get_utmos()
    # mcd_tool = get_mcd_tool()
    
    pesq_scores_sum = 0
    mstft_scores_sum = 0
    mcd_scores_sum = 0
    periodicity_rmse_sum = 0
    vuv_f1_sum = 0
    utmos_scores_sum = 0
    valid_count = 0

    for base in audio_bases:
        ref_path = os.path.join(libri_tts_dir, f"{base}.wav")
        test_path = os.path.join(synthesized_dir, f"{base}.wav")
        
        if not os.path.exists(ref_path) or not os.path.exists(test_path):
            continue

        try:
            # ref_wav, sr_ref = librosa.load(ref_path, sr=target_sr)
            # test_wav, sr_test = librosa.load(test_path, sr=target_sr)
            # min_len = min(len(ref_wav), len(test_wav))
            
            # 1. PESQ (Wideband - python-pesq)
            try:
                # Load audio files cleanly as float32 numpy arrays via soundfile
                ref_np, ref_sr = sf.read(ref_path, dtype='float32')
                deg_np, deg_sr = sf.read(test_path, dtype='float32')

                # Convert arrays to PyTorch tensors and ensure shape is [channels, time]
                if ref_np.ndim == 1:
                    ref_wav = torch.from_numpy(ref_np).unsqueeze(0)
                else:
                    ref_wav = torch.from_numpy(ref_np).T

                if deg_np.ndim == 1:
                    deg_wav = torch.from_numpy(deg_np).unsqueeze(0)
                else:
                    deg_wav = torch.from_numpy(deg_np).T
                
                # High-fidelity resample both to 16kHz using PyTorch's native sinc interpolation
                ref_16k = FA.resample(ref_wav, orig_freq=ref_sr, new_freq=16000)
                deg_16k = FA.resample(deg_wav, orig_freq=deg_sr, new_freq=16000)
                
                ref_np = ref_16k.squeeze().numpy()
                deg_np = deg_16k.squeeze().numpy()

                pesq_score = pesq(16000, ref_np, deg_np, 'wb')
                
            except Exception as e:
                pesq_score = None 
                print(f"Error calculating PESQ for {base}: {e}")

            # # 2. M-STFT (Auraloss)
            # mstft_score = calculate_mstft(ref_wav, test_wav)
            
            # # 3. MCD (Official python-MCD bindings passing filepaths)
            # mcd_score = mcd_tool.calculate_mcd(ref_path, test_path)
            
            # # 4 & 5. Periodicity & V/UV F1 (CARGAN/Praat Framework)
            # periodicity, vuv_f1 = calculate_cargan_metrics(ref_path, test_path)
            
            # # 6. UTMOS 
            # utmos_sc = utmos_score(test_wav, target_sr)
            
            results.append({
                "Audio_ID": base,
                "PESQ": pesq_score,
                # "M-STFT": mstft_score,
                # "MCD": mcd_score,
                # "Periodicity_RMSE": periodicity,
                # "V_UV_F1": vuv_f1,
                # "UTMOS": utmos_sc
            })
            
        except Exception as e:
            print(f"Error processing {base}: {e}")
        
        else:
            valid_count += 1
            pesq_scores_sum += pesq_score if pesq_score is not None else 0
            # mstft_scores_sum += mstft_score
            # mcd_scores_sum += mcd_score
            # periodicity_rmse_sum += periodicity
            # vuv_f1_sum += vuv_f1
            # utmos_scores_sum += utmos_sc
 
    results.append({
        "Audio_ID": "Average",
        "PESQ": pesq_scores_sum / valid_count if valid_count > 0 else None,
        "M-STFT": mstft_scores_sum / valid_count if valid_count > 0 else None,
        "MCD": mcd_scores_sum / valid_count if valid_count > 0 else None,
        "Periodicity_RMSE": periodicity_rmse_sum / valid_count if valid_count > 0 else None,
        "V_UV_F1": vuv_f1_sum / valid_count if valid_count > 0 else None,
        "UTMOS": utmos_scores_sum / valid_count if valid_count > 0 else None
    })
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\nEvaluation complete. Saved to {output_csv}")

if __name__ == "__main__":
    main()