import torchcrepe
import torch
import functools

def from_audio(audio, target_length, hopsize):
    """Preprocess pitch from audio"""

    # Resample hopsize
    # Estimate pitch

    audio = audio.unsqueeze(0)
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
        pad=False)

    # Set low energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False)

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity

def p_p_F(threshold, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
    true_threshold = threshold(true_pitch, true_periodicity)
    pred_threshold = threshold(pred_pitch, pred_periodicity)
    true_voiced = ~torch.isnan(true_threshold)
    pred_voiced = ~torch.isnan(pred_threshold)

    # Update periodicity rmse
    count = true_pitch.shape[1]
    periodicity_total = (true_periodicity - pred_periodicity).pow(2).sum()

    # Update pitch rmse
    voiced = true_voiced & pred_voiced
    voiced_sum = voiced.sum()

    difference_cents = 1200 * (torch.log2(true_pitch[voiced]) -
                               torch.log2(pred_pitch[voiced]))
    pitch_total = difference_cents.pow(2).sum()

    # Update voiced/unvoiced precision and recall
    true_positives = (true_voiced & pred_voiced).sum()
    false_positives = (~true_voiced & pred_voiced).sum()
    false_negatives = (true_voiced & ~pred_voiced).sum()

    pitch_rmse = torch.sqrt(pitch_total / voiced_sum)
    periodicity_rmse = torch.sqrt(periodicity_total / count)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    return pitch_rmse.nan_to_num().item(), periodicity_rmse.item(), f1.nan_to_num().item()