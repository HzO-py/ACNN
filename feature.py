import numpy as np
import scipy.signal as sps
import neurokit2 as nk

def extract_features_detailed(ppg, fs=200):
    """
    Extract the 21 features described in the paper for a single-channel PPG signal,
    and average them across multiple cycles.

    Feature list:
    1) Cardiac Period
    2) Systolic Upstroke Period
    3) Diastolic Time
    4) Systolic Width at 10, 25, 33, 50, 66, 75% (6 items)
    5) Diastolic Width at 10, 25, 33, 50, 66, 75% (6 items)
    6) Low-frequency band Amplitude / Frequency
    7) Medium-frequency band Amplitude / Frequency
    8) High-frequency band Amplitude / Frequency

    => Total 21 features

    Parameters:
    ----
    ppg : 1D numpy array
        Preprocessed single-channel PPG signal.
    fs : int
        Sampling rate in Hz, default is 200 Hz.

    Returns:
    ----
    feats : dict or None
        If at least one complete cardiac cycle is detected, returns a dict:
        {
            "CardiacPeriod": ...,
            "SystolicUpstroke": ...,
            "DiastolicTime": ...,
            "SystolicWidth_10": ...,
            ...
            "DiastolicWidth_75": ...,
            "LF_Amplitude": ...,
            "LF_Frequency": ...,
            "MF_Amplitude": ...,
            "MF_Frequency": ...,
            "HF_Amplitude": ...,
            "HF_Frequency": ...
        }
        All values are averaged across cycles.
        Returns None if no valid cycles are found.
    """

    # =========== 1. Detect peaks (systolic peaks) and valleys ============
    #   - Peaks: local maxima (systolic peaks)
    #   - Valleys: local minima (signal foot)
    # Use a minimum distance threshold to avoid noise (e.g., 0.3 s => 60 samples at 200 Hz)

    # Detect systolic peaks
    _, info = nk.ppg_peaks(ppg, sampling_rate=fs)
    peaks = np.sort(info['PPG_Peaks'])
    # Detect valleys by finding peaks in the inverted signal
    _, info2 = nk.ppg_peaks(-ppg, sampling_rate=fs)
    valleys = np.sort(info2['PPG_Peaks'])

    if len(peaks) < 2 or len(valleys) < 2:
        # Not enough peaks or valleys to form a cycle
        return None

    # =========== 2. Cycle segmentation: foot(n) -> peak(n) -> foot(n+1) ============
    cycles_info = []

    for pk_idx in peaks:
        # Find the nearest valley to the left of the peak
        left_valleys = valleys[valleys < pk_idx]
        if left_valleys.size == 0:
            continue
        foot_left = left_valleys[-1]

        # Find the nearest valley to the right of the peak
        right_valleys = valleys[valleys > pk_idx]
        if right_valleys.size == 0:
            continue
        foot_right = right_valleys[0]

        if foot_left < pk_idx < foot_right:
            cycles_info.append((foot_left, pk_idx, foot_right))

    if not cycles_info:
        return None

    # =========== 3. Compute 15 time-domain features per cycle ============
    percentages = [0.10, 0.25, 0.33, 0.50, 0.66, 0.75]
    all_cycle_feats = []

    for f_left, pk, f_right in cycles_info:
        f_left_val = ppg[f_left]
        peak_val = ppg[pk]
        f_right_val = ppg[f_right]

        # Durations in seconds
        T_cardiac = (f_right - f_left) / fs
        T_systolic = (pk - f_left) / fs
        T_diastolic = (f_right - pk) / fs

        # Amplitudes
        A_syst = peak_val - f_left_val
        A_diast = peak_val - f_right_val

        if A_syst <= 0 or A_diast <= 0:
            # Skip cycles with non-physical amplitudes
            continue

        # Systolic Width at each percentage
        systolic_wave = ppg[f_left:pk+1]
        sw_feats = {}
        for p in percentages:
            target_amp = f_left_val + p * A_syst
            idx_up = np.where(systolic_wave >= target_amp)[0]
            if idx_up.size >= 2:
                width = (idx_up[-1] - idx_up[0]) / fs
            else:
                width = 0.0
            sw_feats[p] = width

        # Diastolic Width at each percentage
        diastolic_wave = ppg[pk:f_right+1]
        dw_feats = {}
        for p in percentages:
            target_amp = peak_val - p * A_diast
            idx_down = np.where(diastolic_wave <= target_amp)[0]
            if idx_down.size >= 2:
                width = (idx_down[-1] - idx_down[0]) / fs
            else:
                width = 0.0
            dw_feats[p] = width

        cycle_dict = {
            "CardiacPeriod": T_cardiac,
            "SystolicUpstroke": T_systolic,
            "DiastolicTime": T_diastolic
        }
        for p in percentages:
            cycle_dict[f"SystolicWidth_{int(p*100)}"] = sw_feats[p]
            cycle_dict[f"DiastolicWidth_{int(p*100)}"] = dw_feats[p]

        all_cycle_feats.append(cycle_dict)

    if not all_cycle_feats:
        return None

    # Average the 15 time features across cycles
    time_keys = list(all_cycle_feats[0].keys())
    time_feats_mean = {k: np.mean([c[k] for c in all_cycle_feats]) for k in time_keys}

    # =========== 4. Compute frequency-domain features (LF, MF, HF) ============
    # Perform Welch PSD on the full signal
    freq_bands = {
        "LF": (0.04, 0.15),
        "MF": (0.15, 0.40),
        "HF": (0.40, 0.80)
    }
    f_welch, pxx_welch = sps.welch(ppg, fs=fs, nperseg=min(1024, len(ppg)))

    freq_feats = {}
    for band, (f1, f2) in freq_bands.items():
        idx = np.where((f_welch >= f1) & (f_welch <= f2))[0]
        if idx.size == 0:
            amp, freq = 0.0, 0.0
        else:
            sub_pxx = pxx_welch[idx]
            sub_f = f_welch[idx]
            i_max = np.argmax(sub_pxx)
            amp = sub_pxx[i_max]
            freq = sub_f[i_max]
        freq_feats[f"{band}_Amplitude"] = amp
        freq_feats[f"{band}_Frequency"] = freq

    # =========== 5. Combine time- and frequency-domain features ============
    feats = {}
    feats.update(time_feats_mean)
    feats.update(freq_feats)
    return feats


import os
import numpy as np
from tqdm import tqdm

def aggregate_features_for_subject(subject, fs=200):
    """
    Aggregate extracted features for each subject across all segments.
    Each segment has 4 channels; we compile a matrix of shape (4, 21).
    """
    segments = subject["ppg_segments"]
    num_channels = 4
    seg_features_list = []

    # Determine feature key order from the first valid segment
    feature_keys = None
    for seg in segments:
        for ch in range(num_channels):
            feats = extract_features_detailed(seg[:, ch], fs)
            if feats is not None:
                feature_keys = sorted(feats.keys())
                break
        if feature_keys is not None:
            break
    if feature_keys is None:
        feature_keys = []

    # Extract features for each segment and channel
    for seg in segments:
        valid_segment = True
        channel_features = []
        for ch in range(num_channels):
            feats = extract_features_detailed(seg[:, ch], fs)
            if feats is None:
                valid_segment = False
                break
            feat_vector = np.array([feats[k] for k in feature_keys], dtype=np.float32)
            channel_features.append(feat_vector)
        if valid_segment:
            seg_features_list.append(np.stack(channel_features, axis=0))

    if seg_features_list:
        aggregated_array = np.stack(seg_features_list, axis=0)
    else:
        print(f"No valid segments for subject {subject['id']}")
        aggregated_array = None

    return {
        "id": subject["id"],
        "sbp": subject["sbp"],
        "dbp": subject["dbp"],
        "features": aggregated_array[0] if aggregated_array is not None else None
    }


if __name__ == "__main__":
    # Load bp_1.npy (list of subject dicts)
    bp1_path = "../dataset/bp_1.npy"
    if not os.path.exists(bp1_path):
        print(f"File {bp1_path} not found!")
        exit(1)

    bp1_data = list(np.load(bp1_path, allow_pickle=True))
    aggregated_subjects = []

    for subject in tqdm(bp1_data, desc="Extracting features from subjects"):
        agg = aggregate_features_for_subject(subject, fs=200)
        if agg["features"] is not None:
            aggregated_subjects.append(agg)

    # Save to bp_f.npy: each subject has 4×21 features plus id, sbp, dbp
    bp2_path = "../dataset/bp_f.npy"
    np.save(bp2_path, aggregated_subjects)
    print(f"Saved aggregated features for {len(aggregated_subjects)} subjects to {bp2_path}")
