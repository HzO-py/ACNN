import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as signal
import pywt
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import zscore
from tqdm import tqdm
import neurokit2 as nk


def printplot(subject_id, raw_signals, norm_signals):
    """
    Plot raw vs filtered PPG signals (Z-score) and quality metric for each channel.
    """
    channel_colors = ["red", "mediumseagreen", "peru", "gold"]
    channel_names = ["Channel 1 (660nm)", "Channel 2 (730nm)",
                     "Channel 3 (850nm)", "Channel 4 (940nm)"]

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(24, 10))
    fig.suptitle(f"Raw vs Filtered PPG Signals (Z-score) for Subject {subject_id}", fontsize=18)

    for ch in range(4):
        color = channel_colors[ch]

        # Compute PPG quality
        quality_signal = nk.ppg_quality(norm_signals[:, ch], sampling_rate=25)
        # Quality is in [0,1]; plot as-is (or scale if needed)

        # Plot raw signal
        axes[ch, 0].plot(raw_signals[:, ch], color='gray')
        axes[ch, 0].set_title(f"Raw {channel_names[ch]}")
        axes[ch, 0].set_ylabel("Amplitude")
        axes[ch, 0].set_xlabel("Sample")

        # Plot filtered (Z-scored) signal and quality
        axes[ch, 1].plot(norm_signals[:, ch], color=color, label="Z-scored Signal")
        axes[ch, 1].plot(quality_signal, color="black", linestyle="--", alpha=0.6, label="PPG Quality")
        axes[ch, 1].set_title(f"Filtered (Z-score) {channel_names[ch]}")
        axes[ch, 1].set_ylim(-3, 3)
        axes[ch, 1].set_ylabel("Z-score")
        axes[ch, 1].set_xlabel("Sample")
        axes[ch, 1].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def process_subject(subject_id, subjects_info):
    """
    Load and process PPG data for one subject:
    - Check SBP/DBP outliers
    - Load .mat, extract 4-channel PPG
    - Bandpass filter and Z-score
    - Downsample and extract central 30s
    - Skip if invalid
    """
    data_dir = "../dataset/ppg_data"
    fs = 200  # Sampling rate (Hz)
    lowcut = 0.5
    highcut = 5.0
    order = 2  # Butterworth filter order
    threshold = 4.0  # Z-score threshold for outlier

    # 1. Ground truth SBP/DBP and outlier check (±3σ)
    sbp_vals = subjects_info.loc[subjects_info['ID'] == subject_id, 'SBP(mmHg)'].values
    dbp_vals = subjects_info.loc[subjects_info['ID'] == subject_id, 'DBP(mmHg)'].values
    if sbp_vals.size == 0 or dbp_vals.size == 0:
        print(f"Subject {subject_id}: Missing SBP/DBP.")
        return None
    sbp = sbp_vals[0]
    dbp = dbp_vals[0]

    sbp_mean = subjects_info['SBP(mmHg)'].mean()
    sbp_std = subjects_info['SBP(mmHg)'].std()
    dbp_mean = subjects_info['DBP(mmHg)'].mean()
    dbp_std = subjects_info['DBP(mmHg)'].std()

    if not (sbp_mean - 3*sbp_std <= sbp <= sbp_mean + 3*sbp_std and
            dbp_mean - 3*dbp_std <= dbp <= dbp_mean + 3*dbp_std):
        print(f"Subject {subject_id}: SBP/DBP outside ±3σ range.")
        return None

    # 2. Load .mat and extract PPG data
    mat_path = os.path.join(data_dir, f"{subject_id}.mat")
    if not os.path.exists(mat_path):
        print(f"Subject {subject_id}: .mat file not found.")
        return None
    mat = scipy.io.loadmat(mat_path)
    try:
        arr = mat['PPGdata'][0,0]
        ppg = np.array(arr['data'])
        if ppg.ndim != 2 or ppg.shape[1] != 4:
            print(f"Subject {subject_id}: PPGdata not shape (N,4).")
            return None
    except Exception:
        try:
            arr = mat['data'][0,0]
            channels = []
            for ch_name in ['channel_1','channel_2','channel_3','channel_4']:
                if ch_name in arr.dtype.names:
                    channels.append(arr[ch_name].flatten())
                else:
                    print(f"Subject {subject_id}: Missing {ch_name}.")
                    return None
            ppg = np.column_stack(channels)
        except Exception as e:
            print(f"Subject {subject_id}: Error extracting channels: {e}")
            return None

    # 3. Bandpass filter and Z-score
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    filtered = np.vstack([signal.filtfilt(b, a, ppg[:,ch]) for ch in range(4)]).T
    normalized = zscore(filtered, axis=0)

    # 4. Downsample to 25 Hz and take central 30s
    target_fs = 25
    ds_factor = fs // target_fs  # 200/25 = 8
    raw_ds = ppg[::ds_factor, :]
    norm_ds = normalized[::ds_factor, :]
    window_len = target_fs * 30  # 25*30=750 samples
    if norm_ds.shape[0] < window_len:
        print(f"Subject {subject_id}: Data shorter than 30s, skipping.")
        return None
    start = (norm_ds.shape[0] - window_len) // 2
    end = start + window_len
    win_norm = norm_ds[start:end, :]
    win_raw = raw_ds[start:end, :]

    # Skip if any |Z|>threshold or raw==0
    if np.any(np.abs(win_norm) > threshold) or np.any(win_raw == 0):
        print(f"Subject {subject_id}: Central 30s has outliers or zeros, skipping.")
        return None

    printplot(subject_id, win_raw, win_norm)

    return {
        'id': subject_id,
        'sbp': sbp,
        'dbp': dbp,
        'ppg_segments': [win_norm]
    }


if __name__ == "__main__":
    # Load subject info from Excel
    info_path = "../dataset/Subjects Information.xlsx"
    subjects_info = pd.read_excel(info_path)

    dataset = []
    for subject_id in tqdm(range(1, 181), desc="Processing subjects"):
        result = process_subject(subject_id, subjects_info)
        if result is not None:
            dataset.append(result)

    np.save("../dataset/bp_1.npy", dataset)
    print(f"\n✅ Total valid subjects: {len(dataset)}")
