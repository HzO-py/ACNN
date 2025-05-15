import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import neurokit2 as nk
from tqdm import tqdm

# Bandpass filter
def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# First and second derivatives
def derivative(signal):
    return np.gradient(signal)

# Envelope extraction function
def extract_envelope(signal, fs):
    # Detect PPG peaks
    _, info = nk.ppg_peaks(signal, sampling_rate=fs)
    peaks = info['PPG_Peaks']
    if len(peaks) < 2:
        return False
    
    # Interpolate peak values to form envelope
    interp = interp1d(peaks, signal[peaks], kind='cubic', fill_value="extrapolate")
    envelope = interp(np.arange(len(signal)))

    # Flatten envelope before the first peak and after the last peak
    first_peak = peaks[0]
    last_peak  = peaks[-1]
    envelope[:first_peak] = envelope[first_peak]
    envelope[last_peak:] = envelope[last_peak]

    return envelope

def plot_features_per_channel(subject_id, features_per_channel):
    channel_colors = ["red", "mediumseagreen", "peru", "gold"]
    channel_names = ["Channel 1 (660nm)", "Channel 2 (730nm)", "Channel 3 (850nm)", "Channel 4 (940nm)"]

    feature_names = ["PPG + Envelope",
                     "First Derivative + Envelope",
                     "Second Derivative + Envelope"]

    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(30, 16))  # one row per channel, one column per feature
    fig.suptitle(f"Extracted Features for Subject {subject_id}", fontsize=20)

    for ch in range(4):
        color = channel_colors[ch]
        channel_features = features_per_channel[ch]

        # Column 0: PPG + envelope
        axes[ch, 0].plot(channel_features["ppg_filtered"], color=color, label="Filtered PPG")
        axes[ch, 0].plot(channel_features["ppg_envelope"], color="black", linestyle="--", alpha=0.7, label="Envelope")
        axes[ch, 0].set_ylim(-3, 3)

        # Column 1: First derivative + envelope
        axes[ch, 1].plot(channel_features["ppg_first_derivative"], color=color, label="1st Derivative")
        axes[ch, 1].plot(channel_features["ppg_first_envelope"], color="black", linestyle="--", alpha=0.7, label="Envelope")

        # Column 2: Second derivative + envelope
        axes[ch, 2].plot(channel_features["ppg_second_derivative"], color=color, label="2nd Derivative")
        axes[ch, 2].plot(channel_features["ppg_second_envelope"], color="black", linestyle="--", alpha=0.7, label="Envelope")

        for k in range(3):
            axes[ch, k].set_title(f"{feature_names[k]} - {channel_names[ch]}", fontsize=14)
            axes[ch, k].set_xlabel("Sample")
            axes[ch, k].set_ylabel("Amplitude")
            axes[ch, k].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def get_derivative():
    fs = 200  # Adjust according to your actual sampling rate
    data = np.load("../dataset/bp_1.npy", allow_pickle=True)
    results = []
    global_min_bp = float('inf')
    global_max_bp = float('-inf')

    for subject in tqdm(data):
        sbp = subject["sbp"]
        dbp = subject["dbp"]
        global_min_bp = min(global_min_bp, sbp, dbp)
        global_max_bp = max(global_max_bp, sbp, dbp)

        processed_segments = subject["ppg_segments"]
        subject_features = []
        
        for segment in processed_segments:
            features_per_channel = []

            # For each channel
            for ch in range(segment.shape[1]):
                ppg_filtered = segment[:, ch]

                # Envelope of PPG
                ppg_envelope = extract_envelope(ppg_filtered, fs)
                if ppg_envelope is False:
                    print(subject["id"])

                # First derivative and its envelope
                ppg_first_derivative = derivative(ppg_filtered)
                ppg_first_envelope = extract_envelope(ppg_first_derivative, fs)

                # Second derivative and its envelope
                ppg_second_derivative = derivative(ppg_first_derivative)
                ppg_second_envelope = extract_envelope(ppg_second_derivative, fs)

                downsample_factor = fs // 200  # e.g., 200 -> 25 Hz downsample

                raw_feats = {
                    "ppg_filtered":          ppg_filtered,
                    "ppg_envelope":          ppg_envelope,
                    "ppg_first_derivative":  ppg_first_derivative,
                    "ppg_first_envelope":    ppg_first_envelope,
                    "ppg_second_derivative": ppg_second_derivative,
                    "ppg_second_envelope":   ppg_second_envelope
                }

                # Downsample features
                ds_feats = {
                    name: feat[::downsample_factor]
                    for name, feat in raw_feats.items()
                }

                features_per_channel.append(ds_feats)
            
            subject_features.append(features_per_channel)
            # plot_features_per_channel(subject["id"], features_per_channel)
        
        results.append({
            "id": subject["id"],
            "sbp": sbp,
            "dbp": dbp,
            "features": subject_features
        })

    print(global_max_bp, global_min_bp)
    np.save("../dataset/bp_d.npy", results)
