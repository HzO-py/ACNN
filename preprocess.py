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




def process_subject(i,abnormal_subjects):

    # Define directories
    data_dir = "../dataset/ppg_data"  # Path to PPG data
    excel_path = "../dataset/Subjects Information.xlsx"  # Path to subjects' info file

    # Load the subjects information
    subjects_info = pd.read_excel(excel_path)

    # Define sampling frequency and filter parameters
    fs = 200  # Sampling frequency (Hz)
    lowcut = 0.5
    highcut = 5
    order = 2  # Second-order Butterworth filter

    # Define wavelet transform parameters
    wavelet = 'cgau1'  # Complex Gaussian wavelet
    scales = np.arange(1, 128)  # Scale range

    # Define windowing parameters
    window_length = 5 * fs  # 5 seconds * 200 Hz = 1000 samples per window
    step_size = 1 * fs  # 1-second step = 200 samples
    
    # Walk through the dataset directory
    mat_file = f"{i}.mat"
    mat_path = os.path.join(data_dir, mat_file)

    if not os.path.exists(mat_path):
        print(f"File not found: {mat_path}")
        return None  # Skip missing files

    # Load .mat file
    mat_contents = scipy.io.loadmat(mat_path)

    # Extract the PPG data structure
    try:
        ppg_data_structure = mat_contents['PPGdata']
        ppg_data_extracted = ppg_data_structure[0, 0]
        ppg_signal_data = np.array(ppg_data_extracted['data'])

        # 如果只是一维或单通道数据，跳过
        if ppg_signal_data.ndim != 2 or ppg_signal_data.shape[1] != 4:
            print(f"Subject {i}: PPGdata does not contain 4 channels.")
            return None
    except:
        try:
            ppg_data_structure = mat_contents['data']
            ppg_data_extracted = ppg_data_structure[0, 0]

            # 尝试提取4个通道
            channels = []
            for ch in ["channel_1", "channel_2", "channel_3", "channel_4"]:
                if ch in ppg_data_extracted.dtype.names:
                    channels.append(np.array(ppg_data_extracted[ch]).flatten())
                else:
                    print(f"Subject {i}: Missing {ch}")
                    return None

            # Stack into (N, 4)
            ppg_signal_data = np.column_stack(channels)

        except Exception as e:
            print(f"Subject {i}: Error loading channels - {e}")
            return None

    # Design a Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply the filter to each channel
    filtered_signals = np.array([signal.filtfilt(b, a, ppg_signal_data[:, j]) for j in range(4)]).T

    # for j in range(4):
    #     filtered_signals[:, j] = (filtered_signals[:, j] - np.min(filtered_signals[:, j])) / (np.max(filtered_signals[:, j]) - np.min(filtered_signals[:, j]))


    # Z-score normalization for each channel
    normalized_signals = zscore(filtered_signals, axis=0)

    # 定义异常点阈值
    threshold = 3

    # 检查每个通道是否有异常点
    abnormal_channel_count = 0
    for j in range(4):
        z = normalized_signals[:, j]
        if np.any(np.abs(z) > threshold):
            abnormal_channel_count += 1

    if 0 and abnormal_channel_count > 0:
        abnormal_subjects.append((i, abnormal_channel_count))


        # Define colors matching the wavelength diagram
        channel_colors = ["red", "mediumseagreen", "peru", "gold"]

        # Plot
        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(24, 10))  # 横轴拉长两倍
        fig.suptitle(f"Raw vs Filtered PPG Signals (Z-score) for Subject {i}", fontsize=18)

        channel_names = ["Channel 1 (660nm)", "Channel 2 (730nm)", "Channel 3 (850nm)", "Channel 4 (940nm)"]

        for j in range(4):
            color = channel_colors[j]

            # Plot raw signal
            axes[j, 0].plot(ppg_signal_data[:, j], color='gray')
            axes[j, 0].set_title(f"Raw {channel_names[j]}")
            axes[j, 0].set_ylabel("Amplitude")
            axes[j, 0].set_xlabel("Sample")

            # Plot filtered + z-score signal
            axes[j, 1].plot(normalized_signals[:, j], color=color)
            axes[j, 1].set_title(f"Filtered (Z-score) {channel_names[j]}")
            axes[j, 1].set_ylabel("Z-score")
            axes[j, 1].set_xlabel("Sample")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    # # Number of windows
    # num_windows = (len(filtered_signals) - window_length) // step_size + 1

    # # Initialize storage for 12-channel stacked images
    # stacked_rgb_images = np.zeros((num_windows, len(scales), window_length, 12))

    # # Compute CWT and generate RGB images for each wavelength
    # for ch in range(4):  # 4 wavelengths
    #     for w in range(num_windows):
    #         start = w * step_size
    #         end = start + window_length
    #         segment = filtered_signals[start:end, ch]  # Extract 5s segment from channel ch

    #         waveform_save_dir = "../dataset/waveforms"
    #         os.makedirs(waveform_save_dir, exist_ok=True)

    #         # ✅ Save waveform plot as an image
    #         plt.figure(figsize=(8, 4))
    #         plt.plot(segment, color='blue', linewidth=1.5)  # Plot waveform
    #         plt.title(f"PPG Waveform - Subject {i} - Channel {ch+1} - Window {w}")
    #         plt.xlabel("Time (samples)")
    #         plt.ylabel("Amplitude")
    #         plt.grid(True)

    #         # ✅ Define save path
    #         waveform_path = os.path.join(waveform_save_dir, f"subject_{i}_channel_{ch+1}_window_{w}.png")

    #         plt.savefig(waveform_path, dpi=300)  # Save image with high resolution
    #         plt.close()

    #         # Perform CWT
    #         coefficients, _ = pywt.cwt(segment, scales, wavelet, 1/fs)

    #         # Normalize scalogram for image conversion
    #         scalogram = np.abs(coefficients)
    #         scalogram = cv2.normalize(scalogram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


    #         # Convert grayscale scalogram to RGB
    #         rgb_image = cv2.applyColorMap(scalogram, cv2.COLORMAP_JET)  # Use Jet colormap as RGB conversion
            
    #         # Store in corresponding depth channels (3-channel per wavelength)
    #         stacked_rgb_images[w, :, :, ch * 3: (ch + 1) * 3] = rgb_image  # Stacking RGB images

    # subject_file=os.path.join('../dataset/subject', f"{i}.npy")
    # np.save(subject_file, stacked_rgb_images)

    # Extract ground truth SBP and DBP values
    sbp_value = subjects_info.loc[subjects_info['ID'] == i, 'SBP(mmHg)'].values
    dbp_value = subjects_info.loc[subjects_info['ID'] == i, 'DBP(mmHg)'].values

    if len(sbp_value) == 0 or len(dbp_value) == 0:
        print(f"Missing SBP/DBP for subject ID {i}")
        return None  # Skip if no matching ground truth

    sbp_value = sbp_value[0]
    dbp_value = dbp_value[0]

    sbp_mean = subjects_info['SBP(mmHg)'].mean()
    sbp_std = subjects_info['SBP(mmHg)'].std()
    dbp_mean = subjects_info['DBP(mmHg)'].mean()
    dbp_std = subjects_info['DBP(mmHg)'].std()
    if not (sbp_mean - 3 * sbp_std <= sbp_value <= sbp_mean + 3 * sbp_std and
            dbp_mean - 3 * dbp_std <= dbp_value <= dbp_mean + 3 * dbp_std):
        print(f"Subject {i}: SBP/DBP out of ±3σ range, skipped.")
        return None
    
    return {
        "id": i,
        # "ppg": subject_file,
        "sbp": sbp_value,
        "dbp": dbp_value
    }


if __name__ == "__main__":
    # Set up multiprocessing pool
    # num_workers = min(cpu_count(), 8)  # Use at most 8 cores to avoid system overload
    # print(num_workers)
    
    # dataset = []

    # with Pool(num_workers) as pool:
    #     for res in tqdm(pool.imap_unordered(process_subject, range(1, 181)), total=180, desc="Processing Subjects"):
    #         if res is not None:
    #             dataset.append(res)

    # np.save("../dataset/bp.npy", dataset)

    # print(f"Total processed subjects: {len(dataset)}")
    for i in tqdm(range(1,181)):
        process_subject(i)




