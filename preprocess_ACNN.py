import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as signal
import pywt
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

# Define directories
data_dir = "../dataset/ppg_data"  # Path to PPG data
excel_path = "../dataset/Subjects Information.xlsx"  # Path to subjects' info file

# Load the subjects information
subjects_info = pd.read_excel(excel_path)

# Define sampling frequency and filter parameters
fs = 200  # Sampling frequency (Hz)
lowcut = 0.5
highcut = 8
order = 2  # Second-order Butterworth filter

# Define wavelet transform parameters
wavelet = 'cgau1'  # Complex Gaussian wavelet
scales = np.arange(1, 128)  # Scale range

# Define windowing parameters
window_length = 5 * fs  # 5 seconds * 200 Hz = 1000 samples per window
step_size = 1 * fs  # 1-second step = 200 samples


def process_subject(i):
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
    except:
        ppg_data_structure = mat_contents['data']
        ppg_data_extracted = ppg_data_structure[0, 0]
        # Convert each channel into a NumPy array and stack them
        ppg_signal_data = np.column_stack([
            np.array(ppg_data_extracted["channel_1"]).flatten(),
            np.array(ppg_data_extracted["channel_2"]).flatten(),
            np.array(ppg_data_extracted["channel_3"]).flatten(),
            np.array(ppg_data_extracted["channel_4"]).flatten()
        ])  # Shape: (12000, 4)


    # Design a Butterworth bandpass filter
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')

    # Apply the filter to each channel
    filtered_signals = np.array([signal.filtfilt(b, a, ppg_signal_data[:, j]) for j in range(4)]).T

    for j in range(4):
        filtered_signals[:, j] = (filtered_signals[:, j] - np.min(filtered_signals[:, j])) / (np.max(filtered_signals[:, j]) - np.min(filtered_signals[:, j]))


    # Number of windows
    num_windows = (len(filtered_signals) - window_length) // step_size + 1

    # Initialize storage for 12-channel stacked images
    stacked_rgb_images = np.zeros((num_windows, len(scales), window_length, 12))

    # Compute CWT and generate RGB images for each wavelength
    for ch in range(4):  # 4 wavelengths
        for w in range(num_windows):
            start = w * step_size
            end = start + window_length
            segment = filtered_signals[start:end, ch]  # Extract 5s segment from channel ch

            # Perform CWT
            coefficients, _ = pywt.cwt(segment, scales, wavelet, 1/fs)

            # Normalize scalogram for image conversion
            scalogram = np.abs(coefficients)
            scalogram = cv2.normalize(scalogram, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


            # Convert grayscale scalogram to RGB
            rgb_image = cv2.applyColorMap(scalogram, cv2.COLORMAP_JET)  # Use Jet colormap as RGB conversion
            
            # Store in corresponding depth channels (3-channel per wavelength)
            stacked_rgb_images[w, :, :, ch * 3: (ch + 1) * 3] = rgb_image  # Stacking RGB images

    subject_file=os.path.join('../dataset/subject', f"{i}.npy")
    np.save(subject_file, stacked_rgb_images)

    # Extract ground truth SBP and DBP values
    sbp_value = subjects_info.loc[subjects_info['ID'] == i, 'SBP(mmHg)'].values
    dbp_value = subjects_info.loc[subjects_info['ID'] == i, 'DBP(mmHg)'].values

    if len(sbp_value) == 0 or len(dbp_value) == 0:
        print(f"Missing SBP/DBP for subject ID {i}")
        return None  # Skip if no matching ground truth

    sbp_value = sbp_value[0]
    dbp_value = dbp_value[0]

    return {
        "id": i,
        "ppg": subject_file,
        "sbp": sbp_value,
        "dbp": dbp_value
    }


if __name__ == "__main__":
    # Set up multiprocessing pool
    num_workers = min(cpu_count(), 8)  # Use at most 8 cores to avoid system overload
    print(num_workers)
    
    dataset = []

    with Pool(num_workers) as pool:
        for res in tqdm(pool.imap_unordered(process_subject, range(1, 181)), total=180, desc="Processing Subjects"):
            if res is not None:
                dataset.append(res)

    np.save("../dataset/bp.npy", dataset)

    print(f"Total processed subjects: {len(dataset)}")