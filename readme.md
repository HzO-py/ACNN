# Multi-Channel PPG → Blood Pressure Estimation

This repository implements several machine-learning methods to estimate systolic (SBP) and diastolic (DBP) blood pressure from **multi-wavelength PPG** signals.

---

## Contents

- [Overview](#overview)  
- [Reference Papers](#reference-papers)  
- [Datasets](#datasets)  
- [Repository Structure](#repository-structure)  

---

## Overview

- We use 4-wavelength PPG to predict SBP/DBP.  
- Replicate the methods from Reference Papers [1], [2] and [3] on the provided four-wavelength PPG dataset (Dataset [1])

---

If you’re interested in **single-channel** PPG→BP, see the Nature “BP Benchmark”:
- **Paper:** [A benchmark for machine-learning based non-invasive blood pressure estimation using photoplethysmogram](https://www.nature.com/articles/s41597-023-02020-6)  
- **Code:** https://github.com/inventec-ai-center/bp-benchmark

---

## Reference Papers

1. **ACNN-BiLSTM**  
   C. Lu et al., “A Deep Learning Approach for Continuous Noninvasive Blood Pressure Measurement Using Multi-Wavelength PPG Fusion,” _Biosensors_ 11(4), 306 (2023).  
   https://www.mdpi.com/2306-5354/11/4/306  

2. **AI-Based Multi-Wavelength PPG Device**  
   Y. Zhang et al., “AI-Based Multi-Wavelength PPG Device for Blood Pressure Monitoring,” _IEEE Trans. Biomed. Circuits Syst._ (2024).  
   https://ieeexplore.ieee.org/document/10596751  

3. **Attention-Based Multi-Channel PPG + Finger Pressure**  
   H. Wang et al., “Deep-learning-based blood pressure estimation using multi channel photoplethysmogram and finger pressure with attention mechanism,” _Sci. Rep._ 13, 12345 (2023).  
   https://www.nature.com/articles/s41598-023-36068-6  

---

## Datasets

- **Processed (4-wavelength PPG)**:  
  Download Google Drive →  
  https://drive.google.com/file/d/1lcpU7WPU17OjYmGxhIUphLqdYyh7tHtu/view?usp=sharing

- **Original**:  
  Four-wavelength_PPG_BP dataset →  
  https://figshare.com/articles/dataset/Blood_Pressure_Measurement_based_on_Four-wavelength_PPG_Signals/23283518/1?file=42344391

---

## Repository Structure

```text
Dataset/
│   ├── ppg_data│Subjects Information.xlsx    ← original dataset
│   ├── bp.npy                         ← 2D CWT images (for ACNN) [1]
│   ├── bp_1.npy                       ← preprocessed PPG (per-subject) for feature/derivative pipelines
│   ├── bp_f.npy                       ← hand-crafted features (for classical_ml) [2]
│   └── bp_d.npy                       ← PPG + 1st/2nd derivatives + envelopes (for CNN) [3]

src/
├── preprocess_ACNN.py                 ← build `bp.npy` for ACNN_BiLSTM.py [1]
├── ACNN_BiLSTM.py                     ← train/test ACNN-BiLSTM model [1]
├── model.py                           ← ACNN-BiLSTM architecture definition [1]
│
├── preprocess_f_or_d.py               ← build `bp_1.npy`
├── feature.py                         ← extract features → `bp_f.npy` [2]
├── derivative.py                      ← extract derivatives/envelopes → `bp_d.npy` [3]
│
├── split.py                           ← split (`bp_f.npy`|`bp_d.npy`) → train/val/test
│
├── classical_ml.py                    ← RandomForest, SVR, GBDT, AdaBoost on `bp_f.npy` [2]
│
├── multiCNN.py                        ← multiCNN architecture definition [3]
├── FourChannels_2CNN.py               ← train multiCNN on `bp_d.npy` [3]
└── FourChannels_2CNN_test.py          ← test pipeline for multiCNN [3]
