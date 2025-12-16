# HARL_MOS2025
# Multi-Task Deep Learning with Over-Sampling and Style Randomization for Improved Cross-Regional Bird Vocalization Recognition
# Overview
Cross-regional bird vocalization recognition (BVR) poses significant challenges due to spectral overlap, class imbalance, and domain shifts caused by ecological and acoustic variability. This paper proposes HARL-MOS, a multi-task deep learning framework that integrates class-balanced over-sampling and style randomization into a dual-branch ResNet50 architecture to improve generalization across diverse regions. HARL-MOS extends a prior auditory representation model by introducing a vocalization type auxiliary classification task to promote style-invariant representations, while over-sampling addresses the long-tail distribution of rare species. A style randomization module perturbs acoustic feature statistics during training to enhance robustness to domain-specific variations. HARL-MOS was evaluated on the DB3V dataset spanning three ecologically distinct regions in the contiguous United States under six cross-region train–test protocols, and on a separate two-site soundscape dataset with overlapping species. Experimental results demonstrated consistent improvements over a standard baseline and the prior HARL framework; in the most challenging D2D1 setting, HARL-MOS improved the macro F1-score by 4.33 percentage points over HARL and by 27.80 points over the baseline. In fine-grained cross-site BVR, HARL-MOS maintained stable improved performance with ACC reaching up to 90.36% and 91.71% for S1S2 and S2S1, indicating reduced sensitivity to domain shift. These results demonstrate that HARL-MOS is a reliable framework for automated BVR, offering practical benefits for biodiversity monitoring and conservation efforts.

![HARL_MOS Blockdiagram](https://github.com/xingfengli/HARL_MOS2025/blob/main/HARL_MOS_models/HARL_vs_HARL_MOS.png)


# Datasets
The DB3V ans S1-S2 datasets were used for all the experiments in this work. 
1. Jing, X., Zhang, L., Xie, J., Gebhard, A., Baird, A., & Schuller, B. (2024). DB3V: A Dialect Dominated Dataset of Bird Vocalisation for Cross-corpus Bird Species Recognition, INTERSPEECH 2024, pp. 127-131, Kos, Greece. 
Zenodo. https://doi.org/10.5281/zenodo.11544734

### Summary of the DB3V dataset

| Species (Common name) | Code | D1 (Western Cordillera) | D2 (Interior Plains) | D3 (Eastern Highlands) | Sound Type | Freq. (kHz) |
|---|---:|---:|---:|---:|---|---|
| Agelaius phoeniceus (Red-winged Blackbird) | 0 | 1,295 | 54  | 839  | Song | 2.8–5.7 |
| Cardinalis cardinalis (Northern Cardinal) | 1 | 778   | 166 | 1,299 | Song | 3.5–4.0 |
| Certhia americana (Brown Creeper) | 2 | 345   | 12  | 132  | Call | 3.7–8.0 |
| Corvus brachyrhynchos (American Crow) | 3 | 645   | 123 | 435  | Call | 0.5–1.8 |
| Molothrus ater (Brown-headed Cowbird) | 4 | 392   | 50  | 96   | Call | 0.5–12.0 |
| Setophaga aestiva (American Yellow Warbler) | 5 | 730   | 9   | 297  | Song | 3.0–8.0 |
| Setophaga ruticilla (American Redstart) | 6 | 199   | 107 | 579  | Song | 3.0–8.0 |
| Spinus tristis (American Goldfinch) | 7 | 223   | 94  | 283  | Song | 1.6–6.7 |
| Tringa semipalmata (Willet) | 8 | 138   | 29  | 106  | Call | 1.5–2.5 |
| Turdus migratorius (American Robin) | 9 | 1,038 | 187 | 791  | Song | 1.8–3.7 |

3. Morgan MM, Braasch J. Open set classification strategies for long-term environmental field recordings for bird species recognition. The Journal of the Acoustical Society of America. 2022 Jun 1;151(6):4028-38.
Zenodo. https://zenodo.org/records/6456604

### Summary of the S1 (Albany) and S2 (Lake George) dataset

| Species (Common name) | Code | S1 | S2 | Sound Type |
|---|---|---:|---:|---|
| Eastern chipmunk “chuck” (*Tamias striatus*) | ECMK | 537 | 1,161 | Call |
| Fall field cricket (*Gryllus pennsylvanicus*) | FFCR | 5,296 | 6,101 | Song |
| Eastern chipmunk “chirp” (*Tamias striatus*) | ECMC | 101 | 2,305 | Call |
| American robin (*Turdus migratorius*) | AMRO | 1,411 | 5,554 | Song |
| American crow (*Corvus brachyrhynchos*) | AMCR | 221 | 1,477 | Call |
| Blue jay (*Cyanocitta cristata*) | BLJA | 770 | 1,002 | Call |

# Instructions to Run Codes in Features (MATLAB 2024b)

## Setup and Visualization Instructions

To set up and visualize Mel and Gamma spectrograms, follow these steps:

1. **Set up the environment**:
   - Run `addpath('your_own_specified_path/features/gammatonegram')` in MATLAB 2024b to include the required files.
   - Replace `your_own_specified_path` with the actual path to the `features/gammatonegram` folder on your device.

2. **Visualize spectrograms**:
   - Run `visualization_demo.m` to generate and display Mel and Gamma spectrograms along with their delta variants.

3. **Extract and store spectrograms**:
   - Run `listFilesAndFolders.m` to extract, process, and store Mel and Gamma spectrograms.

4. **Access pre-extracted features**:
   - Download pre-extracted spectrogram features in `.mat` format from: [Zipped Features (OneDrive)](https://1drv.ms/f/c/f1ce98298ad945ca/EspF2YopmM4ggPHrcAUAAAAB7gAUF3LLA8F9aL1zETtmFQ?e=r1FLv7).
   - Unzip the files to `your_own_specified_path/features/` before running `listFilesAndFolders.m`.

**Resources**:
- [MATLAB 2024b Documentation](https://www.mathworks.com/help/releases/R2024b/matlab/) for environment setup and scripting.
- [Download MATLAB 2024b](https://www.mathworks.com/products/matlab.html) to install the required software.
- [GitHub Markdown Guide](https://docs.github.com/en/get-started/writing-on-github) for formatting this README.

**Reminder**:
- Ensure all folder paths (e.g., `your_own_specified_path`) are updated to match your local setup.
- Verify that the OneDrive link is accessible; contact the repository owner if access is restricted.

# Instructions to Run Python Model Training and Visualization (Python 3.8+, 4060 Ti, 64GB)

1. **Set up the environment**:
   - numpy==1.24.4
   - scipy==1.10.1
   - torch==2.0.1
   - torchaudio==2.0.2
   - torchvision==0.15.2
   - scikit-learn==1.3.2
   - matplotlib==3.7.2
   - umap-learn==0.5.3.
   - Update folder paths in all scripts to match your local setup.

2. **Train Mel spectrogram models**:
   - Run `D1D2_mel_wo_atten.py` for training with D1 and testing with D2, without attention.
   - Run `D1D2_mel_wi_atten.py` for training with D1 and testing with D2, with attention.

3. **Train Gammatone spectrogram models**:
   - Run `D1D2_gamma_wo_atten.py` for training with D1 and testing with D2, without attention.
   - Run `D1D2_gamma_wi_atten.py` for training with D1 and testing with D2, with attention.

4. **Train combined Mel and Gammatone models**:
   - Run `D1D2_mel_plus_gamma_wo_atten.py` for combined training with D1 and testing with D2, without attention.
   - Run `D1D2_mel_plus_gamma_wi_atten.py` for combined training with D1 and testing with D2, with attention.

5. **Visualize results**:
   - Run `ROC.py` and `UMAPs.py` to generate ROC curves and UMAP visualizations.

**Resources**:

- [Python 3 Documentation](https://docs.python.org/3/) for Python environment setup.

**Reminder**:

- Update all folder paths (e.g., `your_own_specified_path`) to match your local setup for MATLAB and Python.
- For Python scripts, adjust paths for other `DmDn` cases (e.g., D1D3) as needed.
- Install Python dependencies (e.g., `pip install numpy matplotlib tensorflow umap-learn scipy`).

