# 🎵 Messy Mashup — Music Genre Classification
 
A multi-model machine learning project for classifying music genres from **noisy, multi-stem audio mashups**. Built for the Kaggle competition `jan-2026-dl-gen-ai-project`, this repo explores four different modelling approaches — from classical ML to deep CNNs — on a challenging 10-genre classification task.
 
---
 
## Problem Statement
 
Given audio mashups constructed by mixing stems (drums, vocals, bass, other) from songs of a single genre and layering ESC-50 environmental noise, predict the genre of each mashup. The task is a 10-class classification over:
 
> Blues · Classical · Country · Disco · Hip-Hop · Jazz · Metal · Pop · Reggae · Rock
 
The challenge lies in the deliberate "messiness": stems from different songs of the same genre are mixed together, and random ambient noise is overlaid at varying intensities.
 
---
 
## Dataset
 
| Component | Description |
|---|---|
| `genres_stems/` | Per-genre folders, each with song subfolders containing `drums.wav`, `vocals.wav`, `bass.wav`, `other.wav` |
| `ESC-50-master/audio/` | Environmental Sound Classification noise clips used for data augmentation |
| `test.csv` | Test set filenames and IDs |
| `sample_submission.csv` | Submission format template |
 
The dataset is sourced from the Kaggle competition input at `/kaggle/input/jan-2026-dl-gen-ai-project/messy_mashup/`.
 
---
 
## Data Pipeline
 
All models share a common **synthetic training data generation** strategy:
 
1. **Mashup Creation** — For each training sample, 4 random songs from the same genre are selected and their stems mixed together with random per-stem gain (`0.4–1.1×`).
2. **Noise Injection** — A random ESC-50 clip is added at low intensity (`5–20% weight`) to simulate real-world noise conditions.
3. **Feature Extraction / Spectrogram** — Depending on the model, either hand-crafted audio features or raw mel-spectrograms are derived from each 30-second (or 10-second for CNN) mashup.
---
 
## Models
 
### 1. `XGB_model.ipynb` — XGBoost Classifier
 
A gradient boosting model operating on hand-crafted audio features. Also benchmarks Logistic Regression and Random Forest as baselines.
 
- **Features:** 88-dimensional vector (MFCCs, Delta-MFCCs, spectral features, chroma, ZCR, RMS, flatness)
- **Architecture:** XGBoost with 600 estimators, learning rate 0.05, max depth 6, subsampling
- **Preprocessing:** `StandardScaler` + `LabelEncoder`
- **Baselines also included:** Logistic Regression, Random Forest (200 estimators)
### 2. `MLP_model.ipynb` — Sklearn MLP Classifier
 
A multi-layer perceptron using scikit-learn, trained on the same 88-dimensional feature set.
 
- **Architecture:** `(512 → 256 → 128)` with ReLU, Adam optimizer
- **Regularization:** L2 penalty (`alpha=0.05`), adaptive learning rate, early stopping
- **Training:** Up to 500 epochs with W&B logging per epoch
### 3. `Pytorch_NN.ipynb` — PyTorch Neural Network
 
A custom deep neural network implemented in PyTorch on the same feature set, with more control over training dynamics.
 
- **Architecture:** `GenreNet` — `Linear(512) → BN → ReLU → Dropout(0.5) → Linear(512) → BN → ReLU → Dropout(0.35) → Linear(256) → Linear(10)`
- **Loss:** Cross-entropy with label smoothing (`0.15`)
- **Optimizer:** AdamW with weight decay (`0.08`)
- **Scheduler:** `ReduceLROnPlateau` on validation accuracy
- **Training:** 200 epochs, batch size 64
### 4. `MEL_CNN_final_model.ipynb` — Mel-Spectrogram CNN ⭐ Final Model
 
The most sophisticated approach, treating genre classification as an image recognition problem on mel-spectrograms.
 
- **Input:** `(1, 128, T)` mel-spectrogram computed at 16kHz, 10-second clips
- **Architecture:** `MelCNN` — 4 convolutional blocks `(1→32→64→128→256)`, each with Conv2D + BN + ReLU + MaxPool + `ResBlock` (residual connections), followed by Global Average Pooling and a fully connected head
- **Augmentation:** SpecAugment (frequency & time masking), tempo stretching (`±12%`), random stem dropout (10%), 1–2 noise clips per sample
- **Loss:** Cross-entropy with label smoothing (`0.1`)
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- **Scheduler:** Cosine Annealing LR
- **Inference:** Test-Time Augmentation (TTA, 3 copies)
- **Data generation:** Parallelised with `joblib` (4 workers), 600 samples/genre
---
 
## Feature Engineering
 
For the XGB, MLP, and PyTorch models, each audio clip is represented as an **88-dimensional feature vector**:
 
| Feature Group | Features | Captures |
|---|---|---|
| MFCCs (20 coefficients) | Mean + Std × 20 = 40 | Timbre |
| Delta MFCCs (20 coefficients) | Mean + Std × 20 = 40 | Rhythm / transitions |
| Spectral Centroid | Mean | Brightness |
| Spectral Rolloff | Mean | High-frequency energy |
| Spectral Contrast | Mean | Distorted vs. clean instruments |
| Chroma STFT | Mean + Std | Harmonic / melodic content |
| Zero Crossing Rate | Mean | Percussiveness |
| RMS Energy | Mean | Loudness |
| Spectral Flatness | Mean | Tone vs. noise |
 
For the CNN, raw **mel-spectrograms** (`128 mel bins × T time frames`) are used directly as 2D image inputs.
 
---
 
## Project Structure
 
```
.
├── XGB_model.ipynb              # Logistic Regression / Random Forest / XGBoost
├── MLP_model.ipynb              # Sklearn MLP
├── Pytorch_NN.ipynb             # PyTorch feedforward network
├── MEL_CNN_final_model.ipynb    # Mel-spectrogram CNN (final submission model)
└── README.md
```
