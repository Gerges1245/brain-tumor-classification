# 🧠 Brain Tumor MRI Classification
### Measuring the Impact of Image Preprocessing & Denoising on CNN Performance

A deep learning project that trains an identical CNN architecture on three versions of the same MRI dataset — **Raw**, **Preprocessed**, and **Cleaned** — to quantify how much classical image processing improves a model's ability to detect brain tumors.

---

## 📌 Overview

Brain tumor diagnosis from MRI scans is a high-stakes classification task where image noise, contrast, and artifacts can meaningfully affect model accuracy. This project investigates that effect directly by building a **three-stage image pipeline from scratch** (no black-box library calls for the core transforms) and training the same CNN on each stage's output, under identical conditions.

| Stage | What happens |
|---|---|
| **1. Raw** | Grayscale conversion + simple resize, no further processing |
| **2. Preprocessed** | Custom aspect-ratio-preserving resize, implemented manually with nearest-neighbor mapping |
| **3. Cleaned** | Preprocessed + manual **bilateral filter** (edge-preserving denoising) + **local contrast enhancement** + **unsharp masking** |

## 🏆 Key Results

| Pipeline | Test Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Raw | 0.7401 | 0.7331 | 0.7401 | 0.7222 | 0.9117 |
| Preprocessed | 0.7053 | 0.6998 | 0.7053 | 0.6645 | 0.8721 |
| **Cleaned** | **0.8028** | **0.8008** | **0.8028** | **0.7965** | **0.9450** |

The **Cleaned** pipeline outperformed the raw baseline by **+6.3 accuracy points** and **+0.033 ROC-AUC**, showing that targeted denoising and contrast enhancement help the CNN focus on tumor structures rather than scanner noise. Interestingly, resizing alone (without denoising) slightly *underperformed* the raw baseline — a reminder that not all preprocessing is automatically helpful.

## 🗂️ Dataset

[Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) on Kaggle — 4 classes:

- `glioma_tumor`
- `meningioma_tumor`
- `no_tumor`
- `pituitary_tumor`

The dataset is downloaded automatically at runtime via [`kagglehub`](https://pypi.org/project/kagglehub/); no manual download is required.

## 🧱 Model Architecture

A sequential CNN built with `tf.keras`:

- 4 convolutional blocks (32 → 64 → 128 → 256 filters), each with Batch Normalization + Max Pooling
- Dense classification head
- Trained with Adam, sparse categorical cross-entropy, class weighting for imbalance
- Early stopping + `ReduceLROnPlateau` callbacks
- Identical architecture, optimizer, and callbacks used for all three pipeline stages, so the only variable is image quality

## 🔬 Methodology

1. **EDA** — class distribution, sample images per class, and pixel-intensity statistics on the raw dataset.
2. **Preprocessing** — a hand-written resize + grayscale routine (no `cv2.resize`), preserving aspect ratio.
3. **Cleaning** — a from-scratch NumPy implementation of:
   - Bilateral filtering (spatial + range Gaussian weighting) for edge-preserving noise reduction
   - Local contrast enhancement
   - Unsharp masking for tumor-boundary sharpening
4. **Quality check** — Laplacian-variance sharpness metrics across all three stages to validate that cleaning actually improves image quality.
5. **Dataset construction** — stratified 70/15/15 train/validation/test split (same split reused across all three variants), plus a manual data-augmentation routine (random flips and rotations).
6. **Training & evaluation** — same CNN trained from scratch on each of the three datasets; evaluated with accuracy, precision, recall, F1, and ROC-AUC, plus confusion matrices and per-class failure analysis.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `OpenCV` · `NumPy` · `scikit-learn` · `scikit-image` · `Matplotlib` · `Seaborn` · `pandas` · `kagglehub`

## 📁 Project Structure

```
.
├── brain_tumor_classification.ipynb   # Main notebook: EDA → preprocessing → cleaning → training → evaluation
└── README.md
```

## 🚀 Getting Started

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn scikit-image pandas kagglehub
```

Open `brain_tumor_classification.ipynb` in Jupyter or Google Colab and run all cells. The dataset downloads automatically on the first run via `kagglehub`.

## 📊 What's Inside the Notebook

The notebook is organized into 12 documented sections covering setup, dataset loading, EDA, the preprocessing pipeline, the image-cleaning pipeline, image-quality metrics, dataset construction, the CNN architecture, training/evaluation for all three stages, the results comparison, final visualizations (loss curves, confusion matrices, failure analysis, pixel histograms, accuracy curves), and a conclusion.

## 📝 Notes

- All preprocessing and denoising steps are implemented manually in NumPy (not via library shortcuts), to make the image-processing pipeline transparent and easy to inspect or modify.
- Random seeds are fixed (`SEED = 42`) for reproducibility.
- This project is for educational and research purposes and is **not** a diagnostic tool.

## 📄 License

This project is open-sourced for educational purposes. Please credit the [original Kaggle dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri) if you reuse it.
