# Pipe Fault Detection

> **Published Paper**
>
> S. M. Aiyan, K. Hossen, S. L. Rahman, M. F. A. Rezvi and R. I. Rafin, **"A Comparative Study of Classical Machine Learning Models and CNN-LSTM for Multiclass Time-Series Classification in Hose Pipe Fault Diagnosis,"** *2025 28th International Conference on Computer and Information Technology (ICCIT)*, Cox's Bazar, Bangladesh, 2025, pp. 1553–1558.
>
> 📄 **DOI:** [10.1109/ICCIT68739.2025.11491319](https://doi.org/10.1109/ICCIT68739.2025.11491319) · [IEEE Xplore](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11491319&isnumber=11490092)

---

Trains and compares five ML models on vibration sensor data to classify pipe faults — from classical baselines to a deep learning network.

### Abstract

Comparative analysis of classical and deep learning models for vibration-based multiclass fault detection in hose pipe systems. A dataset of **118,732 tri-axial accelerometer samples** was collected across **six fault states** (Normal, Twisted, Blocked, Loosed, Leaked, Burst), balanced to **114,354 samples** (~19,059 per class). Classical classifiers were benchmarked against a hybrid CNN-LSTM combining 1D convolution with temporal modeling. Robustness was tested with Gaussian noise (σ = 0.4).

### Key Results

| Model | Clean Accuracy | Noisy Accuracy (σ = 0.4) |
|---|---|---|
| CNN-LSTM | **99.67%** | 97.81% |
| Logistic Regression | >99% | **98.69%** *(most resilient)* |
| Random Forest | >99% | 90.05% *(sharpest drop)* |
| SVM | >99% | — |
| kNN | >99% | — |

> CNN-LSTM offers the best accuracy–robustness trade-off overall.

---

## Models

| Model | Type |
|---|---|
| Random Forest | Classical |
| SVM | Classical |
| Logistic Regression | Classical |
| kNN | Classical |
| CNN-LSTM | Deep Learning |

---

## Input

**`balanced_dataset.csv`** — X, Y, Z accelerometer readings with a `class` fault label per sample.

---

## How It Works

1. Segments data into fixed windows and normalizes
2. Stratified train/val/test split for balanced class distribution
3. Trains all five models
4. Evaluates each on **clean data** and **noise-injected data**
5. Saves results to `results/`

---

## Output

- **`vibration_model_results_readable.csv`** — accuracy, precision, recall, F1 per model
- **Confusion matrices** — one CSV per model, saved to `results/`

---

## Setup

Install dependencies:

```bash
pip install numpy pandas scikit-learn tensorflow
```

Place `balanced_dataset.csv` in the working directory, then run the script. Results write automatically.

---

## Note

Results vary across runs — noise injection is randomized and CNN-LSTM training is non-deterministic. The overall methodology stays consistent; individual numbers may shift.
