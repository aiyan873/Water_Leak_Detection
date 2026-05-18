# Pipe Fault Detection

Trains and compares five ML models on vibration sensor data to classify pipe faults — from classical baselines to a deep learning network.

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
