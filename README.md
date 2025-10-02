# PrognosAI — AI-Driven Predictive Maintenance

Predictive maintenance system using NASA CMAPSS dataset to estimate Remaining Useful Life (RUL) of turbofan engines.

**Current status:** Milestone 1 (Data Preparation & Feature Engineering) completed.

---

## 📌 Project Overview

PrognosAI is a step-by-step project aimed at building a machine learning pipeline for predictive maintenance.

The system will predict **Remaining Useful Life (RUL)** of equipment and enable maintenance alerts before failures occur.

Currently, **Milestone 1** has been completed, which focuses on data preparation, cleaning, and feature engineering.

---

## ✅ Completed: Milestone 1 (Data Preparation & Feature Engineering)

Key steps completed in this milestone:

- **Data ingestion:** Loaded NASA CMAPSS `train_FD001.txt` with correct headers.
- **Integrity check:** Verified no missing values or corrupt entries.
- **RUL calculation:** Remaining Useful Life computed for each engine cycle.
- **Sensor filtering:** Dropped non-informative sensors (zero variance).
- **Sequence generation:** Created overlapping rolling windows (seq_length=50) for LSTM-ready time series.
- **Prepared datasets:** Produced training samples `X_train` and labels `y_train`.

### 📂 Outputs are documented in:

- `Milestone 1 Completion Report_ Data Preparation & Feature Engineering.pdf`
- `Milestone 1.ipynb` (code + visualizations)

---

## 🚧 Future Work (Upcoming Milestones)

The next phases are planned but not yet implemented:

- **Milestone 2** — Model development & training (LSTM/GRU, TensorFlow/Keras)
- **Milestone 3** — Evaluation (RMSE, predicted vs actual plots)
- **Milestone 4** — Risk thresholding & alert logic (warning/critical states)
- **Milestone 5** — Dashboard & visualization (Streamlit/Flask interface)

---

## 🚀 Getting Started

1. **Clone repository:**
   ```bash
   git clone https://github.com/<your-org>/prognosai.git
   cd prognosai
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Milestone 1 preprocessing (via notebook):**
   ```bash
   jupyter notebook notebooks/Milestone\ 1.ipynb
   ```
