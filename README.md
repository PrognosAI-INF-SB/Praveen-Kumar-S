# PrognosAI: Turbofan Engine RUL Prediction & Predictive Maintenance Dashboard

A complete end-to-end predictive maintenance system built using NASA's CMAPSS turbofan engine dataset. The project includes data preprocessing, deep learning model training, RUL prediction, visualization, explainability, and a fully functional Flask-based dashboard.

---

## 🚀 Overview

PrognosAI predicts the **Remaining Useful Life (RUL)** of turbofan engines using sensor data. It provides intuitive dashboards, visual analytics, and downloadable reports for actionable maintenance decisions.

This repository includes:

* Data preprocessing scripts
* Sensor analysis & visualizations
* LSTM/BiLSTM based RUL prediction model
* Unified dataset handling for FD001–FD004
* Flask web dashboard with Plotly graphs
* Automated PDF report generation
* Complete project documentation (Milestones 1–5)

---

## 🧠 Key Features

### 1. **Advanced Deep Learning Model**

* Bidirectional LSTM architecture
* Sequence learning for temporal degradation
* Combined dataset training (FD001–FD004)
* Feature scaling with saved scalers

### 2. **Interactive Dashboard**

* Upload engine data and get instant predictions
* Interactive charts using Plotly:

  * Sensor trends
  * Degradation curves
  * RUL gauge
  * Health indicator
  * Correlation heatmap
* Downloadable PDF RUL reports

### 3. **Explainability Support**

* Sensor contribution visualization
* Degradation pattern interpretation
* Automatic identification of stable vs degrading sensors

### 4. **Robust Inference Pipeline**

* Automatic feature alignment
* Scaler fallback mechanism
* Sequence creation on-the-fly

---

## 📊 Visualizations Included

* Sensor Trend Line Graphs
* Cycle-wise Degradation
* Correlation Heatmaps
* Real-time RUL Gauge
* Health Metric Widget

These visualizations make diagnostics and maintenance planning intuitive and effective.

---

## 🛠 Tech Stack

* **Python 3.10+**
* **TensorFlow / Keras** (BiLSTM Model)
* **Flask** (Web backend)
* **Plotly** (Interactive charts)
* **Matplotlib & Seaborn** (Heatmaps)
* **Pandas & NumPy** (Data processing)
* **Joblib / Pickle** (Scaler + feature encoding persistence)

---

## ▶️ Running the Dashboard

### 1. Clone the repository

```
git clone https://github.com/yourusername/PrognosAI.git
cd PrognosAI/dashboard
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Run the Flask server

```
python app.py
```

### 4. Open in browser

```
http://127.0.0.1:5000/
```

Upload any engine CSV and view:

* RUL prediction
* Sensor trends
* Heatmap
* Degradation plots
* Summary metrics

---

## 📄 PDF Reports

The application generates a downloadable PDF containing:

* Model prediction summary
* RUL value
* Health classification
* Plots and trends
* Dataset metadata

---

## 🧪 Dataset Used

NASA's **CMAPSS Turbofan Engine Degradation Dataset**, containing:

* 4 subsets (FD001–FD004)
* Multi-sensor time-series engine data
* Multiple operating conditions and fault modes

RUL computation is based on cycle difference between failure point and observation.

---

## 📈 Model Performance

The combined dataset model achieves stable performance across all four datasets, with:

* Low MAE
* Smooth training convergence
* Generalized learning for varying conditions

Visual diagnostics confirm the model captures degradation patterns effectively.

---

## 📘 Documentation

Detailed milestone reports:

* Milestone 1: Dataset understanding and EDA
* Milestone 2: Model building and training
* Milestone 3: Dashboard and deployment
* Milestone 4: Report automation and explainability
* Milestone 5: Final evaluation and packaging

These describe the full evolution of the system from research to deployment.

---

## 🤝 Contributing

Contributions are welcome. Feel free to open issues, suggest improvements, or create pull requests.

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for details.

---

## ⭐ Acknowledgements

* NASA CMAPSS dataset
* TensorFlow/Keras community
* Open-source contributors

---

## 🙌 Final Note

PrognosAI demonstrates how data-driven methods and deep learning can greatly enhance predictive maintenance, reduce operational downtime, and support effective decision-making in aviation and industrial systems.

Ready for real-world deployment and future expansion.
