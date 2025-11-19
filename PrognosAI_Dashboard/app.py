from flask import send_from_directory
import os
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, send_file, url_for
import numpy as np
import pandas as pd
import joblib
import pickle
from tensorflow.keras.models import load_model
import keras
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)
@app.route('/uploads/<path:filename>')
def uploaded_files(filename):
    return send_from_directory('uploads', filename)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Paths
MODEL_PATH = "models/PrognosAI_Combined_LSTM_GRU.h5"
FEATURE_PATH = "models/PrognosAI_feature_cols.pkl"
SCALER_PATHS = {
    "FD001": "models/scaler_FD001.pkl",
    "FD002": "models/scaler_FD002.pkl",
    "FD003": "models/scaler_FD003.pkl",
    "FD004": "models/scaler_FD004.pkl",
}

# Load model (handle legacy metrics)
model = load_model(
    MODEL_PATH,
    custom_objects={
        "mse": keras.losses.MeanSquaredError(),
        "mean_squared_error": keras.losses.MeanSquaredError(),
        "mae": keras.losses.MeanAbsoluteError(),
        "mean_absolute_error": keras.losses.MeanAbsoluteError()
    }
)

# Load feature_cols. If missing or wrong, we'll attempt to reconstruct later.
with open(FEATURE_PATH, "rb") as f:
    feature_cols = pickle.load(f)  # may be 25 or 29 entries

# Try to load scalers (joblib). Keep note whether scalers are usable.
scalers = {}
scaler_available = {}
for i, (k, p) in enumerate(SCALER_PATHS.items(), start=1):
    try:
        scalers[k] = joblib.load(p)
        scaler_available[k] = True
    except Exception as e:
        print(f"[WARN] Could not load scaler {p}: {e}")
        scalers[k] = None
        scaler_available[k] = False

# Utility: create sequences from numpy array (expects df array shape (N, features))
def create_sequences_from_array(arr, seq_len=30):
    X = []
    for i in range(len(arr) - seq_len + 1):
        X.append(arr[i:i+seq_len])
    return np.array(X)

# Helper: determine dataset key string
def dataset_key_from_id(did):
    return f"FD00{int(did)}"

# Health status
def get_status(rul):
    if rul > 80:
        return "Healthy", "green"
    elif rul > 40:
        return "Caution", "yellow"
    elif rul > 20:
        return "Warning", "orange"
    else:
        return "Critical", "red"

# Core prediction function: returns rul, status, and data used for plotting
def predict_and_visualize(df, use_scaler=True):
    """
    df: raw dataframe uploaded (with cycle, settings, sensors, dataset_id)
    use_scaler: whether to attempt to scale using joblib scalers; if False fallback to raw pipeline
    returns: dict with rul, status, plot divs and summary stats
    """
    print("\n===== DEBUG =====")
    print("DF SHAPE:", df.shape)
    print("DF HEAD:\n", df.head())
    print("=================\n")

    # Ensure dataset_id exists
    if "dataset_id" not in df.columns:
        raise ValueError("CSV missing 'dataset_id' column.")

    dataset_type = int(df["dataset_id"].iloc[0])
    key = dataset_key_from_id(dataset_type)

    # Pipeline A (accurate if scalers are present):
    try:
        if scaler_available.get(key) and use_scaler:
            scaler = scalers[key]
            if hasattr(scaler, "feature_names_in_"):
                scaler_cols = list(scaler.feature_names_in_)
            else:
                scaler_cols = ["cycle", "setting_1", "setting_2", "setting_3"] + [f"sensor_{i}" for i in range(1,22)]

            df_orig = df.copy()
            X_to_scale = df_orig[scaler_cols]
            X_scaled = scaler.transform(X_to_scale)
            df_scaled = pd.DataFrame(X_scaled, columns=scaler_cols)

            # Add one-hot ds columns
            for j in range(1,5):
                col = f"ds_FD00{j}"
                df_scaled[col] = 1 if j == dataset_type else 0

            # Reorder to feature_cols (create missing zeros if necessary)
            try:
                df_final = df_scaled[feature_cols]
            except Exception:
                missing = [c for c in feature_cols if c not in df_scaled.columns]
                for m in missing:
                    df_scaled[m] = 0.0
                df_final = df_scaled[feature_cols]

            arr = df_final.values
            X = create_sequences_from_array(arr)
            # safety: ensure last axis equals model input dim
            if X.shape[-1] != model.input_shape[-1]:
                needed = model.input_shape[-1] - X.shape[-1]
                if needed > 0:
                    pad = np.zeros((X.shape[0], X.shape[1], needed), dtype=X.dtype)
                    X = np.concatenate([X, pad], axis=-1)
                elif needed < 0:
                    X = X[:, :, :model.input_shape[-1]]

            pred = model.predict(X)
            rul = float(pred[-1][0])
            status, color = get_status(rul)
            used_pipeline = "scaler"
            df_used_for_plots = df_final.copy()
        else:
            raise Exception("Scaler not available or disabled.")
    except Exception as e:
        # Pipeline B fallback: bypass scaler
        print("[INFO] Falling back to raw pipeline (no scaler):", e)
        base_cols = [f"sensor_{i}" for i in range(1,22)]
        ds_cols = [f"ds_FD00{j}" for j in range(1,5)]
        for j in range(1,5):
            df[f"ds_FD00{j}"] = 1 if j == dataset_type else 0

        existing_base = [c for c in base_cols if c in df.columns]
        df25 = df[existing_base + ds_cols].copy()

        for c in base_cols:
            if c not in df25.columns:
                df25[c] = 0.0
        for c in ds_cols:
            if c not in df25.columns:
                df25[c] = 0

        arr25 = df25.values
        F = arr25.shape[1]
        needed = model.input_shape[-1] - F
        if needed < 0:
            arr25 = arr25[:, :model.input_shape[-1]]
            needed = 0
        if needed > 0:
            padcols = np.zeros((arr25.shape[0], needed))
            arr_fixed = np.concatenate([arr25, padcols], axis=1)
        else:
            arr_fixed = arr25

        X = create_sequences_from_array(arr_fixed)
        pred = model.predict(X)
        rul = float(pred[-1][0])
        status, color = get_status(rul)
        used_pipeline = "raw_fallback"
        df_used_for_plots = df25.copy()

    # Build visuals with Plotly and Matplotlib
    plots = {}

    # Prepare cycle list and raw sensors list once (use original df for plotting)
    cycles_list = df["cycle"].tolist() if "cycle" in df.columns else list(range(len(df)))
    raw_sensors = [c for c in df.columns if c.startswith("sensor_")]

    # 1) Sensor trends: first 6 sensors
    chosen = raw_sensors[:6] if len(raw_sensors) >= 6 else raw_sensors
    traces = []
    for s in chosen:
        y_vals = df[s].tolist()
        traces.append(go.Scatter(
            x=cycles_list,
            y=y_vals,
            mode='lines',
            name=s
        ))

    layout = go.Layout(
        title="Sensor Trends (First 6 Sensors)",
        xaxis={'title': 'Cycle'},
        yaxis={'title': 'Sensor Value'}
    )
    fig = go.Figure(data=traces, layout=layout)
    plots['sensor_trends'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # 2) RUL gauge
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rul,
        title={'text': "Predicted RUL (cycles)"},
        gauge={
            'axis': {'range': [0, max(200, rul + 50)]},
            'bar': {'color': color},
            'steps': [
                {'range': [0,20], 'color': "red"},
                {'range': [20,40], 'color': "orange"},
                {'range': [40,80], 'color': "yellow"},
                {'range': [80,400], 'color': "green"}
            ]
        }
    ))
    plots['rul_gauge'] = json.dumps(gauge, cls=plotly.utils.PlotlyJSONEncoder)

    # 3) Health metric
    metric = go.Figure(go.Indicator(mode="number+delta", value=rul, title={"text":"RUL"}, domain={'row':0,'column':0}))
    plots['health_metric'] = json.dumps(metric, cls=plotly.utils.PlotlyJSONEncoder)

    # 4) Heatmap (matplotlib) for correlation across sensors
    try:
        sns.set(style="white")
        sensor_cols_for_heat = [c for c in df_used_for_plots.columns if c.startswith("sensor_")]
        if len(sensor_cols_for_heat) >= 2:
            corr = df_used_for_plots[sensor_cols_for_heat].corr()
            plt.figure(figsize=(6,5))
            sns.heatmap(corr, cmap='viridis')
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format='png', dpi=150)
            plt.close()
            buf.seek(0)
            plots['heatmap_png'] = buf.read()
        else:
            plots['heatmap_png'] = None
    except Exception as e:
        print("Heatmap error:", e)
        plots['heatmap_png'] = None

    # 5) Cycle-wise degradation (top 3 sensors from raw df)
    deg_fig = go.Figure()
    top3 = raw_sensors[:3]
    for s in top3:
        y_vals = df[s].tolist()
        deg_fig.add_trace(go.Scatter(
            x=cycles_list,
            y=y_vals,
            mode='lines',
            name=s
        ))
    deg_fig.update_layout(
        title="Cycle-wise Degradation (Top 3 Sensors)",
        xaxis_title="Cycle",
        yaxis_title="Sensor Value"
    )
    plots['degradation'] = json.dumps(deg_fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return results
    return {
        "rul": round(rul, 2),
        "status": status,
        "color": color,
        "used_pipeline": used_pipeline,
        "plots": plots,
        "df_for_plots": df_used_for_plots
    }


# Index: upload form
@app.route("/")
def index():
    return render_template("index.html")

# Predict route: single file upload
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return redirect(url_for('index'))
    file = request.files["file"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{file.filename}")
    file.save(save_path)

    # Read CSV
    df = pd.read_csv(save_path)

    # Run prediction + visuals
    result = predict_and_visualize(df, use_scaler=True)

    # Save heatmap png to disk for embedding in PDF later
    heat_png = None
    if result['plots']['heatmap_png'] is not None:
        heat_png = os.path.join("uploads", f"heatmap_{timestamp}.png")
        with open(heat_png, "wb") as f:
            f.write(result['plots']['heatmap_png'])
    print("\n===== SERVER DEBUG =====")
    print("sensor_trends JSON:", result['plots']['sensor_trends'][:300], "...")
    print("degradation JSON:", result['plots']['degradation'][:300], "...")
    print("========================\n")

    # Return render with plotly divs and PNG path
    return render_template(
        "dashboard.html",
        rul=result['rul'],
        status=result['status'],
        color=result['color'],
        pipeline=result['used_pipeline'],
        sensor_trends= result['plots']['sensor_trends'],
        rul_gauge = result['plots']['rul_gauge'],
        health_metric = result['plots']['health_metric'],
        degradation = result['plots']['degradation'],
        heatmap_png = heat_png,
        filename=file.filename
    )

# Batch endpoint: accept ZIP or CSV with multiple engines (each engine grouped by unit id or dataset partition)
@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    if "file" not in request.files:
        return redirect(url_for('index'))
    file = request.files["file"]
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)
    # Assume CSV with multiple engines (contains unit_number column or engine id)
    df = pd.read_csv(save_path)
    results = []
    # group by unit_number if exists, else by dataset split
    grouping_col = "unit_number" if "unit_number" in df.columns else "dataset_id"
    for name, group in df.groupby(grouping_col):
        try:
            res = predict_and_visualize(group, use_scaler=True)
            results.append({"id": name, "rul": res["rul"], "status": res["status"]})
        except Exception as e:
            results.append({"id": name, "error": str(e)})
    return render_template("batch.html", results=results, filename=file.filename)

# PDF report generator for the last uploaded file
@app.route("/report", methods=["POST"])
def report():
    # Expect form fields: filename in uploads and rul, status; or read latest file in uploads
    filename = request.form.get("filename")
    rul = request.form.get("rul")
    status = request.form.get("status")
    heatmap = request.form.get("heatmap")  # path or None
    # Create PDF
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, height - 60, "PrognosAI — RUL Prediction Report")
    c.setFont("Helvetica", 12)
    c.drawString(40, height - 90, f"File: {filename}")
    c.drawString(40, height - 110, f"Predicted RUL: {rul} cycles")
    c.drawString(40, height - 130, f"Status: {status}")
    c.drawString(40, height - 150, f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # If heatmap available, embed
    if heatmap and os.path.exists(heatmap):
        try:
            c.drawImage(heatmap, 40, height - 450, width=520, height=280)
        except Exception as e:
            print("Could not draw heatmap:", e)
    c.showPage()
    c.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name="PrognosAI_RUL_Report.pdf", mimetype="application/pdf")

if __name__ == "__main__":
    app.run(debug=True)
