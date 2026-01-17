# 🌦️ AI Emulator for Weather & Earth System Modeling

A deep learning–based emulator for weather prediction and extreme event detection.  
This project trains, evaluates, and compares multiple neural network architectures on historical and live weather data, supporting real-time inference.

---

## 🚀 Project Overview

Traditional numerical weather prediction models are computationally expensive and time-consuming.  
This project explores **AI-based weather emulation** as an efficient alternative to approximate weather parameterization using historical data.

The system focuses on:
- Fast inference using deep learning
- Comparison of multiple neural network architectures
- Detection of extreme weather events
- Live weather prediction using real-time API data

---

## ✨ Features

- Predict next-hour weather parameters
- Detect extreme weather events (heatwaves, thunderstorms, cyclones, heavy rainfall)
- Train and evaluate multiple deep learning models
- Live real-time inference using weather APIs
- Consistent benchmarking across models

---

## 📦 Dataset

**Source:** NASA POWER (Prediction of Worldwide Energy Resources)

**Input Features:**
- Temperature
- Rainfall
- Wind speed
- Relative humidity
- Surface pressure

**Output:**
- Continuous weather predictions (regression)
- Extreme event probability (classification)

**Preprocessing:**
- Sliding window approach
- Uses past 6 hours of weather data to predict the next hour

---

## 🧠 Models Implemented

| Model | Description |
|------|------------|
| MLP | Baseline feedforward neural network |
| LSTM | Captures long-term temporal dependencies |
| GRU | Efficient sequence modeling |
| CNN-LSTM | Local feature extraction + temporal modeling |
| Transformer | Attention-based long-range dependency learning |

---

## 🛠️ Training Details

**Loss Functions**
- Regression: Mean Squared Error (MSE)
- Classification: Binary Cross-Entropy with Logits

**Optimizer**
- Adam optimizer with consistent hyperparameters

**Data Split**
- Training: 70%
- Validation: 15%
- Testing: 15%

**Evaluation Metrics**
- MAE for regression
- Accuracy, Precision, Recall, and F1-score for classification

---

## 📊 Model Evaluation

Evaluate all trained models on the test dataset using:
```bash
python -m eval.evaluate_all_models

**## 🌐 Live Inference**

The project supports real-time weather prediction using external weather APIs.

Run live inference:
```bash
python -m live_inference.run_live

Compare predictions from different models:
python -m live_inference.compare_models

**📁 Project Structure**
ai_emulator_weather/
├── data/
│   ├── raw/
│   └── processed/
├── datasets/
│   └── window_dataset.py
├── models/
│   ├── mlp.py
│   ├── lstm.py
│   ├── gru.py
│   ├── cnn_lstm.py
│   └── transformer.py
├── train/
├── eval/
│   └── evaluate_all_models.py
├── live_inference/
│   ├── run_live.py
│   ├── compare_models.py
│   ├── live_fetch.py
│   └── normalize.py
└── README.md
