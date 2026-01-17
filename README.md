🌍 AI Emulator for Weather & Earth System Modeling

This project presents a deep learning–based AI emulator for weather prediction and extreme event detection using historical and live meteorological data.
The system trains and compares multiple neural network architectures, evaluates them using standard research metrics, and deploys them for real-time inference.

🚀 Project Motivation

Traditional weather prediction models are computationally expensive and complex.
This project explores how AI emulators can approximate weather parameterization using historical climate data while remaining efficient and deployable.

Unlike simple prediction apps, this project includes:

Multiple deep learning models

Proper train/validation/test evaluation

Quantitative comparison using metrics

Live data inference

This adds a strong research component, suitable for academic publication.

📊 Dataset
Source

NASA POWER (Prediction of Worldwide Energy Resources)

Hourly historical weather data

Locations Used

Bangalore

Mumbai

Chennai

Delhi

Input Features

Rainfall

Temperature

Wind speed

Relative humidity

Surface pressure

Output Targets

Regression (Next hour):

Temperature

Rainfall

Wind speed

Classification (Extreme events):

Cloudburst

Thunderstorm

Heatwave

Coldwave

Cyclone-like

🧠 Dataset Engineering

A sliding window approach is used:

Input: past 6 hours × 5 features

Output: next-hour predictions

This converts raw weather data into a time-series supervised learning problem.

Implemented in:

datasets/window_dataset.py

🏗️ Models Implemented

All models use the same dataset, inputs, targets, and normalization for fair comparison.

Model	Description
MLP	Baseline model (no temporal awareness)
LSTM	Captures long-term temporal dependencies
GRU	Efficient alternative to LSTM
CNN-LSTM	Combines local feature extraction + temporal modeling
Transformer	Attention-based long-range dependency modeling

Each model predicts:

Continuous weather variables (regression)

Extreme event probabilities (classification)

⚙️ Training Setup

Loss Function:

Regression → MSE Loss

Classification → BCEWithLogits Loss

Optimizer: Adam

Same hyperparameters across models

Models saved as .pt checkpoints

Training scripts are located in:

train/

📈 Evaluation Methodology
Data Split

Training: 70%

Validation: 15%

Testing: 15% (unseen during training)

Metrics Used

Classification:

Accuracy

Precision

Recall

F1-score

Regression:

Mean Absolute Error (MAE)

Evaluation is performed only on the test set, ensuring unbiased results.

Evaluation script:

eval/evaluate_all_models.py

🌐 Live Inference (Deployment)

The project supports real-time weather inference using live data.

Live Pipeline

Fetch last 6 hours of weather data (Open-Meteo API)

Normalize using training statistics

Run inference using trained models

Output:

Next-hour weather predictions

Extreme event probabilities

Live inference code:

live_inference/


Run:

python -m live_inference.run_live

🔄 Multi-Model Live Comparison

A comparison module allows the same live input to be passed into all trained models, enabling side-by-side comparison of predictions.

This demonstrates:

Model stability

Sensitivity to real-world data

Differences in temporal reasoning

Run:

python -m live_inference.compare_models

🧪 Key Results

Temporal models (CNN-LSTM, GRU, Transformer) outperform MLP

CNN-LSTM achieves strong balance between accuracy and stability

Transformer shows potential but requires higher computational resources

Live inference demonstrates deployability of AI emulators

📂 Project Structure
ai_emulator/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── datasets/
│   └── window_dataset.py
│
├── models/
│   ├── mlp.py
│   ├── lstm.py
│   ├── gru.py
│   ├── cnn_lstm.py
│   └── transformer.py
│
├── train/
│   └── training scripts
│
├── eval/
│   └── evaluate_all_models.py
│
├── live_inference/
│   ├── run_live.py
│   ├── compare_models.py
│   ├── live_fetch.py
│   └── normalize.py
│
└── README.md

📝 Conclusion

This project demonstrates how AI emulators can effectively model weather dynamics by learning from historical data.
By combining rigorous evaluation with live deployment, the system bridges the gap between research and real-world application.
