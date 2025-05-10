# CS 4824 - Anomaly Detection in IoT-Enabled Smart Real Estate Systems

This project investigates and compares two approaches to detecting anomalies in smart building energy usage data:

- A **statistical method** using Z-score analysis
- A **deep learning approach** using an LSTM neural network

It uses a synthetic dataset that simulates energy usage alongside temperature, humidity, and occupancy information.

---

## Project Structure

| File                          | Description |
|------------------------------|-------------|
| `synthetic_energy_data.csv`  | Simulated smart building dataset with timestamps and energy usage |
| `anomaly_detection_project.py` | Loads and displays dataset overview and class balance |
| `statistical_anomaly_detector.py` | Detects anomalies using z-score thresholding and plots results |
| `lstm_preprocessing.py`      | Scales energy data, creates sequences, and sets up train/test loaders |
| `lstm_model.py`              | Defines the LSTM architecture and trains the model |
| `lstm_plot_results.py`       | Loads trained model and plots predicted vs actual energy values |
| `README.md`                  | Project overview and instructions |

---

## Dataset Description

- `timestamp`: hourly time data
- `temperature`, `humidity`, `occupancy`: sensor data
- `energy`: target variable (energy usage)
- `anomaly`: labeled anomaly column (for evaluation)

---

## Methods

### 1. Statistical Anomaly Detection (Z-score)
- Computes z-scores for each feature
- Flags points where any feature exceeds a threshold (`z > 3`)
- Highlights anomalies on an energy usage plot

### 2. LSTM-Based Anomaly Detection
- Preprocesses time series data into 24-hour windows
- Trains an LSTM to predict next energy value
- Compares predictions to actual values for evaluation

---

## How to Run

> Make sure you have Python 3.8+ and the required packages:

```bash
pip install pandas numpy matplotlib torch scikit-learn scipy
