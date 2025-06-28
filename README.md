# 🚦 TrafficTelligence: Advanced Traffic Volume Estimation Using Machine Learning

## 📌 Project Overview

**TrafficTelligence** is an advanced traffic prediction system that utilizes structured data and machine learning algorithms to estimate traffic volume. Unlike traditional systems that rely on CCTV, YOLO, or video input, TrafficTelligence leverages a comprehensive dataset with features like climate, holiday, and rainy conditions to provide accurate, data-driven traffic forecasting.

## 🔍 Problem Statement

Urban traffic congestion results in increased travel times, pollution, and stress. Traditional traffic estimation systems often rely on expensive infrastructure like cameras and sensors. Our goal is to create a cost-effective solution that uses environmental and temporal data to forecast traffic patterns.

## 🎯 Objectives

- Predict traffic volume using structured datasets
- Enable authorities to plan and manage traffic flow proactively
- Lay the foundation for real-time system integration in the future

## 🧠 Technologies Used

- **Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy` – Data Processing  
  - `matplotlib`, `seaborn` – Data Visualization  
  - `scikit-learn` – Machine Learning Models  
  - `joblib` – Model Saving  
- **Tools:** Jupyter Notebook, VS Code  
- **Version Control:** Git, GitHub

## 🗂️ Project Structure
TrafficTelligence/
├── .ipynb_checkpoints/ # Jupyter notebook checkpoints
├── Flask/
│ ├── templates/ # HTML templates (for Flask UI)
│ ├── app.py # Flask application script
│ ├── encoder.pkl # Encoded label classes
│ └── model.pkl # Trained ML model
├── IBM/
│ └── Flask/
│ └── traffic_volume_lbm_scoring end point.ipynb # IBM cloud model deployment/testing
├── Requirements.txt # List of required Python packages
├── Traffic volume estimation.docx # Project report/documentation
├── traffic volume.csv # Dataset used for training/testing
├── traffic volume.ipynb # Jupyter notebook for EDA and modeling


## 📊 Dataset Features

- `date_time`  
- `holiday`  
- `rain_1h`, `snow_1h`, `clouds_all`  
- `weather_main`, `weather_description`  
- `temp`, `humidity`, `wind_speed`  
- `traffic_volume` (target)

## 🧪 Key Features

- Predicts hourly traffic volume
- Flask-based web application for user interaction
- Model deployment compatibility with IBM Cloud
- Clean data preprocessing and model training pipeline
- Exported model and encoder for real-time use

## 🧠 Technologies Used

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `flask`
- **ML Model:** Random Forest Regressor (trained & saved as `model.pkl`)
- **Deployment:** Flask Web Framework, IBM Cloud (Jupyter)
- **IDE:** Jupyter Notebook, VS Code



## 🚀 Model Summary

- **Algorithm Used:** Random Forest Regressor  
- **Performance Metrics:**  
  - Training R² Score: **0.92**  
  - Validation R² Score: **0.86**

## 🛠️ How to Run

1. **Clone the repository:**

   git clone https://github.com/yourusername/TrafficTelligence.git

    cd TrafficTelligence

   Install dependencies:
    pip install -r requirements.txt

    To run:
    cd "TrafficTelligence:Advanced Traffic Volume Estimation using MachineLearning\Flask"
    python app.py

📄 Requirements File
pandas, numpy, matplotlib, scikit-learn, pickle, flask

📬 Contact
For any queries, contact: anees.abdul420@gmail.com

