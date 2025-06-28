🚦 TrafficTelligence - Advanced Traffic Volume Estimation with Machine Learning

The **TrafficTelligence** project aims to revolutionize traffic management through advanced machine learning techniques for **accurate and intelligent traffic volume estimation**.

---

📌 Project Overview

This system predicts traffic volume using historical, temporal, weather, and holiday data. It applies preprocessing, exploration, and machine learning models to determine the best predictor of traffic patterns.

---

📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup](#setup)
- [Steps in the Pipeline](#steps-in-the-pipeline)
- [Model Evaluation](#model-evaluation)
- [Deployment](#deployment)
- [Visualizations](#visualizations)
- [Future Enhancements](#future-enhancements)
- [Authors](#authors)

---

✨ Features

The dataset includes the following features:

- **Temporal Data**: Extracted from datetime – day, month, year, hour, minute, second.
- **Weather Conditions**: Temperature, rainfall, snowfall, and weather categories.
- **Traffic Volume**: Target variable indicating the number of vehicles.
- **Holidays**: Whether the date is a public holiday.

---

🧰 Technologies Used

**Programming Language**: Python

**Libraries & Tools**:

- `pandas`, `numpy` – Data manipulation
- `seaborn`, `matplotlib` – Visualizations
- `scikit-learn` – Machine learning algorithms and tools
- `xgboost` – Gradient boosting model
- `pickle` – Model serialization (used inside `model.zip`)


---
📂 Directory Structure
TrafficTelligence/
├── .ipynb_checkpoints/
├── Flask/
│ ├── templates/
│ ├── app.py
│ ├── encoder.pkl
│ └── model.pkl 👈 Extracted from model.zip
│
├── IBM/
│ └── Flask/
│ └── traffic_volume_lbm_scoring_end_point.ipynb
│
├── Requirements.txt
├── Traffic volume estimation.docx
├── traffic volume.csv
├── traffic volume.ipynb

⚙️ Setup Instructions

1. **Clone the repository**:

git clone https://github.com/your-username/TrafficTelligence.git
cd TrafficTelligence

Install dependencies:
pip install pandas numpy seaborn matplotlib scikit-learn xgboost

Prepare the dataset:
Place your dataset file (traffic volume.csv) inside the project  folder.
dataset link: https://drive.google.com/file/d/1iV5PfYAmI6YP0_0S4KYy1ZahHOqMgDbM/view

Extract the trained model:
Place model.zip into the model/ folder.

Extract model.zip so that it contains model.pkl files inside.

Model.zip/
├── model.pkl

🧠 These files will be used for future predictions.

Run the main script:
cd “TrafficTelligence/Flask
python app.py


🔄 Steps in the Pipeline
1. 📊 Data Preprocessing
Load dataset

Handle missing values:

Numeric → Replace with mean

Categorical → Replace with most frequent ('Clouds')

Date-Time split into multiple columns

Label encode categorical values

Standardize numerical values

2. 🔍 Exploratory Data Analysis
Correlation heatmap

Count plots

Pair plots

Box plots

3. 🧠 Model Training
Trained models include:

Linear Regression

Decision Tree Regressor

Random Forest Regressor ✅ (Best performer)

Support Vector Regressor

XGBoost Regressor

4. 📈 Model Evaluation
Evaluated using:

R-squared Score (R²): Measures prediction accuracy

Root Mean Squared Error (RMSE): Average prediction error

🏆 Model Evaluation
Best Model: Random Forest Regressor

R² Score:

Training: 0.92

Validation: 0.86

Reason for selection: Lowest RMSE and highest accuracy

🚀 Deployment
Best model and encoder saved as:

model/model.pkl

model/encoder.pkl

Use pickle to load these files in your app for prediction.

Visualizations:
📉 Correlation Heatmap

🔁 Pair Plots

📊 Count Plots

📦 Box Plots

🧪 Future Enhancements
Add live traffic or road condition features.

Apply hyperparameter tuning (e.g., GridSearchCV).

Deploy as a web app or REST API for real-time predictions.

👥 Authors:
Abdul Anees
contact anees.abdul420@gmail.com for queries
