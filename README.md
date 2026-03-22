# 🚀 End-to-End MLOps Pipeline for Salary Prediction

## 📌 Project Overview

This project demonstrates a complete **end-to-end MLOps pipeline** for predicting employee salaries using a Machine Learning model.

It covers the full lifecycle:

* Data preprocessing
* Model training
* Experiment tracking
* API deployment
* CI/CD automation
* Docker containerization

---

## 🧠 Problem Statement

Predict employee salary based on:

* Age
* Experience
* Education Level
* Company Size
* Role Level
* Location Tier
* Skills Score

---

## 🏗️ Project Architecture

```
project/
│
├── data/
│   └── salary_data.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│
├── pipeline/
│   └── training_pipeline.py
│
├── model/
│   └── model.pkl
│
├── app.py
├── requirements.txt
├── Dockerfile
└── .github/workflows/ci.yml
```

---

## ⚙️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* MLflow
* Prefect
* Docker
* GitHub Actions

---

## 🔄 Workflow

1. **Data Preprocessing**

   * Load dataset
   * Split features and target

2. **Model Training**

   * Train Linear Regression model
   * Evaluate using R² and RMSE

3. **Experiment Tracking**

   * Log metrics using MLflow

4. **Pipeline Orchestration**

   * Managed using Prefect

5. **API Deployment**

   * FastAPI serves predictions

6. **CI/CD**

   * GitHub Actions automates:

     * Training pipeline
     * Model validation
     * Docker build & push

---

## 🧪 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlops-salary-prediction.git
cd mlops-salary-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run training pipeline

```bash
python pipeline/training_pipeline.py
```

### 5. Start FastAPI server

```bash
uvicorn app:app --reload
```

👉 Open: http://127.0.0.1:8000/docs

---

## 🐳 Docker Usage

### Build image

```bash
docker build -t yourusername/mlops-salary-prediction .
```

### Run container

```bash
docker run -p 8000:8000 yourusername/mlops-salary-prediction
```

---

## 🔁 CI/CD Pipeline

GitHub Actions workflow:

* Runs training pipeline
* Verifies model creation
* Builds Docker image
* Pushes to Docker Hub

---

## 📊 API Endpoint

### POST `/predict`

#### Input:

```json
{
  "age": 30,
  "experience": 5,
  "education_level": 2,
  "company_size": 2,
  "role_level": 2,
  "location_tier": 1,
  "skills_score": 7
}
```

#### Output:

```json
{
  "predicted_salary": 850000
}
```

---

## 🎯 Key Features

* Modular project structure
* MLflow experiment tracking
* Prefect pipeline orchestration
* FastAPI deployment
* Docker containerization
* CI/CD with GitHub Actions

---

## 🚀 Future Improvements

* Add feature scaling & pipelines
* Hyperparameter tuning (GridSearchCV)
* Model versioning & monitoring
* Cloud deployment (AWS / Render)

---

## 🙌 Acknowledgement

This project was built as part of a self-learning journey in MLOps, focusing on practical implementation and real-world system design.

---

## 📌 Author

Your Name
GitHub: https://github.com/your-username
