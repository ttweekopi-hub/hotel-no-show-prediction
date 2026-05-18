# Hotel No-Show Prediction Pipeline

Hello! Welcome to my project repository for the **Hotel No-Show Prediction** technical assessment.

This project is dedicated to understanding and predicting hotel booking cancellations and no-shows. In the hospitality industry, when a customer bookings a room but does not show up (a "no-show"), it leads to lost revenue and empty rooms. My goal is to build an end-to-end Machine Learning pipeline that analyzes historical booking data, builds a predictive model, and deploys it as a FastAPI web application with a live Swagger UI.

---

## 📌 Project Overview
In this project, I am building:
1. **Exploratory Data Analysis (EDA)**: A deep dive into the dataset using a Jupyter Notebook (`notebooks/eda.ipynb`) to uncover trends, identify key factors leading to no-shows, and understand the shape of the data.
2. **End-to-End Machine Learning Pipeline**: A set of modular Python scripts that handle:
   - Data preprocessing and cleaning.
   - Feature engineering (extracting meaningful signals from dates, prices, etc.).
   - Model training and tuning.
   - Model evaluation (using robust metrics like F1-score and ROC-AUC).
3. **FastAPI Web Service**: A lightweight web application that serves the trained model via a clean REST API, allowing users or other services to send a guest's details and receive a prediction on whether they will show up or not.
4. **CI/CD Pipeline**: Automated checks using GitHub Actions that run tests and build the environment, ensuring that every code update is functional.

---

## 📂 Repository Structure
Here is a high-level overview of the structure I am setting up:
```text
hotel-no-show-prediction/
├── .github/workflows/    # GitHub Actions CI/CD workflows
├── data/                 # Data directory (ignored on Git, except for mock setups)
├── models/               # Saved model artifacts (.pkl / .joblib)
├── notebooks/            # Jupyter Notebooks for analysis (e.g., eda.ipynb)
├── src/                  # Modular Python source code (preprocess, train, predict)
├── .gitignore            # Git exclusion rules
├── README.md             # Project documentation (this file)
├── requirements.txt      # Python dependencies
└── run.sh                # Executable shell script to run the entire pipeline
```

---

## 🔒 Data Privacy & Git Policy (`noshow.db`)
> [!IMPORTANT]
> In accordance with the technical assessment guidelines, the main SQLite database **`noshow.db`** containing the raw, private guest data **is strictly excluded from this repository** via the `.gitignore` file and will never be pushed to GitHub.
> 
> **How the pipeline runs in CI/CD:**
> To enable the automated GitHub Actions quality gate to run successfully without the actual dataset, I will implement a mock database generator script. This script automatically spins up a lightweight mock SQLite database with synthetic columns matching the schema of the real database, allowing the entire pipeline (preprocess -> train -> evaluate) to be fully tested automatically on every push!

---

## 🚀 Next Steps
1. **EDA & Data Analysis**: Complete the exploratory data analysis and document visual insights.
2. **Pipeline Development**: Build the preprocessing, training, and evaluation modules inside `src/`.
3. **API & Deployment**: Set up FastAPI and containerize the project using Docker for easy execution.
