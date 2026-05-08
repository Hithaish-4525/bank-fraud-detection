#  FinGuard: Bank Fraud Detection System

An end-to-end Machine Learning system that detects fraudulent transactions using XGBoost, FastAPI, and Streamlit.

##  Project Overview
This project uses the Kaggle Credit Card Fraud Detection dataset to train a model that handles extreme class imbalance (only 0.17% fraud). It includes:
- **FastAPI**: A high-performance backend to serve predictions.
- **Streamlit**: A dashboard for fraud analysts to visualize risks.
- **XGBoost**: A powerful model optimized for detecting rare fraud patterns.

##  Tech Stack
- Python (XGBoost, Pandas, Scikit-Learn)
- FastAPI (API Development)
- Streamlit (Frontend Dashboard)
- Joblib (Model Serialization)

##  Folder Structure
- `api/`: FastAPI backend logic.
- `dashboard/`: Streamlit UI.
- `models/`: Saved model and scaler files.
- `src/`: Training and preprocessing scripts.

##  How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Download `creditcard.csv` from Kaggle and place in `/data`.
3. Train model: `python src/train.py`
4. Run API: `uvicorn api.main:app --reload`
5. Run Dashboard: `streamlit run dashboard/app.py`
