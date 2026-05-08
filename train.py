import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score

def train_model():
    # 1. Load Kaggle Data
    print("Loading dataset...")
    df = pd.read_csv('../data/creditcard.csv') # Ensure file is here
    
    # 2. Preprocessing
    # Scaling 'Amount' and 'Time' (the only non-scaled features)
    scaler = StandardScaler()
    df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Handle Imbalance
    # Calculate ratio: (Negative Cases / Positive Cases)
    case_ratio = len(y_train[y_train==0]) / len(y_train[y_train==1])
    
    print(f"Training XGBoost with imbalance ratio: {case_ratio:.2f}")
    
    # 4. Train Model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=case_ratio, # Crucial step for fraud
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    preds = model.predict(X_test)
    print("\nModel Performance:")
    print(classification_report(y_test, preds))
    print(f"Area Under PR Curve: {average_precision_score(y_test, model.predict_proba(X_test)[:,1]):.4f}")
    
    # 6. Save Model and Scaler
    joblib.dump(model, '../models/fraud_model.pkl')
    joblib.dump(scaler, '../models/scaler.pkl')
    print("Model saved successfully in /models/")

if __name__ == "__main__":
    train_model()