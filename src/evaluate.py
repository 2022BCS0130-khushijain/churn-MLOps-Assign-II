import pandas as pd
import joblib
from sklearn.metrics import f1_score, roc_auc_score

def evaluate():
    df = pd.read_csv("data/processed/processed.csv")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    model = joblib.load("models/model.pkl")

    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    print("F1 Score:", f1_score(y, pred))
    print("ROC-AUC:", roc_auc_score(y, prob))

if __name__ == "__main__":
    evaluate()