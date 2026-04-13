import pandas as pd
import joblib
import mlflow
from sklearn.metrics import f1_score, roc_auc_score

mlflow.set_tracking_uri("sqlite:///mlflow.db")

@mlflow.trace
def load_artifacts():
    df = pd.read_csv("data/processed/processed.csv")
    model = joblib.load("models/model.pkl")
    return df, model

@mlflow.trace
def compute_metrics(model, X, y):
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"f1": f1_score(y, pred), "roc_auc": roc_auc_score(y, prob)}

def evaluate():
    with mlflow.start_run():
        df, model = load_artifacts()

        X = df.drop("churn", axis=1)
        y = df["churn"]

        metrics = compute_metrics(model, X, y)

        mlflow.log_metrics(metrics)
        print("F1 Score:", metrics["f1"])
        print("ROC-AUC:", metrics["roc_auc"])

if __name__ == "__main__":
    evaluate()