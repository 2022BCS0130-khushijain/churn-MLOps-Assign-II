import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    df = pd.read_csv("data/processed/processed.csv")

    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    # MLflow logging
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForest")
        mlflow.log_artifact("models/model.pkl")

if __name__ == "__main__":
    train()