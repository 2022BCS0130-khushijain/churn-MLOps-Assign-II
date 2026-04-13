import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

@mlflow.trace
def load_data():
    df = pd.read_csv("data/processed/processed.csv")
    return df

@mlflow.trace
def split_data(df):
    X = df.drop("churn", axis=1)
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2)

@mlflow.trace
def fit_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def train():
    with mlflow.start_run():
        mlflow.log_param("model", "RandomForest")

        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)
        model = fit_model(X_train, y_train)

        joblib.dump(model, "models/model.pkl")
        mlflow.log_artifact("models/model.pkl")

if __name__ == "__main__":
    train()