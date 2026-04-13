import pandas as pd

def preprocess():
    df = pd.read_csv("data/raw/data.csv")

    # Simple cleaning (extend if needed)
    df.fillna(0, inplace=True)

    df.to_csv("data/processed/processed.csv", index=False)

if __name__ == "__main__":
    preprocess()