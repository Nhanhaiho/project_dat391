import pandas as pd


def load_train_data(path: str):
    """
    Load training dataset
    """
    df = pd.read_csv(path)
    return df


def load_test_data(path: str):
    """
    Load test dataset (no labels)
    """
    df = pd.read_csv(path)
    return df


def load_stopwords(path: str):
    """
    Load Vietnamese stopwords
    """
    with open(path, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f if line.strip()])

    return stopwords