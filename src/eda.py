import matplotlib.pyplot as plt
import pandas as pd


def class_distribution(df, label_col="Rating"):
    """
    Return class distribution
    """
    dist = df[label_col].value_counts(dropna=False)
    return dist


def plot_class_distribution(df, label_col="Rating"):
    """
    Plot class distribution
    """
    dist = df[label_col].value_counts()

    dist.plot(kind="bar")

    plt.title("Class Distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    plt.show()


def text_length_stats(df, text_col="Comment"):
    """
    Compute text length statistics
    """
    lengths = df[text_col].astype(str).apply(lambda x: len(x.split()))

    stats = lengths.describe()

    return stats


def plot_text_length_histogram(df, text_col="Comment"):
    """
    Plot histogram of review lengths
    """
    lengths = df[text_col].astype(str).apply(lambda x: len(x.split()))

    plt.hist(lengths, bins=50)

    plt.title("Review Length Distribution")
    plt.xlabel("Number of words")
    plt.ylabel("Frequency")

    plt.show()