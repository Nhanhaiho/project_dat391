import re


def clean_text_basic(text: str):
    """
    Basic cleaning for Vietnamese text
    """
    text = str(text).lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove special characters
    text = re.sub(r"[^a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", "", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_for_baseline(text, stopwords):
    """
    Preprocess text for TF-IDF models
    """
    text = clean_text_basic(text)

    tokens = text.split()

    tokens = [t for t in tokens if t not in stopwords]

    return " ".join(tokens)


def preprocess_for_phobert(text):
    """
    Lightweight preprocessing for PhoBERT
    """
    text = str(text).strip()

    return text