from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score


def build_tfidf(train_text, val_text, test_text):
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1,2)
    )

    X_train = vectorizer.fit_transform(train_text)
    X_val = vectorizer.transform(val_text)
    X_test = vectorizer.transform(test_text)

    return vectorizer, X_train, X_val, X_test


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train):
    model = LinearSVC()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)

    report = classification_report(y_val, preds)

    return acc, report