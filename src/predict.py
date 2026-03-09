import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "models/phobert_binary_best"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

def predict_sentiment(text):

    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()

    if pred == 1:
        return "Positive"
    else:
        return "Negative"