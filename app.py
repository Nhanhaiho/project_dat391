import streamlit as st
from src.predict import predict_sentiment

st.title("Vietnamese Sentiment Analysis")

text = st.text_area("Đồ ăn ngon thế")

if st.button("Predict"):
    result = predict_sentiment(text)
    st.write("Sentiment:", result)