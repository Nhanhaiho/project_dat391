import streamlit as st
from src.predict import predict_sentiment

st.title("Vietnamese Sentiment Analysis")

text = st.text_area("Đồ ăn ok phết, nhưng mà phục vụ chán quá, không có tâm gì cả. Mình sẽ không quay lại đây nữa đâu.")

if st.button("Predict"):
    result = predict_sentiment(text)
    st.write("Sentiment:", result)