import streamlit as st
import pandas as pd
from src.predict import predict_sentiment

st.set_page_config(
    page_title="Vietnamese Sentiment Analysis",
    layout="centered"
)

st.title("🇻🇳 Vietnamese Sentiment Analysis")
st.write("Nhập review đồ ăn để AI dự đoán cảm xúc.")

if "history" not in st.session_state:
    st.session_state.history = []

# input
text = st.text_area("Nhập review ", height=150)

# buttons
col1, col2 = st.columns(2)

with col1:
    predict_button = st.button("Predict")

with col2:
    reset_button = st.button("Reset")

if reset_button:
    st.session_state.history = []
    st.rerun()

# predict
if predict_button:
    if text.strip() == "":
        st.warning("Vui lòng nhập review")
    else:
        label, confidence = predict_sentiment(text)

        if label == "Positive":
            st.success(f" Sentiment: {label}")
        else:
            st.error(f"Sentiment: {label}")

        st.write(f"Confidence: **{confidence*100:.2f}%**")

        st.session_state.history.append({
            "Review": text,
            "Sentiment": label
        })

# history table
if st.session_state.history:
    st.subheader("Prediction History")

    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    st.subheader("Sentiment Distribution")
    chart_data = df["Sentiment"].value_counts()
    st.bar_chart(chart_data)