import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec

# Load the trained classifier and Word2Vec model
model = joblib.load('model.pkl')
w2v_model = Word2Vec.load('word2vec.model')

# Helper function to convert text → vector
def get_vector(text):
    words = text.split()
    words = [w for w in words if w in w2v_model.wv]
    if len(words) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[words], axis=0)

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("Fake News Detection App")
st.write("Enter a news headline or paragraph to check if it is **real or fake**. The model also provides a confidence score if available.")

# User input
user_input = st.text_area("Enter the news text below:")

# Button to check news authenticity
if st.button("Check News Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Convert input text to vector
        input_vec = get_vector(user_input).reshape(1, -1)

        # Predict and get confidence
        prediction = model.predict(input_vec)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_vec)[0]
            confidence = np.max(probs) * 100

        # Display results
        if prediction == 1:
            st.success("This appears to be **Real News**.")
        else:
            st.error("This appears to be **Fake News**.")

        if confidence is not None:
            st.write(f"**Model Confidence:** {confidence:.2f}%")
        else:
            st.info("Confidence score is not available for this model.")

st.markdown("---")
st.caption("Developed using Streamlit, Word2Vec, and scikit-learn")
