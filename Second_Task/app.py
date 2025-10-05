import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec
import os

# --- Paths ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
W2V_PATH = os.path.join(BASE_DIR, 'word2vec.model')

# --- Load models ---
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check deployment.")
    st.stop()

try:
    w2v_model = Word2Vec.load(W2V_PATH)
except FileNotFoundError:
    st.error(f"Word2Vec model file not found at {W2V_PATH}. Please check deployment.")
    st.stop()

# --- Helper function ---
def get_vector(text):
    words = text.split()
    words = [w for w in words if w in w2v_model.wv]
    if not words:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[words], axis=0)

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("Fake News Detection")
st.write("Enter a news headline or paragraph to check if it's real or fake, along with confidence score.")

user_input = st.text_area("Enter your news text below:")

if st.button("Check News Authenticity"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        input_vec = get_vector(user_input).reshape(1, -1)
        prediction = model.predict(input_vec)[0]

        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(input_vec)) * 100
        else:
            confidence = None

        if prediction == 1:
            st.success("This appears to be Real News.")
        else:
            st.error("This appears to be Fake News.")

        if confidence is not None:
            st.write(f"Model Confidence: {confidence:.2f}%")
        else:
            st.info("Confidence score not available for this model.")

st.markdown("---")
st.caption("Developed using Streamlit, Word2Vec, and scikit-learn")
