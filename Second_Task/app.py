import streamlit as st
import joblib
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# Load vectorizers and models
# -----------------------------
# Word2Vec (for Decision Tree / Random Forest)
w2v_model = Word2Vec.load("word2vec.model")
Rfw2v = joblib.load("RandomForestw2v.pkl")  # uses Word2Vec

# TF-IDF (for Logistic Regression / SVM)
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
Rftf = joblib.load("RandomForesttf.pkl")       # uses TF-IDF

# Map model names to models and their vectorizer type
models = {
    "Word2Vec": {"model": Rfw2v, "vectorizer": "word2vec"},
    "TF-IDF": {"model": Rftf, "vectorizer": "tfidf"},
}

# -----------------------------
# Helper functions
# -----------------------------
def text_to_w2v_vector(text):
    words = text.split()
    words = [w for w in words if w in w2v_model.wv]
    if len(words) == 0:
        return np.zeros(w2v_model.vector_size)
    return np.mean(w2v_model.wv[words], axis=0)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("Fake News Detection App")
st.write("Enter a news headline or paragraph to check if it's real or fake.")

# Sidebar for selecting model
st.sidebar.title("Choose a Vectorizer")
selected_model_name = st.sidebar.radio("Model", list(models.keys()))
selected_model_info = models[selected_model_name]
selected_model = selected_model_info["model"]
vectorizer_type = selected_model_info["vectorizer"]

# User input
user_input = st.text_area("Enter your news text:")

if st.button("Check News Authenticity"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Choose vectorization method based on selected model
        if vectorizer_type == "word2vec":
            input_vec = text_to_w2v_vector(user_input).reshape(1, -1)
        elif vectorizer_type == "tfidf":
            input_vec = tfidf_vectorizer.transform([user_input])
        else:
            st.error("Unknown vectorizer type!")
            st.stop()

        # Prediction
        prediction = selected_model.predict(input_vec)[0]

        # Confidence if available
        confidence = None
        if hasattr(selected_model, "predict_proba"):
            confidence = np.max(selected_model.predict_proba(input_vec)) * 100

        # Display results
        if prediction == 1:
            st.success("This seems to be Real News")
        else:
            st.error("This seems to be Fake News")

        if confidence is not None:
            st.info(f"Model Confidence: {confidence:.2f}%")
