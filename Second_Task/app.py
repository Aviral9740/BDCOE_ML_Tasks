import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#Loading model and vectorizer
@st.cache_resource
def load_tfidf_model():
    try:
        model = joblib.load("RandomForesttf.pkl")  
        vectorizer = joblib.load("tfidf_vectorizer.pkl")  
        return model, vectorizer
    except FileNotFoundError as e:
        st.error("Model or vectorizer file not found. Make sure both 'RandomForesttf.pkl' and 'tfidf_vectorizer.pkl' exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        st.stop()

# Load resources
with st.spinner("Loading model and vectorizer..."):
    model, tfidf_vectorizer = load_tfidf_model()

def predict_fake_news(text):
    transformed_text = tfidf_vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(transformed_text)) * 100
    return prediction, confidence

#UI
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>Fake News Detection App</h1>
    <p style='text-align: center; color: #555;'>
    Enter a news headline or paragraph to check if it's real or fake using a TF-IDF based machine learning model.
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Text input area
user_input = st.text_area("Enter your news text below:")

# Prediction button
if st.button("Check News Authenticity"):
    if not user_input.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Analyzing..."):
            prediction, confidence = predict_fake_news(user_input)

        # Display results
        if prediction == 1:
            st.success("This seems to be Real News.")
        else:
            st.error("This seems to be Fake News.")

        if confidence is not None:
            st.info(f"Model Confidence: **{confidence:.2f}%**")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Developed with Streamlit and scikit-learn | TF-IDF based model")
