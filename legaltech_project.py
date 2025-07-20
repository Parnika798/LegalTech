import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import joblib

from scipy.sparse import hstack

# =======================
# Load spaCy model
# =======================
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =======================
# Text preprocessing
# =======================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def generate_hr_summary(risk):
    if risk == "High":
        return "This clause contains high-risk language or lacks protections. HR should review closely."
    elif risk == "Medium":
        return "This clause contains moderate concerns. May need revision."
    else:
        return "Clause appears standard and low-risk."

risk_colors = {
    "Low": "#d4edda",     # Green
    "Medium": "#fff3cd",  # Yellow
    "High": "#f8d7da"     # Red
}

# =======================
# Load Model and Vectorizer
# =======================
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("logreg_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =======================
# Streamlit App
# =======================
st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")
st.title("📄 Clause Risk Level Analyzer")

st.markdown("""
Upload a `.txt` file containing a **policy or document**.  
Each paragraph will be treated as a clause.  
This app will:
- Predict the **risk level**
- Generate a **summary**
- Display **color-coded insights**  
""")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses a TF-IDF + Logistic Regression model to identify legal clause risk levels.  
    It’s trained on labeled HR/Legal policy clauses and provides HR-friendly summaries.
    """)

uploaded_file = st.file_uploader("📂 Upload a .txt document", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    clauses = [p.strip() for p in content.split("\n") if p.strip()]

    st.subheader("🔎 Results")

    for i, clause in enumerate(clauses):
        cleaned = clean_text(clause)
        lemmatized = tokenize_and_lemmatize(cleaned)
        clause_len = len(lemmatized.split())

        X_tfidf = vectorizer.transform([lemmatized])
        X_final = hstack([X_tfidf, np.array([[clause_len]])])

        pred = model.predict(X_final)[0]
        pred_label = label_encoder.inverse_transform([pred])[0]
        hr_summary = generate_hr_summary(pred_label)

        # Colored box
        st.markdown(f"""
        <div style='background-color:{risk_colors[pred_label]}; padding: 15px; border-radius: 10px; margin-bottom: 15px'>
            <h4>Clause {i+1}</h4>
            <p><strong>Clause Text:</strong><br>{clause}</p>
            <p><strong>Predicted Risk Level:</strong> <span style='font-weight:bold;'>{pred_label}</span></p>
            <p><strong>HR Summary:</strong> {hr_summary}</p>
        </div>
        """, unsafe_allow_html=True)

    st.success("✅ All clauses analyzed. Scroll to view results.")

