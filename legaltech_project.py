import streamlit as st
import re
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = joblib.load("model/logreg_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")

st.title("📄 Clause Risk Analyzer")
st.markdown("Upload a policy document (e.g., grievance policy), and the system will highlight risky clauses with HR-friendly summaries.")

# Color map for risk level
color_map = {
    "Low": "#d4edda",      # Green
    "Medium": "#fff3cd",   # Yellow
    "High": "#f8d7da"      # Red
}

# Sample HR insight generator (template-based)
def get_hr_insight(clause, risk):
    if risk == "High":
        if "termination" in clause.lower() or "suspension" in clause.lower():
            return "Mentions termination/suspension — review disciplinary process."
        elif "retaliation" in clause.lower():
            return "Clause involves retaliation risks."
        elif "confidentiality" in clause.lower():
            return "Clause may involve sensitive or private employee information."
        else:
            return "Contains potential legal or policy risks. Needs thorough HR review."
    elif risk == "Medium":
        return "May involve procedural ambiguity or rights. HR attention advised."
    else:
        return "Clause appears safe or informational."

# Paragraph tokenizer
def split_into_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
    return paragraphs

uploaded_file = st.file_uploader("Upload a policy text file (.txt)", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    paragraphs = split_into_paragraphs(text)

    st.markdown("### 📘 Clause-Level Risk Analysis")
    for idx, clause in enumerate(paragraphs, start=1):
        X = vectorizer.transform([clause])
        risk_pred = model.predict(X)[0]
        hr_summary = get_hr_insight(clause, risk_pred)

        st.markdown(f"""
        <div style="background-color:{color_map[risk_pred]}; padding:15px; border-radius:8px; margin-bottom: 15px;">
            <strong>Clause {idx}:</strong><br>
            <em>{clause}</em><br><br>
            <strong>Risk Level:</strong> <span style="color:black;">{risk_pred}</span><br>
            <strong>HR Insight:</strong> {hr_summary}
        </div>
        """, unsafe_allow_html=True)
