# clause_risk_analyzer.py
import streamlit as st
import pandas as pd
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# --- Rule-based risk scoring ---
high_risk_keywords = ["retaliation", "termination without cause", "discretion", "non-disclosure", "withhold"]
ambiguous_phrases = ["may", "might", "reasonable efforts", "from time to time"]

@st.cache_data

def rule_based_risk_score(text):
    score = 0
    reasons = []
    for word in high_risk_keywords:
        if word in text.lower():
            score += 2
            reasons.append(f"contains high-risk keyword: '{word}'")
    for word in ambiguous_phrases:
        if word in text.lower():
            score += 1
            reasons.append(f"contains ambiguous term: '{word}'")
    return score, reasons

# --- Clause Type Detection ---
def detect_clause_type(text):
    if "retaliation" in text.lower():
        return "Retaliation Risk"
    elif "confidential" in text.lower():
        return "Confidentiality Risk"
    elif "access" in text.lower() or "privacy" in text.lower():
        return "Access/Privacy Risk"
    elif "grievance" in text.lower():
        return "Grievance Procedure"
    else:
        return "General Policy"

# --- Predict Risk with Logic Layer ---
def predict_risk(text):
    rule_score, reasons = rule_based_risk_score(text)
    x_input = vectorizer.transform([text])
    model_pred = model.predict(x_input)[0]
    
    if rule_score >= 3:
        final_risk = "High"
    elif rule_score == 2:
        final_risk = "Medium"
    else:
        final_risk = model_pred

    return final_risk, reasons

# --- Color Codes ---
def risk_color(risk):
    return {
        "Low": "#d4edda",
        "Medium": "#fff3cd",
        "High": "#f8d7da"
    }.get(risk, "#ffffff")

# --- Streamlit UI ---
st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")
st.title("📄 Clause Risk Analyzer for HR Policies")

uploaded_file = st.file_uploader("Upload a text file with clauses (one per paragraph)", type=[".txt"])

if uploaded_file is not None:
    text = uploaded_file.read().decode("utf-8")
    clauses = [para.strip() for para in text.split("\n\n") if para.strip()]
    
    data = []
    for i, clause in enumerate(clauses, 1):
        clause_type = detect_clause_type(clause)
        risk, reasons = predict_risk(clause)
        color = risk_color(risk)

        st.markdown(f"""
        <div style='background-color:{color}; padding:10px; border-radius:10px;'>
            <h4>🧾 Clause {i}</h4>
            <b>Clause Type:</b> {clause_type}<br>
            <b>Predicted Risk Level:</b> <span style='color:red'>{risk}</span><br>
            <b>Why it was flagged:</b>
            <ul>{''.join([f'<li>{reason}</li>' for reason in reasons])}</ul>
            <b>Original Clause:</b><br><i>{clause}</i>
        </div>
        <br>
        """, unsafe_allow_html=True)

    # Downloadable summary
    df = pd.DataFrame({
        "Clause ID": [f"Clause {i+1}" for i in range(len(clauses))],
        "Clause Text": clauses,
        "Clause Type": [detect_clause_type(cl) for cl in clauses],
        "Risk Level": [predict_risk(cl)[0] for cl in clauses]
    })
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Risk Summary CSV", csv, "clause_risk_summary.csv", "text/csv")

