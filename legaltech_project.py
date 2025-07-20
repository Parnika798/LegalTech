import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import hstack, csr_matrix
import base64
from io import StringIO
import PyPDF2

# -------------------
# Load spaCy
# -------------------
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# -------------------
# Clean & Lemmatize
# -------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# -------------------
# Load dataset
# -------------------
df = pd.read_csv("Clause Dataset.csv", encoding='latin-1')
df = df[['Clause Text', 'Risk Level']].dropna()
df['Cleaned'] = df['Clause Text'].astype(str).apply(clean_text)
df['Lemmatized'] = df['Cleaned'].apply(lemmatize)
df['Clause Length'] = df['Lemmatized'].apply(lambda x: len(x.split()))
df = df[df['Risk Level'].map(df['Risk Level'].value_counts()) > 1]

# -------------------
# Encode labels
# -------------------
le = LabelEncoder()
y = le.fit_transform(df['Risk Level'])

# -------------------
# TF-IDF + Clause Length
# -------------------
vectorizer = TfidfVectorizer(max_features=5000)
X_text = vectorizer.fit_transform(df['Lemmatized'])
X = hstack([X_text, np.array(df['Clause Length']).reshape(-1, 1)])

# -------------------
# Split and balance
# -------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# -------------------
# Train model
# -------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_bal, y_train_bal)

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="📑 Clause Risk Scanner", layout="centered")
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #fdfdfd;
        }
        .stButton button {
            background-color: #2c3e50;
            color: white;
            font-weight: 600;
            padding: 0.4rem 1rem;
            border-radius: 6px;
        }
        h1, h2, h3, h4, h5 {
            color: #2c3e50;
            font-family: 'Segoe UI', sans-serif;
        }
        .sidebar .sidebar-content {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

st.title("HR ComplianceScope")
st.markdown("Analyze uploaded policy clauses and instantly classify them as High, Medium, or Low Risk.")

with st.sidebar:
    st.markdown("## ⚙️ How It Works")
    st.caption("This scanner uses TF-IDF + Logistic Regression to assess legal/HR clauses.")
    st.markdown("""
    - 🔎 Recognizes sensitive legal/HR terms
    - 📐 Evaluates clause length and structure
    - 🧠 Offers clear HR-friendly summaries
    """)
    st.markdown("---")
    st.markdown("## 🎯 Why Use This")
    st.success("""
    - 🚨 Identify risky content early
    - 📋 Speed up compliance reviews
    - 🤝 Reduce manual oversight errors
    """)

# -------------------
# Risk level color
# -------------------
def color_risk(label):
    color_map = {
        "High": "#e74c3c",
        "Medium": "#f39c12",
        "Low": "#27ae60"
    }
    return f"<span style='color:{color_map.get(label, 'gray')}; font-weight:600'>{label}</span>"

# -------------------
# Clause Extraction
# -------------------
def extract_clauses(content):
    raw_clauses = re.split(r'\n\s*\n', content.strip())
    clauses = []
    for clause in raw_clauses:
        clause_text = clause.replace('\n', ' ').strip()
        if not clause_text or len(clause_text.split()) < 5:
            continue
        title_case_ratio = sum(w.istitle() for w in clause_text.split()) / len(clause_text.split())
        is_title_like = title_case_ratio > 0.8 and len(clause_text.split()) <= 6
        contains_sentence_end = any(p in clause_text for p in ['.', '!', '?'])
        if is_title_like or not contains_sentence_end:
            continue
        clauses.append(clause_text)
    return clauses

# -------------------
# PDF Text Reader
# -------------------
def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# -------------------
# Analyze Uploaded File
# -------------------
if "analyzed_results" not in st.session_state:
    st.session_state.analyzed_results = []

uploaded_file = st.file_uploader("📁 Upload Clause File", type=["txt", "pdf"])
if uploaded_file:
    content = read_pdf(uploaded_file) if uploaded_file.name.endswith(".pdf") else uploaded_file.read().decode("utf-8")
    clauses = extract_clauses(content)

    if st.button("🔍 Analyze Clauses"):
        st.session_state.analyzed_results.clear()
        summary = {label: 0 for label in le.classes_}

        for i, clause in enumerate(clauses):
            cleaned = clean_text(clause)
            lemmatized = lemmatize(cleaned)
            clause_len = len(lemmatized.split())
            x_input = hstack([vectorizer.transform([lemmatized]), np.array([[clause_len]])])
            if not isinstance(x_input, csr_matrix):
                x_input = x_input.tocsr()
            pred = model.predict(x_input)[0]
            label = le.inverse_transform([pred])[0]
            summary[label] += 1

            tfidf_part = x_input[:, :-1].toarray().flatten()
            top_indices = tfidf_part.argsort()[-5:][::-1]
            keywords = [vectorizer.get_feature_names_out()[j] for j in top_indices if tfidf_part[j] > 0]

            if keywords:
                keyword_str = ", ".join([f"<b>{kw}</b>" for kw in keywords])
                explanation = f"Words like {keyword_str} indicate this clause relates to compliance, obligations, or disciplinary policy, prompting a <b>{label}</b> risk label."
            else:
                explanation = f"No strong terms detected, but length and phrasing indicate a <b>{label}</b> risk level."

            st.session_state.analyzed_results.append((i+1, clause, label, explanation))

        # Pie Chart
        st.subheader("📊 Risk Level Distribution")
        fig, ax = plt.subplots()
        ax.pie(summary.values(), labels=summary.keys(), autopct='%1.1f%%', startangle=140, colors=["#e74c3c", "#f39c12", "#27ae60"])
        ax.axis('equal')
        st.pyplot(fig)

        st.markdown(f"### Summary of {len(clauses)} Clauses")
        for l in le.classes_:
            st.markdown(f"- {l} Risk: `{summary[l]}`")

# -------------------
# Show Results with Filter
# -------------------
if st.session_state.analyzed_results:
    choice = st.selectbox("🎛️ Filter by Risk", ["All"] + list(le.classes_))
    for idx, text, label, explanation in st.session_state.analyzed_results:
        if choice != "All" and label != choice:
            continue
        st.markdown(f"---\n#### Clause {idx}")
        st.markdown(f"**Clause:** {text}")
        st.markdown(f"**Risk Level:** {color_risk(label)}", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#f4f6f7;padding:10px;border-left:4px solid #ccc;'>{explanation}</div>", unsafe_allow_html=True)

    st.success("✅ Review complete.")

