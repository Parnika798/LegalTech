import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
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
st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")
st.title("📄 Clause Risk Level Analyzer")

st.sidebar.markdown("## 🤖 What This Tool Does")
st.sidebar.info("""
Uses **TF-IDF + Logistic Regression** to flag legal/HR policy clauses as:
- 🟥 High Risk  
- 🟧 Medium Risk  
- 🟩 Low Risk  

Each clause is analyzed based on legal terms and complexity (length, structure).
""")

st.sidebar.markdown("---")

st.sidebar.markdown("## 💡 Why It Matters")
st.sidebar.success("""
- ⚖️ Spot risky clauses before they escalate  
- ⏱️ Speed up policy & contract reviews  
- 🧠 HR-friendly summaries (no legal jargon)  
- 📊 Bring consistency to manual reviews  
""")

uploaded_file = st.file_uploader("📂 Upload a .txt or .pdf document", type=["txt", "pdf"])

# -------------------
# Risk level color
# -------------------
def color_risk(label):
    color_map = {
        "High": "red",
        "Medium": "orange",
        "Low": "green"
    }
    return f"<span style='color:{color_map.get(label, 'gray')}; font-weight:bold'>{label}</span>"

# -------------------
# Extract clauses
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
# Handle PDF upload
# -------------------
def read_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"
    return text

# -------------------
# Analyze Clauses
# -------------------
if "analyzed_results" not in st.session_state:
    st.session_state.analyzed_results = []

if uploaded_file:
    if uploaded_file.name.endswith(".pdf"):
        content = read_pdf(uploaded_file)
    else:
        content = uploaded_file.read().decode("utf-8")

    clauses = extract_clauses(content)

    if st.button("🔍 Analyze Clauses"):
        st.subheader("📊 Risk Analysis Results")
        st.session_state.analyzed_results.clear()
        risk_counter = {label: 0 for label in le.classes_}

        for i, clause in enumerate(clauses):
            cleaned = clean_text(clause)
            lemmatized = lemmatize(cleaned)
            clause_len = len(lemmatized.split())

            x_input = vectorizer.transform([lemmatized])
            x_input = hstack([x_input, np.array([[clause_len]])])
            if not isinstance(x_input, csr_matrix):
                x_input = x_input.tocsr()

            pred_idx = model.predict(x_input)[0]
            risk_label = le.inverse_transform([pred_idx])[0]
            risk_counter[risk_label] += 1

            tfidf_features = vectorizer.get_feature_names_out()
            tfidf_part = x_input[:, :-1].toarray().flatten()
            top_indices = tfidf_part.argsort()[-5:][::-1]
            top_keywords = [tfidf_features[j] for j in top_indices if tfidf_part[j] > 0]

            if top_keywords:
                explanation = (
                    f"The system identified words like **{', '.join(top_keywords)}**, which are often found "
                    f"in clauses dealing with **disciplinary actions, legal obligations, or policy violations**. "
                    f"These words contributed to the **{risk_label}** risk classification."
                )
            else:
                explanation = (
                    f"The system did not find strong legal or HR-sensitive terms, but based on the **length** and "
                    f"statistical patterns, it assigned a **{risk_label}** risk label."
                )

            st.session_state.analyzed_results.append((i+1, clause, risk_label, explanation))

        st.markdown("### 📈 Summary")
        st.markdown(f"- Total Clauses: `{len(clauses)}`")
        for label in le.classes_:
            st.markdown(f"- {label} Risk: `{risk_counter[label]}`")

# -------------------
# Filtering UI
# -------------------
if st.session_state.analyzed_results:
    filter_option = st.selectbox("🔎 Filter by Risk Level", options=["All"] + list(le.classes_))

    for idx, clause, risk_label, explanation in st.session_state.analyzed_results:
        if filter_option != "All" and risk_label != filter_option:
            continue

        st.markdown(f"---\n### 🧾 Clause {idx}")
        st.markdown(f"**Original Clause:** {clause}")
        st.markdown(f"**Predicted Risk Level:** {color_risk(risk_label)}", unsafe_allow_html=True)
        st.info(explanation)

    st.success("✅ Filter applied. Scroll for results.")

