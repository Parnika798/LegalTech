import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import hstack

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

# Optional: evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------
# Streamlit UI
# -------------------
st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")
st.title("📄 Clause Risk Level Analyzer")
st.markdown("Upload a `.txt` file with **one legal clause per line** to analyze.")

with st.sidebar:
    st.info("""
This tool uses **TF-IDF** and **Logistic Regression** to analyze legal clauses and flag potential risk.
""")

uploaded_file = st.file_uploader("📂 Upload clause document (.txt)", type=["txt"])

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
# Clause prediction
# -------------------
if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    clauses = [line.strip() for line in content.split("\n") if line.strip()]
    
    if st.button("🔍 Analyze Clauses"):
        st.subheader("📊 Risk Analysis Results")
        for i, clause in enumerate(clauses):
            cleaned = clean_text(clause)
            lemmatized = lemmatize(cleaned)
            clause_len = len(lemmatized.split())
            x_input = vectorizer.transform([lemmatized])
            x_input = hstack([x_input, np.array([[clause_len]])])
            pred_idx = model.predict(x_input)[0]
            risk_label = le.inverse_transform([pred_idx])[0]

            st.markdown(f"---\n### 🧾 Clause {i+1}")
            st.markdown(f"**Original Clause:** {clause}")
            st.markdown(f"**Predicted Risk Level:** {color_risk(risk_label)}", unsafe_allow_html=True)

            # HR-friendly explanation
            tfidf_features = vectorizer.get_feature_names_out()
            clause_array = x_input.toarray().flatten()
            clause_vector = clause_array[:-1]  # all but last is TF-IDF

            top_indices = clause_vector.argsort()[-5:][::-1]
            top_keywords = [tfidf_features[i] for i in top_indices if clause_vector[i] > 0]
            if top_keywords:
                summary = f"The clause was flagged due to keywords: **{', '.join(top_keywords)}**"
            else:
                summary = "No strong keywords identified for this prediction."
            st.info(summary)

        st.success("✅ Completed! Scroll through to view all predictions.")
