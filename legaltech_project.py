import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import lime
import lime.lime_text
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import hstack

# =====================
# Load spaCy model
# =====================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# =====================
# Cleaning Functions
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def tokenize_and_lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

# =====================
# Load and preprocess dataset
# =====================
df = pd.read_csv("Clause Dataset.csv", encoding='latin-1')
df = df[['Clause Text', 'Risk Level']].dropna()
df['Cleaned Clause'] = df['Clause Text'].astype(str).apply(clean_text)
df['Lemmatized Clause'] = df['Cleaned Clause'].apply(tokenize_and_lemmatize)
df['clause_len'] = df['Lemmatized Clause'].apply(lambda x: len(x.split()))
df = df[df['Risk Level'].map(df['Risk Level'].value_counts()) > 1]

# Label encode
label_encoder = LabelEncoder()
y_risk = label_encoder.fit_transform(df['Risk Level'])

# =====================
# TF-IDF + Clause Length Features
# =====================
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(df['Lemmatized Clause'])
X_final = hstack([X_tfidf, np.array(df['clause_len']).reshape(-1, 1)])

# =====================
# Split and balance
# =====================
X_train, X_test, y_train, y_test, text_train, text_test = train_test_split(
    X_final, y_risk, df['Lemmatized Clause'], test_size=0.2, stratify=y_risk, random_state=42
)
ros = RandomOverSampler(random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

# =====================
# Train final model
# =====================
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_bal, y_train_bal)

# Evaluation (optional log)
y_pred = logreg.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# =====================
# LIME setup
# =====================
lime_vectorizer = TfidfVectorizer(max_features=5000)
lime_pipeline = make_pipeline(lime_vectorizer, LogisticRegression(max_iter=1000, class_weight='balanced'))
lime_pipeline.fit(df['Lemmatized Clause'], y_risk)
explainer = lime.lime_text.LimeTextExplainer(class_names=label_encoder.classes_)

# =====================
# Streamlit UI
# =====================
st.set_page_config(page_title="Clause Risk Analyzer", layout="wide")
st.title("📄 Clause Risk Level Analyzer")
st.markdown("""
Upload a text document containing **multiple legal clauses** (one clause per line).
This tool will:
- Predict **Risk Level** for each clause
- Show **LIME explanations** and **simplified summaries** of why that prediction was made
""")

with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("This tool uses a Logistic Regression model trained on TF-IDF vectors from legal clauses, along with LIME to explain predictions.")

uploaded_file = st.file_uploader("📂 Upload a .txt document", type=["txt"])

if uploaded_file:
    content = uploaded_file.read().decode("utf-8")
    clauses = [clause.strip() for clause in content.split("\n") if clause.strip()]

    if st.button("🔍 Analyze Clauses"):
        st.subheader("🔎 Results")
        for i, clause in enumerate(clauses):
            cleaned = clean_text(clause)
            lemmatized = tokenize_and_lemmatize(cleaned)
            pred = lime_pipeline.predict([lemmatized])[0]
            pred_label = label_encoder.inverse_transform([pred])[0]

            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"### 🧾 Clause {i+1}")
            with col2:
                st.markdown(f"**Predicted Risk Level:** `{pred_label}`")

            st.markdown(f"> {clause}")

            with st.expander("🔍 LIME Explanation + Summary"):
                explanation = explainer.explain_instance(
                    lemmatized, lime_pipeline.predict_proba, num_features=10
                )
                st.markdown("#### 🔍 Top Weighted Features:")
                for word, weight in explanation.as_list():
                     st.markdown(f"- **{word}**: `{round(weight, 3)}`")

                top_words = [term for term, weight in explanation.as_list() if weight > 0][:5]
                simplified = (
                    f"The model focused on these keywords for the risk prediction: **{', '.join(top_words)}**"
                    if top_words else "No strong keywords were found."
                )
                st.info(simplified)

        st.success("✅ Analysis complete! Expand the boxes to explore why each clause received its risk label.")


