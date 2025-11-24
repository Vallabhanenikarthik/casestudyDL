import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# Default paths (change if your files are elsewhere)
DEFAULT_TFIDF_JOBLIB = "./model_output/tfidf_lr.joblib"
DEFAULT_META_JOBLIB = "./model_output/meta_lr.joblib"
DEFAULT_TEST_PRED_CSV = "./model_output/test_predictions.csv"

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📢 Fake News Detector — TF-IDF + Ensemble")

# Utility: load models
@st.cache_resource(show_spinner=False)
def load_models(tfidf_path=DEFAULT_TFIDF_JOBLIB, meta_path=DEFAULT_META_JOBLIB):
    models = {}
    if os.path.exists(tfidf_path):
        obj = joblib.load(tfidf_path)
        models['vect'] = obj['vectorizer']
        models['clf'] = obj['clf']
    else:
        raise FileNotFoundError(f"TF-IDF joblib not found at {tfidf_path}")

    if os.path.exists(meta_path):
        models['meta'] = joblib.load(meta_path)
    else:
        models['meta'] = None

    return models

# Prediction helper (TF-IDF + meta only)
def predict_text(text, models):
    vect = models['vect']; clf = models['clf']; meta = models.get('meta', None)
    # TF-IDF prob
    X_tfidf = vect.transform([text])
    prob_tfidf = clf.predict_proba(X_tfidf)[0,1]

    if meta is not None:
        X_meta = np.array([[prob_tfidf, prob_tfidf]])  # use TF-IDF twice if no BERT
        final_prob = meta.predict_proba(X_meta)[0,1]
    else:
        final_prob = prob_tfidf

    return final_prob, prob_tfidf

# Streamlit layout — two tabs
tab = st.tabs(["Predict", "Inspect"])

with tab[0]:
    st.header("Predict single article")
    st.write("Paste a news article (title + body) below and click Predict.")

    tfidf_path = st.text_input("TF-IDF joblib path", DEFAULT_TFIDF_JOBLIB)
    meta_path = st.text_input("Meta classifier joblib path", DEFAULT_META_JOBLIB)

    text = st.text_area("News text", height=200, key="news_input")

if st.button("Predict"):
    if not text.strip():
        st.error("Please paste some text to predict.")
    else:
        try:
            models = load_models(tfidf_path, meta_path)
        except Exception as e:
            st.error(f"Error loading models: {e}")
        else:
            with st.spinner("Predicting..."):
                final_prob, prob_tfidf = predict_text(text, models)
                label = "FAKE" if final_prob >= 0.5 else "REAL"
                st.metric(label=label, value=f"{final_prob:.4f}")
                st.write(f"TF-IDF prob: {prob_tfidf:.4f}")

with tab[1]:
    st.header("Inspect test predictions & mistakes")
    st.write("Load `test_predictions.csv` (created by the training script) to inspect false positives/negatives.")

    uploaded = st.file_uploader("Or upload a test_predictions.csv file", type=['csv'])
    test_csv_path = DEFAULT_TEST_PRED_CSV
    if uploaded is not None:
        df_test = pd.read_csv(uploaded)
    else:
        if os.path.exists(DEFAULT_TEST_PRED_CSV):
            df_test = pd.read_csv(DEFAULT_TEST_PRED_CSV)
        else:
            df_test = None

    if df_test is None:
        st.warning(f"No test_predictions.csv found at {DEFAULT_TEST_PRED_CSV}. Upload one or check the path.")
    else:
        st.write(f"Loaded {len(df_test)} test rows")
        if 'label' not in df_test.columns or 'pred' not in df_test.columns:
            st.error('test_predictions.csv must contain `label` and `pred` columns')
        else:
            labels = df_test['label'].values
            preds = df_test['pred'].values
            cm = confusion_matrix(labels, preds)
            st.subheader("Confusion Matrix")
            st.write(cm)

            st.subheader("Classification Report")
            cr = classification_report(labels, preds, output_dict=True)
            st.dataframe(pd.DataFrame(cr).transpose())

            # show false positives (pred=1 but label=0) and false negatives (pred=0 but label=1)
            fp = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)]
            fn = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)]

            st.markdown(f"**False positives (pred=1 but label=0):** {len(fp)} rows")
            if len(fp) > 0:
                st.dataframe(fp[['text','label','pred','prob']].head(10))
                if st.button("Save false positives to CSV"):
                    outp = Path("./model_output/false_positives.csv")
                    fp.to_csv(outp, index=False)
                    st.success(f"Saved to {outp}")

            st.markdown(f"**False negatives (pred=0 but label=1):** {len(fn)} rows")
            if len(fn) > 0:
                st.dataframe(fn[['text','label','pred','prob']].head(10))
                if st.button("Save false negatives to CSV"):
                    outn = Path("./model_output/false_negatives.csv")
                    fn.to_csv(outn, index=False)
                    st.success(f"Saved to {outn}")

            # generate a simple HTML report
            if st.button("Generate HTML report"):
                report_html = []
                report_html.append(f"<h1>Fake News Model Report</h1>")
                report_html.append(f"<p>Total test rows: {len(df_test)}</p>")
                report_html.append(f"<h2>Confusion matrix</h2>")
                report_html.append(f"<pre>{cm}</pre>")
                report_html.append(f"<h2>Classification report</h2>")
                report_html.append(df_test[['label','pred','prob']].head(20).to_html(index=False))

                out_file = Path("./model_output/report.html")
                out_file.write_text('\n'.join(report_html), encoding='utf-8')
                st.success(f"Report written to {out_file}")
                st.markdown("[Open report in browser](./model_output/report.html)")

st.sidebar.markdown("---")
st.sidebar.markdown("Paths used by default:")
st.sidebar.text(DEFAULT_TFIDF_JOBLIB)
st.sidebar.text(DEFAULT_META_JOBLIB)
st.sidebar.text(DEFAULT_TEST_PRED_CSV)
