# streamlit_app_beautiful.py
import streamlit as st
import joblib
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os

# ----------------------
# Helper functions
# ----------------------
def load_joblib(path_or_buffer):
    # path_or_buffer may be a file-like or path
    if hasattr(path_or_buffer, "read"):
        # file-like from uploader
        # joblib.load accepts file-like; but to be safe write to BytesIO
        return joblib.load(path_or_buffer)
    return joblib.load(path_or_buffer)

def make_prediction_text(text, tfidf, model, top_k=5):
    X = tfidf.transform([text])
    # prefer predict_proba if available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = getattr(model, "classes_", None)
        if classes is None:
            pred = model.predict(X)[0]
            return pred, [(pred, 1.0)]
        idx_sorted = np.argsort(probs)[::-1]
        results = [(classes[i], float(probs[i])) for i in idx_sorted[:top_k]]
        return results[0][0], results
    else:
        pred = model.predict(X)[0]
        return pred, [(pred, 1.0)]

def df_to_download_link(df, filename="predictions.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"data:file/csv;base64,{b64}"

def plot_probabilities(topk):
    labels = [t[0] for t in topk]
    probs = [t[1] for t in topk]
    fig, ax = plt.subplots(figsize=(6,2.5))
    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, probs)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0,1)
    ax.set_xlabel("Probability")
    plt.tight_layout()
    return fig

# ----------------------
# Page layout & CSS
# ----------------------
st.set_page_config(page_title="🎬 Video Category Classifier", page_icon="🎯", layout="wide")

# Custom CSS + HTML header
st.markdown(
    """
    <style>
    :root{
      --bg1: #0f172a;
      --bg2: #071029;
      --card: rgba(255,255,255,0.03);
      --muted: #94a3b8;
      --accent1: #7c3aed;
      --accent2: #06b6d4;
      --glass-border: rgba(255,255,255,0.04);
    }
    html, body, [class*="css"]  {
      background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%) !important;
      color: #e6eef8;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .top-container {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .app-card {
      background: var(--card);
      border: 1px solid var(--glass-border);
      padding: 1.25rem;
      border-radius: 14px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.6);
    }
    .brand {
      display:flex;
      gap:12px;
      align-items:center;
    }
    .logo {
      width:64px;
      height:64px;
      border-radius:12px;
      background: linear-gradient(135deg,var(--accent1), var(--accent2));
      display:flex;
      align-items:center;
      justify-content:center;
      font-weight:700;
      color:white;
      font-size:22px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.6);
    }
    .title {
      font-size:20px;
      font-weight:700;
      margin-bottom:2px;
    }
    .subtitle {
      color: var(--muted);
      font-size:13px;
      margin-top:0;
    }
    .pill {
      background: linear-gradient(90deg,var(--accent1), var(--accent2));
      color:white;
      padding:6px 12px;
      border-radius:999px;
      font-weight:600;
      box-shadow: 0 6px 20px rgba(124,58,237,0.18);
      font-size:13px;
    }
    .example-btn {
      border-radius:10px;
      padding:6px 10px;
      border:1px solid rgba(255,255,255,0.04);
      background: transparent;
      color: #e6eef8;
    }
    .small-muted { color: var(--muted); font-size:13px; }
    .footer {
      color: var(--muted);
      font-size:13px;
      margin-top:8px;
    }
    /* responsive tweaks */
    @media (max-width: 900px) {
      .top-container { flex-direction: column; align-items:flex-start; gap:8px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# header
st.markdown(
    f"""
    <div class="top-container">
      <div class="brand app-card" style="flex:1">
        <div style="display:flex; gap:16px; align-items:center;">
          <div class="logo">YT</div>
          <div>
            <div class="title">🎬 Trending Video Classifier</div>
            <div class="subtitle">Paste title / tags / description or upload CSV. Beautiful UI + strong UX.</div>
          </div>
        </div>
      </div>
      <div style="width:320px" class="app-card">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <div>
            <div style="font-weight:700">Quick Actions</div>
            <div class="small-muted">Load models & run predictions</div>
          </div>
          <div class="pill">Streamlit • Joblib</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# main layout columns
col_main, col_right = st.columns([2.2, 1])

# ----------------------
# Left: Inputs and Predictions
# ----------------------
with col_main:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### 🔧 Load models (tfidf + classifier)")
    tfidf_file = st.file_uploader("Upload TF-IDF (joblib / pkl)", type=["pkl","joblib"], key="tfidf_file")
    model_file = st.file_uploader("Upload Category Predictor (joblib / pkl)", type=["pkl","joblib"], key="model_file")
    col_a, col_b = st.columns([1,1])
    with col_a:
        use_local_btn = st.checkbox("Auto-load from local files (tfidf_joblib.pkl & category_predictor_joblib.pkl)", value=True)
    with col_b:
        load_button = st.button("Load models")

    # attempt to load into session_state
    if "models_loaded" not in st.session_state:
        st.session_state["models_loaded"] = False

    if load_button:
        try:
            if tfidf_file is not None and model_file is not None:
                tfidf = load_joblib(tfidf_file)
                model = load_joblib(model_file)
            elif use_local_btn:
                tfidf = load_joblib("tfidf_joblib.pkl")
                model = load_joblib("category_predictor_joblib.pkl")
            else:
                st.error("Please upload both files or enable auto-load from local files.")
                tfidf = model = None

            if tfidf is not None and model is not None:
                st.session_state["tfidf"] = tfidf
                st.session_state["model"] = model
                st.session_state["models_loaded"] = True
                st.success("✅ Models loaded successfully")
        except Exception as e:
            st.session_state["models_loaded"] = False
            st.error(f"Failed to load models: {e}")

    # Single prediction card
    st.markdown("### ✍️ Single Prediction")
    with st.form(key="single_pred_form"):
        tcol1, tcol2 = st.columns([3,1])
        with tcol1:
            title = st.text_input("Video Title", placeholder="e.g., Messi vs Brazil Full Match")
            tags = st.text_input("Tags (comma-separated)", placeholder="football, highlights")
            desc = st.text_area("Description (optional)", height=120, placeholder="Enter video description or summary...")
        with tcol2:
            st.markdown("**Examples**")
            if st.form_submit_button("Example: Messi"):
                title = "Messi vs Brazil Full Match"
                tags = "football,messi,highlights"
                desc = "Full match highlights and commentary."
            if st.form_submit_button("Example: Phone Review"):
                title = "Samsung S24 Ultra Review"
                tags = "tech,review,samsung"
                desc = "Camera, battery and display testing and comparisons."

        submit_single = st.form_submit_button("Predict Category")

    if submit_single:
        if not st.session_state.get("models_loaded", False):
            # try auto load if available
            try:
                tfidf = load_joblib("tfidf_joblib.pkl")
                model = load_joblib("category_predictor_joblib.pkl")
                st.session_state["tfidf"] = tfidf
                st.session_state["model"] = model
                st.session_state["models_loaded"] = True
                st.success("Auto-loaded local models.")
            except Exception:
                st.error("Models are not loaded. Upload them or place 'tfidf_joblib.pkl' and 'category_predictor_joblib.pkl' in this folder.")
        if st.session_state.get("models_loaded", False):
            combined = " ".join([title or "", tags or "", desc or ""])
            try:
                pred, topk = make_prediction_text(combined, st.session_state["tfidf"], st.session_state["model"], top_k=6)
                st.markdown(f"**Predicted Category:** <span style='color:#fef08a;font-weight:700;font-size:18px'>{pred}</span>", unsafe_allow_html=True)
                st.markdown("**Top probabilities:**")
                for lab, p in topk:
                    st.markdown(f"- {lab}: `{p:.3f}`")
                # show probability bar chart
                fig = plot_probabilities(topk[:6])
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

    # Batch prediction card
    st.markdown('<div class="app-card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown("### 📥 Batch Prediction from CSV")
    st.markdown('<div class="small-muted">CSV must contain columns: <code>title</code>, <code>tags</code> (optional), <code>description</code> (optional).</div>', unsafe_allow_html=True)
    csv_u = st.file_uploader("Upload CSV for batch predictions", type=["csv"], key="csv_uploader")
    if csv_u is not None:
        try:
            df = pd.read_csv(csv_u)
            st.markdown("**Preview**")
            st.dataframe(df.head(6))
            run_batch = st.button("Run batch prediction on uploaded CSV")
            if run_batch:
                if not st.session_state.get("models_loaded", False):
                    try:
                        tfidf = load_joblib("tfidf_joblib.pkl")
                        model = load_joblib("category_predictor_joblib.pkl")
                        st.session_state["tfidf"] = tfidf
                        st.session_state["model"] = model
                        st.session_state["models_loaded"] = True
                        st.success("Auto-loaded local models.")
                    except Exception:
                        st.error("Models not loaded. Upload them above or place pickles in the same folder.")
                if st.session_state.get("models_loaded", False):
                    df['title'] = df.get('title','').fillna('')
                    df['tags'] = df.get('tags','').fillna('')
                    df['description'] = df.get('description','').fillna('')
                    df['combined'] = (df['title'] + " " + df['tags'] + " " + df['description']).str.strip()
                    X = st.session_state['tfidf'].transform(df['combined'].tolist())
                    try:
                        preds = st.session_state['model'].predict(X)
                    except Exception:
                        # fallback row by row
                        preds = []
                        for i in range(X.shape[0]):
                            preds.append(st.session_state['model'].predict(X[i]))
                    df['predicted_category'] = preds
                    st.success("Batch prediction complete ✅")
                    st.dataframe(df.head(10))
                    link = df_to_download_link(df)
                    st.markdown(f"[⬇️ Download predictions](%s)" % link, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Right: Info, Stats & Tips
# ----------------------
with col_right:
    st.markdown('<div class="app-card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ Info & Tips")
    st.markdown("- **Model:** Upload `joblib`/`pkl` files exported from scikit-learn.")
    st.markdown("- **TF-IDF:** app expects a fitted TF-IDF vectorizer.")
    st.markdown("- **CSV columns:** `title`, `tags`, `description` (any missing columns will be handled).")
    st.markdown("---")
    st.markdown("### ⚙️ Quick troubleshooting")
    st.markdown("""
    - If you get `ImportError: numpy._core` or similar, install a matching numpy version:
      ```bash
      pip install --upgrade numpy
      ```
    - If `joblib.load` fails due to different sklearn versions, try recreating the model with your current scikit-learn.
    """)
    st.markdown("---")
    st.markdown("### 🔎 Small demo")
    st.markdown("Use this short sample text to try predictions quickly:")
    st.code("Title: Top 10 Messi Goals\nTags: messi,football,highlights\nDescription: We compiled Messi's best goals in 2024", language="text")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
st.markdown(
    """
    <div style="margin-top:14px;display:flex;justify-content:space-between;align-items:center">
      <div class="small-muted">Built with ❤️ • Paste, Upload, Predict</div>
      <div class="small-muted">Place `tfidf_joblib.pkl` & `category_predictor_joblib.pkl` in the app folder to auto-load</div>
    </div>
    """,
    unsafe_allow_html=True,
)
