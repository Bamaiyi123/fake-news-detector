import os
# Prevent transformers from importing torch (use TF-only loading)
os.environ.setdefault("HF_HUB_REQUEST_TIMEOUT", "120")

import io
import shutil
import streamlit as st
import torch
import numpy as np
import pandas as pd
import requests

from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# Styling (purple/blue glass theme)
# -----------------------------
st.set_page_config(page_title="📰 Fake News Dashboard", layout="wide", page_icon="🧠")

CSS = """
<style>
:root {
  --primary: #ff6b35;
  --primary-dark: #e55100;
  --primary-light: #ff8c5a;
  --bg-dark: #0a0e27;
  --bg-darker: #050812;
  --card-light: rgba(255, 107, 53, 0.08);
  --card-border: rgba(255, 107, 53, 0.15);
  --text-primary: #ffffff;
  --text-secondary: #b0bec5;
  --success: #10b981;
  --danger: #ef4444;
  --warning: #f59e0b;
  --accent-glow: rgba(255, 107, 53, 0.3);
}

* {
  transition: all 0.2s ease;
}

html, body {
  background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 50%, #1a1f3a 100%);
  color: var(--text-primary);
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Streamlit defaults override */
.stApp {
  background: linear-gradient(135deg, var(--bg-darker) 0%, var(--bg-dark) 50%, #1a1f3a 100%);
}

.stMarkdown, .stText, p {
  color: var(--text-primary) !important;
}

/* Topbar styling */
.app-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 28px;
  background: linear-gradient(90deg, rgba(255, 107, 53, 0.05), rgba(255, 107, 53, 0.02));
  border-bottom: 2px solid var(--card-border);
  border-radius: 0 0 16px 16px;
  backdrop-filter: blur(12px);
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(255, 107, 53, 0.1);
}

.brand {
  display: flex;
  gap: 10px;
  align-items: center;
  font-weight: 800;
  font-size: 24px;
  background: linear-gradient(135deg, var(--primary), var(--primary-light));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 0 20px rgba(255, 107, 53, 0.3);
}

/* Icon buttons */
.icon-btn {
  background: rgba(255, 107, 53, 0.08);
  border: 1.5px solid var(--card-border);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 18px;
  padding: 10px 14px;
  border-radius: 12px;
  font-weight: 600;
}

.icon-btn:hover {
  background: rgba(255, 107, 53, 0.15);
  border-color: var(--primary);
  color: var(--primary);
  box-shadow: 0 0 15px var(--accent-glow);
  transform: translateY(-2px);
}

.icon-btn:active {
  transform: translateY(0);
}

/* Card styling (glassmorphism) */
.card {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.06), rgba(255, 107, 53, 0.02));
  border: 1.5px solid var(--card-border);
  border-radius: 18px;
  padding: 24px;
  box-shadow: 
    0 8px 32px rgba(255, 107, 53, 0.12),
    inset 0 1px 2px rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.card:hover {
  border-color: var(--primary);
  box-shadow: 
    0 16px 48px rgba(255, 107, 53, 0.18),
    inset 0 1px 2px rgba(255, 107, 53, 0.08);
  transform: translateY(-4px);
}

/* Grid layout */
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
  margin-bottom: 24px;
}

/* KPI styling */
.kpi {
  display: flex;
  gap: 16px;
  align-items: center;
  padding: 12px;
  background: rgba(255, 107, 53, 0.04);
  border-radius: 12px;
  border-left: 4px solid var(--primary);
}

.kpi-value {
  font-size: 24px;
  font-weight: 800;
  color: var(--primary);
  text-shadow: 0 0 10px rgba(255, 107, 53, 0.2);
}

.kpi-label {
  font-size: 13px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 600;
}

/* Token score badges */
.token-score {
  display: inline-block;
  margin: 4px 6px;
  padding: 8px 12px;
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.1), rgba(255, 107, 53, 0.05));
  border: 1px solid rgba(255, 107, 53, 0.2);
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  cursor: default;
  transition: all 0.2s ease;
}

.token-score:hover {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.2), rgba(255, 107, 53, 0.1));
  border-color: var(--primary);
  color: var(--primary);
  box-shadow: 0 0 10px rgba(255, 107, 53, 0.15);
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--primary), var(--primary-dark));
  color: white;
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  font-weight: 700;
  font-size: 14px;
  box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3);
  transition: all 0.3s ease;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stButton > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
  background: linear-gradient(135deg, var(--primary-light), var(--primary));
}

.stButton > button:active {
  transform: translateY(0);
}

/* Input fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
  background: rgba(255, 107, 53, 0.05) !important;
  border: 1.5px solid var(--card-border) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
  padding: 12px 16px !important;
  font-weight: 500;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stSelectbox > div > div > select:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 15px rgba(255, 107, 53, 0.2) !important;
  background: rgba(255, 107, 53, 0.08) !important;
}

/* Sidebar */
.stSidebar {
  background: linear-gradient(180deg, rgba(255, 107, 53, 0.04), rgba(255, 107, 53, 0.01));
  border-right: 2px solid var(--card-border);
}

.stSidebar [data-testid="stSidebarContent"] {
  background: transparent;
}

/* Sidebar title & radio buttons */
.stSidebar .stMarkdown h1 {
  color: var(--primary);
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.stRadio > div {
  gap: 8px;
}

.stRadio > div > label > div {
  background: rgba(255, 107, 53, 0.08);
  border: 1.5px solid var(--card-border);
  border-radius: 10px;
  padding: 10px 16px;
  font-weight: 600;
  transition: all 0.2s ease;
}

.stRadio > div > label > div:hover {
  background: rgba(255, 107, 53, 0.12);
  border-color: var(--primary);
  color: var(--primary);
}

/* Checkbox */
.stCheckbox > label {
  color: var(--text-primary);
  font-weight: 500;
}

.stCheckbox > label > span:first-child {
  background: rgba(255, 107, 53, 0.1);
  border: 1.5px solid var(--card-border);
  border-radius: 6px;
}

/* Success/Error/Warning messages */
.stSuccess {
  background: rgba(16, 185, 129, 0.1);
  border: 1.5px solid var(--success);
  border-radius: 12px;
  padding: 16px;
  color: var(--success);
}

.stError {
  background: rgba(239, 68, 68, 0.1);
  border: 1.5px solid var(--danger);
  border-radius: 12px;
  padding: 16px;
  color: var(--danger);
}

.stWarning {
  background: rgba(245, 158, 11, 0.1);
  border: 1.5px solid var(--warning);
  border-radius: 12px;
  padding: 16px;
  color: var(--warning);
}

.stInfo {
  background: rgba(255, 107, 53, 0.08);
  border: 1.5px solid var(--primary);
  border-radius: 12px;
  padding: 16px;
  color: var(--text-primary);
}

/* Dataframe styling */
.stDataFrame {
  border-radius: 12px;
  overflow: hidden;
  border: 1.5px solid var(--card-border);
}

.stDataFrame thead {
  background: linear-gradient(90deg, rgba(255, 107, 53, 0.1), rgba(255, 107, 53, 0.05));
}

/* Slider */
.stSlider > div > div > div > div {
  background: linear-gradient(90deg, var(--primary), var(--primary-dark));
  border-radius: 10px;
}

/* Metric boxes */
.stMetric {
  background: linear-gradient(135deg, rgba(255, 107, 53, 0.08), rgba(255, 107, 53, 0.04));
  border: 1.5px solid var(--card-border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 4px 15px rgba(255, 107, 53, 0.08);
}

.stMetric [data-testid="metricDeltaContainer"] {
  color: var(--primary);
  font-weight: 700;
}

/* Expander */
.streamlit-expanderHeader {
  background: rgba(255, 107, 53, 0.08);
  border: 1.5px solid var(--card-border);
  border-radius: 12px;
  color: var(--text-primary);
  font-weight: 700;
}

.streamlit-expanderHeader:hover {
  background: rgba(255, 107, 53, 0.12);
  border-color: var(--primary);
}

/* Divider */
hr {
  border-color: var(--card-border);
  margin: 24px 0;
}

/* Caption text */
.stCaption {
  color: var(--text-secondary);
  font-size: 12px;
  font-weight: 500;
}

/* Subheader */
h2, h3 {
  color: var(--text-primary);
  font-weight: 800;
  text-transform: capitalize;
}

h2 {
  border-bottom: 3px solid var(--primary);
  padding-bottom: 8px;
  margin-bottom: 20px;
}

/* Responsive design */
@media (max-width: 768px) {
  .app-topbar {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
  }

  .grid {
    grid-template-columns: 1fr;
  }

  .stButton > button {
    width: 100%;
  }

  .brand {
    font-size: 18px;
  }
}

/* Scrollbar styling */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: rgba(255, 107, 53, 0.05);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, var(--primary), var(--primary-dark));
  border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
  background: linear-gradient(180deg, var(--primary-light), var(--primary));
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Load model & tokenizer (TF / default)
# -----------------------------
@st.cache_resource
def load_model(model_name="mrm8488/bert-tiny-finetuned-fake-news-detection"):
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, from_pt=True)
     model.eval()
     return model, tokenizer

try:
    model, tokenizer = load_model()
except Exception as e:
    st.title("📰 Fake News Detector")
    st.error("Failed to load model/tokenizer. See error below:")
    st.code(repr(e))
    st.stop()

# -----------------------------
# Prediction & helpers
# -----------------------------
def predict(text, model, tokenizer, max_length=128, return_attentions=False):
    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="tf", max_length=max_length)
    outputs = model(**inputs, output_attentions=return_attentions, training=False)
    logits = outputs.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred = int(np.argmax(probs))
    attn = None
    if return_attentions and getattr(outputs, "attentions", None) is not None:
        try:
            attns = outputs.attentions
            agg = None
            for layer in attns:
                arr = np.array(layer)  # (batch, heads, seq, seq)
                mean = arr.mean(axis=(0,1))  # (seq, seq)
                col = mean[:, 0]
                agg = col if agg is None else agg + col
            scores = (agg / len(attns)).tolist()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].numpy()[0])
            attn = list(zip(tokens, scores))
        except Exception:
            attn = None
    return pred, float(np.max(probs)), float(probs[0]), float(probs[1]), attn

def human_label_from_pred(pred, invert=False):
    # default numeric mapping: index 1 -> REAL, index 0 -> FAKE
    res = "REAL" if pred == 1 else "FAKE"
    if invert:
        res = "REAL" if res == "FAKE" else "FAKE"
    return res

# -----------------------------
# UI layout
# -----------------------------
st.title("📰 Fake News Detection — Dashboard")
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Dashboard", "Single Predict", "Batch Predict", "Fetch & Predict", "About"])
st.sidebar.markdown("---")
max_len = st.sidebar.slider("Max tokens", 64, 512, 128, step=32)
show_attention = st.sidebar.checkbox("Show token importance", value=True)
invert_labels = st.sidebar.checkbox("Invert label mapping", value=False)

# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    st.header("Dashboard")
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Model**  \n`bert-base-uncased`  \n\n<span class='small'>Backend: TensorFlow</span>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Examples**")
    examples = [
        "Breaking: Celebrity endorses miracle cure — doctors shocked",
        "Government announces new infrastructure spending plan",
        "Study shows chocolate linked with longer life"
    ]
    for ex in examples:
        if st.button(f"🔎 {ex}", key=f"ex_{ex[:12]}"):
            st.session_state["example_text"] = ex
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    last = st.session_state.get("last_result", None)
    if last:
        st.markdown("**Last prediction**")
        st.write(last["text"])
        st.info(f'{last["prediction"]}  — confidence {last["confidence"]:.2%}')
    else:
        st.markdown("**Last prediction**")
        st.write("_No predictions yet_")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    colA, colB = st.columns([2,1])
    with colA:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        example_val = st.session_state.get("example_text", "")
        txt = st.text_area("Enter headline or article:", value=example_val, height=160, key="dash_input")
        if st.button("Analyze (Dashboard)"):
            pred, conf, p0, p1, att = predict(txt, model, tokenizer, max_length=max_len, return_attentions=show_attention)
            label = human_label_from_pred(pred, invert=invert_labels)
            st.write(f"raw probs: index0={p0:.3f}, index1={p1:.3f}")
            st.success(f"{label}  — confidence {conf:.2%}")
            st.session_state["last_result"] = {"text": txt, "prediction": label, "confidence": conf, "att": att}
            if att and show_attention:
                st.write("Token importance:")
                for t,s in att[:60]:
                    st.markdown(f"<span class='token-score' title='{s:.4f}'>{t}</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with colB:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Quick actions")
        if st.button("Analyze example 1"):
            st.session_state["example_text"] = examples[0]
        st.markdown("Upload CSV for batch predictions in the Batch page.")
        st.markdown('</div>', unsafe_allow_html=True)

elif page == "Single Predict":
    st.header("Single Prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    input_text = st.text_area("Paste headline or article:", height=200)
    api_key = st.text_input("NewsAPI key (optional)", type="password", key="newsapi_key")
    lookup = st.checkbox("Lookup source online (NewsAPI)", value=False)
    source_info = None
    if lookup and st.button("Find source", key="find_source"):
        if not api_key:
            st.warning("Enter a NewsAPI key.")
        else:
            try:
                params = {"qInTitle": input_text, "apiKey": api_key, "pageSize": 1, "sortBy": "relevancy"}
                resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10); resp.raise_for_status()
                data = resp.json(); articles = data.get("articles", [])
                if articles:
                    a = articles[0]
                    source_info = {"source": a.get("source", {}).get("name", ""), "url": a.get("url", ""), "publishedAt": a.get("publishedAt", "")}
                    st.success(f"Found source: {source_info['source']}")
                    st.write(f"[Open article]({source_info['url']})")
                else:
                    st.info("No matching article found.")
            except Exception as e:
                st.error(f"Source lookup failed: {e}")

    if st.button("Analyze"):
        if input_text.strip() == "":
            st.warning("Enter text first.")
        else:
            pred, conf, p0, p1, att = predict(input_text, model, tokenizer, max_length=max_len, return_attentions=show_attention)
            label = human_label_from_pred(pred, invert=invert_labels)
            st.write(f"raw probs: index0={p0:.3f}, index1={p1:.3f}")
            st.success(f"{label} — confidence {conf:.2%}")
            if source_info:
                st.info(f"Detected source: **{source_info['source']}** — [Open article]({source_info['url']})")
                if source_info.get("publishedAt"): st.caption(f"Published at: {source_info['publishedAt']}")
            st.metric("P(Real)", f"{p1:.2%}"); st.metric("P(Fake)", f"{p0:.2%}")
            st.session_state["last_result"] = {"text": input_text, "prediction": label, "confidence": conf, "att": att, "source": source_info}
            if show_attention and att:
                df_att = pd.DataFrame(att, columns=["token","score"]).head(60); st.table(df_att)
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Batch Predict":
    st.header("Batch Prediction")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV or TXT (one text per line)", type=["csv","txt"])
    if uploaded is not None:
        raw = uploaded.getvalue()
        try:
            decoded = raw.decode("utf-8", errors="replace")
            lines = [l.strip() for l in decoded.splitlines() if l.strip()]
            st.info(f"Parsed {len(lines)} records.")
            if st.button("Run batch"):
                results = []
                progress = st.progress(0)
                for i, txt in enumerate(lines):
                    pred, conf, p0, p1, _ = predict(txt, model, tokenizer, max_length=max_len, return_attentions=False)
                    mapped = human_label_from_pred(pred, invert=invert_labels)
                    results.append({"text": txt, "prediction": mapped, "confidence": conf, "p_fake": p0, "p_real": p1})
                    progress.progress((i+1)/len(lines))
                df = pd.DataFrame(results); st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8"); st.download_button("Download CSV", data=csv_bytes, file_name="batch_predictions.csv", mime="text/csv")
                if len(results)>0: st.session_state["last_result"] = results[0]
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

elif page == "Fetch & Predict":
    st.header("Fetch & Predict from Web")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    api_key = st.text_input("NewsAPI key (get free key at https://newsapi.org)", type="password", key="newsapi_key")
    col1, col2 = st.columns(2)
    with col1: query = st.text_input("Search query (e.g., 'technology')", value="technology")
    with col2: num_articles = st.slider("Number of articles", 1, 50, 10)
    if st.button("Fetch & Analyze"):
        if not api_key: st.error("Please enter a NewsAPI key")
        else:
            try:
                with st.spinner("Fetching..."):
                    url = "https://newsapi.org/v2/everything"
                    params = {"q": query, "apiKey": api_key, "pageSize": num_articles, "sortBy": "publishedAt"}
                    response = requests.get(url, params=params, timeout=10); response.raise_for_status(); data = response.json()
                    if data.get("status") != "ok": st.error(f"API Error: {data.get('message', 'Unknown')}")
                    else:
                        articles = data.get("articles", []); st.success(f"Fetched {len(articles)} articles. Analyzing...")
                        results = []; progress = st.progress(0)
                        for i, article in enumerate(articles):
                            title = article.get("title",""); desc = article.get("description","") or ""; src = article.get("source",{}).get("name","Unknown"); url_article = article.get("url","")
                            text_to_predict = f"{title}. {desc}"
                            if text_to_predict.strip():
                                pred, conf, p0, p1, _ = predict(text_to_predict, model, tokenizer, max_length=max_len, return_attentions=False)
                                label = human_label_from_pred(pred, invert=invert_labels)
                                results.append({"title": title, "source": src, "prediction": label, "confidence": conf, "p_fake": p0, "p_real": p1, "url": url_article})
                            progress.progress((i+1)/len(articles))
                        df = pd.DataFrame(results)
                        def highlight_prediction(r): return ["background-color: #ff6b6b"]*len(r) if r["prediction"]=="FAKE" else ["background-color: #51cf66"]*len(r)
                        st.dataframe(df[["title","source","prediction","confidence"]].style.apply(highlight_prediction, axis=1), use_container_width=True)
                        csv_bytes = df.to_csv(index=False).encode("utf-8"); st.download_button("Download results CSV", data=csv_bytes, file_name="web_predictions.csv", mime="text/csv")
                        st.subheader("Articles")
                        for idx, row in df.iterrows(): emoji = "🔴" if row["prediction"]=="FAKE" else "🟢"; st.write(f"{emoji} [{row['title']}]({row['url']}) — {row['source']}"); st.caption(f"Confidence: {row['confidence']:.2%}")
            except requests.exceptions.RequestException as e: st.error(f"Failed to fetch articles: {e}")
            except Exception as e: st.error(f"Error: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.header("About")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    **Fake News Detector** — dashboard UI built with Streamlit.
    - Uses BERT (`bert-base-uncased`) for classification (TF).
    - Sidebar navigation, top taskbar, glass cards, icon buttons.
    - Single and batch prediction pages.
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ — Streamlit + Transformers. Ensure Streamlit runs in same Python env as installed packages.")

