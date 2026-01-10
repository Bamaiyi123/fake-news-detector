import os
# Prevent transformers from importing torch (use TF-only loading)


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
:root{
  --card-bg: rgba(255,255,255,0.06);
  --glass-bg: rgba(255,255,255,0.06);
  --glass-border: rgba(255,255,255,0.08);
  --accent: #4f46e5;
  --muted: rgba(255,255,255,0.65);
}
body { background: linear-gradient(135deg,#0f172a 0%, #001219 100%); color: #e6eef8; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
.app-topbar { display:flex; align-items:center; justify-content:space-between; gap:12px; padding:12px 24px; }
.brand { display:flex; gap:12px; align-items:center; font-weight:700; font-size:20px; color: var(--accent); }
.icon-btn { background: transparent; border: none; color: var(--muted); cursor: pointer; font-size:18px; padding:8px; border-radius:10px; }
.icon-btn:hover { background: rgba(255,255,255,0.03); color: white; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border:1px solid var(--glass-border); border-radius:14px; padding:18px; box-shadow: 0 6px 20px rgba(2,6,23,0.6); }
.kpi { display:flex; gap:14px; align-items:center; }
.kpi-value { font-size:20px; font-weight:700; color: var(--accent); }
.small { color: var(--muted); font-size:13px; }
.token-score { display:inline-block; margin:2px 4px; padding:6px 8px; border-radius:8px; background: rgba(255,255,255,0.02); font-size:12px; }
.stButton > button { background: linear-gradient(135deg, var(--accent), #2a2bd6); color: white; border: none; border-radius: 12px; padding: 12px 24px; font-weight:700; }
.stSidebar { background: linear-gradient(180deg, rgba(79,70,229,0.03), rgba(79,70,229,0.01)); border-right: 2px solid var(--glass-border); }
.grid { display:grid; grid-template-columns: repeat(3, 1fr); gap:16px; }
@media (max-width: 800px) {
  .app-topbar { flex-direction:column; align-items:flex-start; gap:8px; }
  .grid { grid-template-columns: 1fr; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Load model & tokenizer (PyTorch)
# -----------------------------
@st.cache_resource
def load_model(model_name="mrm8488/bert-tiny-finetuned-fake-news-detection"):
     tokenizer = AutoTokenizer.from_pretrained(model_name)
     model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
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
def predict(text, model_arg=None, tokenizer_arg=None, max_length=128, return_attentions=None, return_att=None):
    """
    Backwards-compatible predict wrapper:
    - accepts either (text, model, tokenizer, ...) or (text, return_att=...) calling styles
    - uses module-level `model`/`tokenizer` if none provided
    - accepts both `return_attentions` and legacy `return_att` keywords
    """
    # resolve model/tokenizer
    mdl = model_arg if model_arg is not None else globals().get("model")
    tok = tokenizer_arg if tokenizer_arg is not None else globals().get("tokenizer")
    if mdl is None or tok is None:
        raise RuntimeError("Model/tokenizer not available. Ensure load_model() succeeded.")

    # resolve attention flag (support both names)
    if return_att is None and return_attentions is None:
        ra = False
    elif return_att is None:
        ra = bool(return_attentions)
    else:
        ra = bool(return_att)

    inputs = tok(text, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
    outputs = mdl(**inputs, output_attentions=ra)
    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
    pred = int(torch.argmax(outputs.logits[0]))
    attn = None
    if ra and getattr(outputs, "attentions", None) is not None:
        try:
            attns = outputs.attentions
            agg = None
            for layer in attns:
                arr = layer.cpu().numpy()  # (batch, heads, seq, seq)
                mean = arr.mean(axis=(0,1))  # (seq, seq)
                col = mean[:, 0]
                agg = col if agg is None else agg + col
            scores = (agg / len(attns)).tolist()
            tokens = tok.convert_ids_to_tokens(inputs["input_ids"][0].tolist())
            attn = list(zip(tokens, scores))
        except Exception:
            attn = None
    return pred, float(np.max(probs)), float(probs[0]), float(probs[1]), attn

def infer_label_indices(model, tokenizer):
    """
    Heuristically infer which class index corresponds to 'FAKE' by scoring
    a small calibration set of known fake/real headlines. Stores result in
    session_state as 'label_index_for_fake' (0 or 1).
    """
    # small, obvious calibration examples
    fake_examples = [
        "Breaking: NASA confirms aliens landed in Times Square last night",
        "Miracle pill cures all diseases, scientists stunned",
        "Government to replace currency with pizza next month"
    ]
    real_examples = [
        "Stock market closes up 1.5% on positive earnings reports",
        "City council approves $50 million infrastructure budget",
        "New climate report released by international scientific body"
    ]

    def avg_probs(texts):
        probs_acc = np.zeros(2, dtype=float)
        n = 0
        for t in texts:
            try:
                inputs = tokenizer(t, truncation=True, padding=True, return_tensors="pt", max_length=128)
                outputs = model(**inputs)
                p = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                probs_acc += p
                n += 1
            except Exception:
                continue
        return probs_acc / max(1, n)

    fake_avg = avg_probs(fake_examples)
    real_avg = avg_probs(real_examples)

    # whichever index has higher mean probability on fake examples -> fake index
    fake_index = int(np.argmax(fake_avg))
    # store for later use
    st.session_state["label_index_for_fake"] = fake_index
    st.session_state["label_index_for_real"] = 1 - fake_index
    return fake_index

# replace resolve_label_map/human_label_from_pred behavior with auto-detection
def human_label_from_pred(pred, model, invert=False):
    """
    Determine REAL/FAKE using model.config.id2label if explicit, otherwise
    use an inferred mapping from a small calibration set and cache it.
    """
    # 1) try id2label mapping (explicit textual labels)
    try:
        cfg = getattr(model, "config", None)
        if cfg and getattr(cfg, "id2label", None):
            id2 = cfg.id2label
            v = str(id2.get(pred, "")).lower()
            if "fake" in v or "false" in v or "fraud" in v:
                res = "FAKE"
            elif "real" in v or "true" in v or "legit" in v:
                res = "REAL"
            else:
                # fall back to inference below
                raise ValueError("id2label ambiguous")
            if invert:
                res = "REAL" if res == "FAKE" else "FAKE"
            return res
    except Exception:
        pass

    # 2) use cached inference or run inference once
    if "label_index_for_fake" not in st.session_state:
        try:
            infer_label_indices(model, tokenizer)
        except Exception:
            # last resort numeric default
            res = "REAL" if pred == 1 else "FAKE"
            if invert:
                res = "REAL" if res == "FAKE" else "FAKE"
            return res

    fake_idx = st.session_state.get("label_index_for_fake", 1)
    # map pred -> human label
    res = "FAKE" if pred == fake_idx else "REAL"
    if invert:
        res = "REAL" if res == "FAKE" else "FAKE"
    return res

# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Dashboard", "Single Predict", "Batch Predict", "Fetch & Predict", "About"], index=0)

# -----------------------------
# UI layout
# -----------------------------
st.title("📰 Fake News Detection — Dashboard")
max_len = st.sidebar.slider("Max tokens", 64, 512, 128, step=32)
show_attention = st.sidebar.checkbox("Show token importance", value=True)
invert_labels = st.sidebar.checkbox("Invert label mapping", value=False)

# -----------------------------
# Pages
# -----------------------------
if page == "Dashboard":
    st.header("Dashboard")
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
            label = human_label_from_pred(pred, model, invert=invert_labels)
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

    # Input + optional NewsAPI key (reuses session key if set elsewhere)
    input_text = st.text_area("Paste headline or article:", height=200)
    api_key = st.text_input("NewsAPI key (optional — required for online source lookup)", type="password", key="newsapi_key")

    # Lookup controls
    lookup = st.checkbox("Lookup source online (NewsAPI)", value=False)
    source_info = None
    if lookup:
        st.markdown("Use NewsAPI to find the likely source for this headline.")
        if st.button("Find source", key="find_source"):
            if not api_key:
                st.warning("Enter a NewsAPI key to enable online lookup (get free key at https://newsapi.org).")
            else:
                try:
                    with st.spinner("Searching for source..."):
                        params = {
                            "qInTitle": input_text,
                            "apiKey": api_key,
                            "pageSize": 1,
                            "sortBy": "relevancy",
                        }
                        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
                        resp.raise_for_status()
                        data = resp.json()
                        articles = data.get("articles", [])
                        if articles:
                            a = articles[0]
                            source_info = {
                                "source": a.get("source", {}).get("name", ""),
                                "url": a.get("url", ""),
                                "publishedAt": a.get("publishedAt", "")
                            }
                            st.success(f"Found source: {source_info['source']}")
                            st.write(f"[Open article]({source_info['url']})")
                        else:
                            st.info("No matching article found for that headline.")
                except Exception as e:
                    st.error(f"Source lookup failed: {e}")

    # Analyze / Predict
    if st.button("Analyze"):
        if input_text.strip() == "":
            st.warning("Enter text first.")
        else:
            with st.spinner("Predicting..."):
                pred, conf, p_fake, p_real, att = predict(input_text, return_att=show_attention)

            label = human_label_from_pred(pred, model, invert=invert_labels)
            st.write(f"raw probs: index0={p_fake:.3f}, index1={p_real:.3f}")
            st.success(f"{label} — confidence {conf:.2%}")

            # show detected source if available
            if source_info:
                st.info(f"Detected source: **{source_info['source']}** — [Open article]({source_info['url']})")
                if source_info.get("publishedAt"):
                    st.caption(f"Published at: {source_info['publishedAt']}")

            st.metric("P(Real)", f"{p_real:.2%}")
            st.metric("P(Fake)", f"{p_fake:.2%}")

            st.session_state["last_result"] = {
                "text": input_text,
                "prediction": label,
                "confidence": conf,
                "att": att,
                "source": source_info
            }

            if show_attention and att:
                st.write("Token importance (top tokens):")
                df_att = pd.DataFrame(att, columns=["token", "score"]).head(60)
                st.table(df_att)

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
                    mapped = human_label_from_pred(pred, model, invert=invert_labels)
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
    st.write("Fetch real news headlines from the web and predict if they are fake or real.")
    
    # NewsAPI key input
    api_key = st.text_input("NewsAPI key (get free key at https://newsapi.org)", type="password", key="newsapi_key")
    
    col1, col2 = st.columns(2)
    with col1:
        query = st.text_input("Search query (e.g., 'technology', 'politics')", value="technology")
    with col2:
        num_articles = st.slider("Number of articles to fetch", 1, 50, 10)
    
    if st.button("Fetch & Analyze"):
        if not api_key:
            st.error("Please enter a NewsAPI key. Get one free at https://newsapi.org")
        else:
            try:
                with st.spinner("Fetching articles from NewsAPI..."):
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        "q": query,
                        "apiKey": api_key,
                        "pageSize": num_articles,
                        "sortBy": "publishedAt"
                    }
                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()
                    data = response.json()
                    
                    if data.get("status") != "ok":
                        st.error(f"API Error: {data.get('message', 'Unknown error')}")
                    else:
                        articles = data.get("articles", [])
                        st.success(f"Fetched {len(articles)} articles. Analyzing...")
                        
                        results = []
                        progress = st.progress(0)
                        
                        for i, article in enumerate(articles):
                            title = article.get("title", "")
                            description = article.get("description", "") or ""
                            source = article.get("source", {}).get("name", "Unknown")
                            url_article = article.get("url", "")
                            
                            # Predict on title + description
                            text_to_predict = f"{title}. {description}"
                            if text_to_predict.strip():
                                pred, conf, p_fake, p_real, _ = predict(text_to_predict, return_att=False)
                                label = human_label_from_pred(pred, model, invert=invert_labels)
                                results.append({
                                    "title": title,
                                    "source": source,
                                    "prediction": label,
                                    "confidence": conf,
                                    "p_fake": p_fake,
                                    "p_real": p_real,
                                    "url": url_article
                                })
                            progress.progress((i + 1) / len(articles))
                        
                        # Display results
                        df = pd.DataFrame(results)
                        st.subheader(f"Results ({len(results)} articles)")
                        
                        # Color-code by prediction
                        def highlight_prediction(row):
                            if row["prediction"] == "FAKE":
                                return ["background-color: #ff6b6b"] * len(row)
                            else:
                                return ["background-color: #51cf66"] * len(row)
                        
                        st.dataframe(
                            df[["title", "source", "prediction", "confidence"]].style.apply(highlight_prediction, axis=1),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results CSV", data=csv_bytes, file_name="web_predictions.csv", mime="text/csv")
                        
                        # Show article links
                        st.subheader("Articles")
                        for idx, row in df.iterrows():
                            emoji = "🔴" if row["prediction"] == "FAKE" else "🟢" 
                            st.write(f"{emoji} [{row['title']}]({row['url']}) — {row['source']}")
                            st.caption(f"Confidence: {row['confidence']:.2%}")
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to fetch articles: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:  # About
    st.header("About")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    **Fake News Detector** — dashboard UI built with Streamlit.
    - Uses BERT fine-tuned (mrm8488/bert-tiny-finetuned-fake-news-detection) for classification (TF).
     - Sidebar navigation, top taskbar, glass cards, icon buttons.
     - Single and batch prediction pages.
     """)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Built with ❤️ — Streamlit + Transformers. Ensure Streamlit runs in same Python env as installed packages. adeyi bamaiyi. thanks mr steve, thanks torbita. love u guys  ")
