# streamlit_app.py
# -*- coding: utf-8 -*-
"""
ASD Speech → STT → Intent Dashboard (Streamlit)
- Demo-friendly: works with your CSV or generates synthetic data
- Panels: Overview, Intent Analysis, Session Tracking, Multimodal Browser, Model Metrics

CSV schema (columns; extra columns ignored):
- child_id (str/int)
- session_id (str/int)
- timestamp (ISO8601 or yyyy-mm-dd HH:MM:SS)
- audio_uri (local path or http/https URL)
- stt_text (str)
- intent_pred (str)
- confidence (float 0~1)
- intent_true (str, optional)
- sentiment (one of {"pos","neu","neg"}, optional)
- duration_sec (float, optional)
- keywords ("word1, word2, ...", optional)
"""

import io
import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
try:
    from wordcloud import WordCloud
    _HAS_WC = True
except Exception:
    _HAS_WC = False

# -------------------------------
# 0) Page config
# -------------------------------
st.set_page_config(
    page_title="ASD Intent Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("ASD Speech → Intent Analysis Dashboard")
st.caption("Interactive dashboard to explore ASD children's utterances, intents, and model performance.")

# -------------------------------
# 1) Data loading / demo generator
# -------------------------------
INTENTS = ["request", "question", "affirmation", "rejection", "greeting", "emotion", "other"]
SENTI = ["pos", "neu", "neg"]
# Consistent intent colors across charts
INTENT_COLORS = {
    "request": "#4E79A7",
    "question": "#59A14F",
    "affirmation": "#9C755F",
    "rejection": "#E15759",
    "greeting": "#F28E2B",
    "emotion": "#B07AA1",
    "other": "#76B7B2",
}

@st.cache_data(show_spinner=False)
def generate_demo(n_children: int = 5, days: int = 14, rows_per_day: int = 60, with_truth: bool = True):
    rng = np.random.default_rng(42)
    start = datetime.now() - timedelta(days=days)
    rows = []
    for child in range(1, n_children + 1):
        for d in range(days):
            day = start + timedelta(days=d)
            sessions_today = rng.integers(1, 4)  # 1~3 sessions/day
            for s in range(sessions_today):
                session_id = f"C{child}-S{d}-{s}"
                n_utts = max(5, int(abs(rng.normal(rows_per_day/ sessions_today, 6))))
                for i in range(n_utts):
                    ts = day + timedelta(minutes=int(rng.integers(0, 60*10)))  # within ~10h window
                    true_intent = rng.choice(INTENTS)
                    # model prediction with some noise
                    if with_truth:
                        pred = true_intent if rng.random() < 0.78 else rng.choice(INTENTS)
                    else:
                        pred = rng.choice(INTENTS)
                        true_intent = None
                    conf = float(np.clip(rng.normal(0.78 if pred == true_intent else 0.55, 0.12), 0.05, 0.99))
                    senti = rng.choice(SENTI, p=[0.35, 0.5, 0.15])
                    dur = float(np.clip(abs(rng.normal(2.7, 1.1)), 0.3, 10))
                    text = rng.choice([
                        "안녕하세요 선생님.",
                        "이거 해도 돼요?",
                        "물 좀 마실래요.",
                        "싫어요, 하기 싫어요.",
                        "기분이 좋아요!",
                        "이건 뭐예요?",
                        "저랑 놀래요?",
                    ])
                    audio_uri = ""
                    rows.append({
                        "child_id": f"C{child}",
                        "session_id": session_id,
                        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "audio_uri": audio_uri,
                        "stt_text": text,
                        "intent_pred": pred,
                        "confidence": conf,
                        "intent_true": true_intent,
                        "sentiment": senti,
                        "duration_sec": dur,
                        "keywords": "놀이, 질문, 요청",
                    })
    df = pd.DataFrame(rows)
    return df

@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    # Normalize expected columns / types
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    expected = [
        "child_id", "session_id", "timestamp", "audio_uri", "stt_text",
        "intent_pred", "confidence", "intent_true", "sentiment", "duration_sec", "keywords"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df

st.sidebar.header("Data")
font_path_override = st.sidebar.text_input(
    "Korean font path (optional)",
    value="C:/Windows/Fonts/malgun.ttf",  # 윈도우 기본
    help="예: C:/Windows/Fonts/malgun.ttf"
)

upload = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
with_truth = st.sidebar.checkbox("Assume ground truth exists (for demo)", value=True, help="If your CSV has intent_true, uncheck to use it instead.")

if upload is not None:
    df = load_csv(upload)
else:
    df = generate_demo(with_truth=with_truth)

# Basic cleaning
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

df["intent_pred"] = df["intent_pred"].fillna("other")

# -------------------------------
# 2) Sidebar filters
# -------------------------------
st.sidebar.header("Filters")
children = sorted(df["child_id"].dropna().unique().tolist())
child_sel = st.sidebar.multiselect("Child", children, default=children)

sessions = sorted(df.loc[df["child_id"].isin(child_sel), "session_id"].dropna().unique().tolist())
session_sel = st.sidebar.multiselect("Session", sessions, default=sessions[: min(10, len(sessions))])

intents = sorted(df["intent_pred"].dropna().unique().tolist())
intent_sel = st.sidebar.multiselect("Intent (pred)", intents, default=intents)

conf_min, conf_max = st.sidebar.slider("Confidence range", 0.0, 1.0, (0.0, 1.0), 0.01)

# Date range
if df["timestamp"].notna().any():
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    date_range = st.sidebar.date_input("Date range", (tmin.date(), tmax.date()))
else:
    date_range = None

# Apply filters
mask = (
    df["child_id"].isin(child_sel)
    & df["session_id"].isin(session_sel)
    & df["intent_pred"].isin(intent_sel)
    & df["confidence"].between(conf_min, conf_max)
)
if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and pd.notna(df["timestamp"]).any():
    d0, d1 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    mask = mask & df["timestamp"].between(d0, d1)

fdf = df[mask].copy()

# -------------------------------
# 3) Overview KPIs
# -------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Utterances", f"{len(fdf):,}")
col2.metric("Sessions", f"{fdf['session_id'].nunique():,}")
col3.metric("Children", f"{fdf['child_id'].nunique():,}")

if fdf["confidence"].notna().any():
    col4.metric("Avg. Confidence", f"{fdf['confidence'].mean():.2f}")
else:
    col4.metric("Avg. Confidence", "-")

st.divider()

# -------------------------------
# 4) Intent distribution & time series
# -------------------------------
left, right = st.columns((1, 1))

with left:
    st.subheader("Intent distribution (pred)")
    intent_counts = fdf["intent_pred"].value_counts().reset_index()
    intent_counts.columns = ["intent_pred", "count"]
    fig = px.bar(
        intent_counts,
        x="intent_pred",
        y="count",
        text="count",
        color="intent_pred",
        color_discrete_map=INTENT_COLORS,
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Count", xaxis_title="Intent", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


with right:
    st.subheader("Utterances over time")
    if fdf["timestamp"].notna().any():
        ts = fdf.set_index("timestamp").resample("D").size().reset_index(name="count")
        fig_ts = px.line(ts, x="timestamp", y="count")
        fig_ts.update_layout(yaxis_title="Count", xaxis_title="Date")
        st.plotly_chart(fig_ts, use_container_width=True)
    else:
        st.info("No timestamp data available.")

# -------------------------------
# 5) Session Tracking
# -------------------------------
st.subheader("Session-level intent mix")
if not fdf.empty:
    mix = (
        fdf.groupby(["session_id", "intent_pred"]).size().reset_index(name="n")
        .pivot(index="session_id", columns="intent_pred", values="n").fillna(0)
    )
    mix_percent = mix.div(mix.sum(axis=1), axis=0) * 100
    fig_heat = px.imshow(mix_percent, aspect="auto", color_continuous_scale="Blues")
    fig_heat.update_layout(xaxis_title="Intent", yaxis_title="Session", coloraxis_colorbar_title="%")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.info("No data after filters.")

# -------------------------------
# 6) Multimodal Browser (audio + text)
# -------------------------------
st.subheader("Multimodal browser")
show_cols = ["timestamp", "child_id", "session_id", "intent_pred", "confidence", "sentiment", "duration_sec", "stt_text", "audio_uri"]
st.dataframe(fdf[show_cols].sort_values("timestamp" if fdf["timestamp"].notna().any() else show_cols[0]).reset_index(drop=True), use_container_width=True)

sel_idx = st.number_input("Select a row index to preview (from table above)", min_value=0, max_value=max(0, len(fdf)-1), value=0, step=1)
if len(fdf) > 0:
    row = fdf.iloc[int(sel_idx)]
    st.write("**Text**:", row.get("stt_text", ""))
    st.write("**Predicted intent**:", row.get("intent_pred", "-"), " | **Confidence**:", f"{row.get('confidence', float('nan')):.2f}" if pd.notna(row.get("confidence")) else "-")
    if isinstance(row.get("audio_uri"), str) and row.get("audio_uri"):
        try:
            st.audio(row["audio_uri"])  # supports local/URL
        except Exception:
            st.warning("Could not preview audio. Check path/URL or CORS.")
# Export current filtered table
show_cols = ["timestamp","child_id","session_id","intent_pred","confidence",
             "sentiment","duration_sec","stt_text","audio_uri"]
csv_bytes = fdf[show_cols].to_csv(index=False).encode("utf-8-sig")
st.download_button("Download filtered CSV", csv_bytes,
                   file_name="utterances_filtered.csv", mime="text/csv")
st.subheader("WordCloud (by intent)")
wc_opts = [i for i in INTENTS if i in fdf["intent_pred"].unique()]
if len(wc_opts):
    wc_intent = st.selectbox("Choose intent for wordcloud", options=wc_opts)
    wc_df = fdf[fdf["intent_pred"] == wc_intent]
    if _HAS_WC and not wc_df.empty:
        texts = wc_df["stt_text"].dropna().astype(str).tolist()
        if "keywords" in wc_df.columns:
            texts += wc_df["keywords"].dropna().astype(str).tolist()
        # 쉼표를 공백으로 바꿔서 토큰화에 유리하게
        corpus = " ".join(t.replace(",", " ") for t in texts)

        # 👉 한글 폰트 경로 지정: 사이드바 입력값이 없으면 기본값 사용
        font_path = font_path_override or "C:/Windows/Fonts/malgun.ttf"

        from wordcloud import WordCloud
        wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                collocations=False,
                font_path="fonts/NanumGothic.ttf"  # 여기만 추가
            ).generate(text_data)


        st.image(wc.to_array(), use_column_width=True, caption=f"WordCloud — {wc_intent}")
    elif not _HAS_WC:
        st.info("`pip install wordcloud` 후 다시 실행하세요.")
    else:
        st.info("선택한 인텐트에 텍스트가 없습니다.")

st.subheader("Before / After — intent ratio comparison")

all_sessions_sorted = sorted(fdf["session_id"].unique().tolist())
if len(all_sessions_sorted) >= 2:
    default_before = all_sessions_sorted[: min(3, len(all_sessions_sorted)//2 or 1)]
    default_after  = all_sessions_sorted[-min(3, len(all_sessions_sorted)//2 or 1):]

    cba, cab = st.columns(2)
    with cba:
        before_sel = st.multiselect("Before sessions", all_sessions_sorted,
                                    default=default_before, key="before_sel")
    with cab:
        after_sel = st.multiselect("After sessions", all_sessions_sorted,
                                   default=default_after, key="after_sel")

    def intent_percent(df_sub):
        if df_sub.empty: return pd.Series(dtype=float)
        cnt = df_sub["intent_pred"].value_counts()
        return (cnt / cnt.sum() * 100).reindex(INTENTS, fill_value=0.0)

    before_pct = intent_percent(fdf[fdf["session_id"].isin(before_sel)])
    after_pct  = intent_percent(fdf[fdf["session_id"].isin(after_sel)])

    # metrics
    grid = st.columns(len(INTENTS))
    for i, intent in enumerate(INTENTS):
        delta = float(after_pct.iloc[i] - before_pct.iloc[i])
        grid[i].metric(intent, f"{after_pct.iloc[i]:.1f}%", delta=f"{delta:+.1f} pp")

    # grouped bar
    comp = pd.DataFrame({
        "intent": INTENTS,
        "before%": before_pct.values,
        "after%":  after_pct.values,
    })
    fig_ba = px.bar(
        comp.melt(id_vars="intent", var_name="phase", value_name="percent"),
        x="intent", y="percent", color="phase", barmode="group",
        color_discrete_map={"before%": "#9E9E9E", "after%": "#4E79A7"}
    )
    fig_ba.update_layout(yaxis_title="% of utterances", xaxis_title="Intent",
                         legend_title="Phase")
    st.plotly_chart(fig_ba, use_container_width=True)
else:
    st.info("세션이 2개 이상일 때 Before/After 비교가 가능합니다.")

# -------------------------------
# 7) Model Metrics (requires intent_true)
# -------------------------------
st.subheader("Model metrics")
if fdf["intent_true"].notna().any():
    eval_df = fdf.dropna(subset=["intent_true"]).copy()
    y_true = eval_df["intent_true"].astype(str)
    y_pred = eval_df["intent_pred"].astype(str)

    labels = sorted(set(y_true.unique()).union(y_pred.unique()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.3f}")
    c2.metric("Macro Precision", f"{np.mean(prec):.3f}")
    c3.metric("Macro Recall", f"{np.mean(rec):.3f}")
    c4.metric("Macro F1", f"{np.mean(f1):.3f}")

    fig_cm = px.imshow(cm, x=labels, y=labels, color_continuous_scale="Reds")
    fig_cm.update_layout(xaxis_title="Predicted", yaxis_title="True", coloraxis_colorbar_title="Count")
    st.plotly_chart(fig_cm, use_container_width=True)

    perf = pd.DataFrame({
        "intent": labels,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    })
    fig_perf = px.bar(perf.melt(id_vars="intent", var_name="metric", value_name="score"), x="intent", y="score", color="metric", barmode="group")
    fig_perf.update_layout(yaxis_range=[0,1])
    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.info("Ground-truth labels (intent_true) not found in filtered data. Upload CSV with intent_true to see metrics.")

st.divider()

st.caption("Tip: Use consistent colors for each intent across charts; export filtered tables to CSV via the three-dot menu.")
