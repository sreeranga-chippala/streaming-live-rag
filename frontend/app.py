from __future__ import annotations

import os
import uuid

import requests
import streamlit as st


st.set_page_config(
    page_title="PRISM LiveRAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
:root {
    --bg: #07111f;
    --panel: rgba(15, 27, 45, .82);
    --border: rgba(148, 163, 184, .16);
    --text: #eef4ff;
    --muted: #91a4bd;
    --accent: #63e6be;
    --accent2: #8ab4ff;
}
.stApp {
    background:
      radial-gradient(circle at 15% 5%, rgba(99,230,190,.12), transparent 28%),
      radial-gradient(circle at 90% 10%, rgba(138,180,255,.13), transparent 28%),
      var(--bg);
    color: var(--text);
}
.block-container { max-width: 1450px; padding-top: 1.2rem; }
.hero {
    padding: 22px 26px; border: 1px solid var(--border);
    border-radius: 24px; background: var(--panel);
    backdrop-filter: blur(18px); margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; letter-spacing: -.04em; }
.hero p { color: var(--muted); margin: 6px 0 0; }
.pill {
    display: inline-block; padding: 6px 12px; border-radius: 999px;
    background: rgba(99,230,190,.12); color: var(--accent);
    border: 1px solid rgba(99,230,190,.25); font-size: .82rem;
}
.card {
    padding: 18px; border-radius: 20px; background: var(--panel);
    border: 1px solid var(--border); min-height: 150px;
}
.label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.value { font-size: 1.45rem; font-weight: 700; margin-top: 6px; }
.source { padding: 12px 14px; border-left: 3px solid var(--accent2);
          background: rgba(138,180,255,.06); border-radius: 10px; margin: 8px 0; }
.stage { padding: 10px 12px; border-radius: 12px; background: rgba(255,255,255,.035); margin: 7px 0; }
.small { color: var(--muted); font-size: .85rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

API_URL = os.getenv("PRISM_API_URL", "http://localhost:8000")

st.markdown(
    f"""
    <div class="hero">
      <div class="pill">● LIVE · STREAMING RAG</div>
      <h1>PRISM LiveRAG</h1>
      <p>Retrieval before the utterance ends · Session {st.session_state.session_id[:8]}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([0.9, 1.6], gap="large")

with left:
    st.markdown("### Live transcript")
    transcript = st.text_area(
        "Speak / paste an utterance",
        height=150,
        placeholder="Ask a multi-part question…",
        label_visibility="collapsed",
    )

    final = st.toggle("Utterance complete", value=True)

    if st.button("Send to LiveRAG", use_container_width=True, type="primary"):
        if transcript.strip():
            payload = {
                "session_id": st.session_state.session_id,
                "text": transcript,
                "is_final": final,
            }
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                st.session_state.messages.append(
                    {"user": transcript, "data": data}
                )
                st.rerun()
            except requests.RequestException as exc:
                st.error(f"API connection failed: {exc}")

    st.markdown("### Pipeline")
    stages = [
        ("01", "Intent understanding"),
        ("02", "Query decomposition"),
        ("03", "Parallel retrieval"),
        ("04", "Fusion / RRF"),
        ("05", "Cross-encoder reranking"),
        ("06", "Grounded generation"),
    ]
    for number, name in stages:
        st.markdown(
            f'<div class="stage"><b>{number}</b> &nbsp; {name}</div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### Conversation")

    if not st.session_state.messages:
        st.info("Your live answer and evidence will appear here.")
    else:
        latest = st.session_state.messages[-1]
        data = latest["data"]

        st.markdown(f"**You**  \n{latest['user']}")

        if data.get("answer"):
            st.markdown("#### Grounded answer")
            st.markdown(data["answer"])

        cols = st.columns(4)
        metrics = [
            ("Action", data.get("action", "—")),
            ("Retrieved", str(data.get("retrieval_count", 0))),
            ("Grounded", str(data.get("groundedness", "—"))),
            ("Latency", f'{data.get("latency_ms", "—")} ms'),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(
                    f'<div class="card"><div class="label">{label}</div>'
                    f'<div class="value">{value}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Evidence")
        sources = data.get("citations", [])
        if not sources:
            st.caption("No retrieval evidence was returned for this turn.")
        for source in sources:
            score = source.get("score")
            score_text = f"{score:.3f}" if isinstance(score, float) else "—"
            st.markdown(
                f"""
                <div class="source">
                  <b>[{source.get('citation_id')}] {source.get('source')}</b>
                  <div class="small">score: {score_text} · chunk: {source.get('chunk_id') or '—'}</div>
                  <div>{source.get('excerpt', '')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "PRISM LiveRAG · Person 3 layer: generation, citations, grounding, "
    "evaluation, API and demo experience"
)
