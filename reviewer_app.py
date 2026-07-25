from __future__ import annotations

import html
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

API_BASE = os.getenv("MEDTERM_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(
    page_title="Term Trace · Clinical review",
    page_icon="◌",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --ink: #12252d;
  --muted: #587078;
  --paper: #f5f8f8;
  --panel: #ffffff;
  --line: #cad8da;
  --teal: #087f78;
  --teal-dark: #075c59;
  --amber: #c47a13;
  --danger: #a63d40;
  --ice: #e5f3f2;
}
.stApp { background: var(--paper); color: var(--ink); }
.block-container { max-width: 1440px; padding: 2.1rem 3.2rem 5rem; }
html, body, [class*="css"] { font-family: Aptos, "Segoe UI", sans-serif; }
h1, h2, h3 { font-family: Bahnschrift, "Arial Narrow", sans-serif !important; letter-spacing: -0.035em; }
button, input, textarea { border-radius: 5px !important; }
.brandline { display:flex; align-items:flex-end; justify-content:space-between; border-bottom:1px solid var(--line); padding-bottom:1rem; margin-bottom:1.8rem; }
.wordmark { font-family:Bahnschrift,"Arial Narrow",sans-serif; font-size:2.1rem; font-weight:650; letter-spacing:-.05em; color:var(--ink); }
.wordmark i { color:var(--teal); font-style:normal; }
.eyebrow { color:var(--muted); font:700 .72rem/1.2 Consolas,monospace; letter-spacing:.12em; text-transform:uppercase; }
.safety-lock { border:1px solid var(--line); padding:.45rem .7rem; color:var(--teal-dark); font:700 .72rem Consolas,monospace; background:#fff; }
.case-card { background:var(--panel); border:1px solid var(--line); padding:1.25rem 1.35rem; box-shadow:0 8px 24px rgba(19,51,58,.04); }
.context { font-size:1.18rem; line-height:1.75; color:var(--ink); padding:1rem 0 .4rem; }
.context mark { background:#ffdfa8; color:var(--ink); padding:.08rem .22rem; border-bottom:2px solid var(--amber); }
.trace { display:flex; align-items:center; gap:.55rem; padding:.75rem .85rem; background:var(--ice); border-left:3px solid var(--teal); margin:.7rem 0; overflow-x:auto; }
.trace-label { font:700 .68rem Consolas,monospace; letter-spacing:.1em; color:var(--teal-dark); text-transform:uppercase; }
.phone { white-space:nowrap; font-size:1.08rem; color:var(--ink); }
.rail { height:2px; min-width:28px; background:repeating-linear-gradient(90deg,var(--teal) 0 4px,transparent 4px 8px); }
.status-review { color:var(--amber); font-weight:750; }
.status-abstain { color:var(--muted); font-weight:750; }
.warning { border-left:3px solid var(--danger); background:#fbefef; padding:.75rem 1rem; color:#6f292c; margin:.65rem 0; }
.empty { border:1px dashed var(--line); padding:2.5rem; text-align:center; color:var(--muted); background:rgba(255,255,255,.55); }
[data-testid="stDataFrame"] { border:1px solid var(--line); }
.stButton > button[kind="primary"] { background:var(--teal); border-color:var(--teal); font-weight:700; }
.stButton > button[kind="primary"]:hover { background:var(--teal-dark); border-color:var(--teal-dark); }
*:focus-visible { outline:3px solid #58b7b0 !important; outline-offset:2px; }
@media (max-width: 760px) {
 .block-container { padding:1.2rem 1rem 4rem; }
 .brandline { align-items:flex-start; gap:1rem; }
 .wordmark { font-size:1.65rem; }
}
@media (prefers-reduced-motion: reduce) { * { scroll-behavior:auto !important; transition:none !important; } }
</style>
""",
    unsafe_allow_html=True,
)


def api_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = httpx.request(method, f"{API_BASE}{path}", timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        raise RuntimeError(f"API returned {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Cannot reach the review API at {API_BASE}") from exc


def highlighted_context(span: dict[str, Any]) -> str:
    context = span["context"]
    target = span["span_text"]
    location = context.casefold().find(target.casefold())
    if location < 0:
        return html.escape(context)
    return (
        html.escape(context[:location])
        + f"<mark>{html.escape(context[location : location + len(target)])}</mark>"
        + html.escape(context[location + len(target) :])
    )


def candidate_frame(span: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for rank, candidate in enumerate(span["candidates"], start=1):
        scores = candidate["score_breakdown"]
        rows.append(
            {
                "Rank": rank,
                "Term": candidate["term"],
                "Concept": candidate["concept_id"],
                "Phonetic": scores["phonetic"],
                "Spelling": scores["orthographic"],
                "Final": candidate["score"],
                "Risk": candidate["risk_tier"],
                "Source": candidate["provenance"],
            }
        )
    return pd.DataFrame(rows)


st.markdown(
    """
<div class="brandline">
  <div><div class="eyebrow">Clinical transcript adjudication</div><div class="wordmark">Term <i>Trace</i></div></div>
  <div class="safety-lock">AUTO-COMMIT · LOCKED</div>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Review identity")
    reviewer_id = st.text_input(
        "Reviewer ID", value=st.session_state.get("reviewer_id", "reviewer-01")
    )
    st.session_state["reviewer_id"] = reviewer_id
    st.caption(f"API · {API_BASE}")
    if st.button("Check connection", use_container_width=True):
        try:
            health = api_request("GET", "/health")
            st.success(f"{health['dictionary_entries']} terms · {health['artifact_version']}")
        except RuntimeError as exc:
            st.error(str(exc))
    st.divider()
    st.subheader("Dictionary search")
    dictionary_query = st.text_input(
        "Term, brand, or abbreviation",
        key="dictionary-query",
        placeholder="e.g. Coumadin",
    )
    if dictionary_query:
        try:
            dictionary_results = api_request(
                "GET",
                "/v1/dictionary/search",
                params={"q": dictionary_query, "limit": 8},
            )
            if dictionary_results:
                for item in dictionary_results:
                    st.markdown(
                        f"**{item['term']}**  \n"
                        f"`{item['concept_id']}` · {item['risk_tier']} · {item['provenance']}"
                    )
            else:
                st.caption("No controlled-dictionary matches.")
        except RuntimeError as exc:
            st.error(str(exc))

st.markdown('<div class="eyebrow">New case</div>', unsafe_allow_html=True)
left, right = st.columns([1.45, 0.85], gap="large")
with left:
    transcript = st.text_area(
        "Transcript",
        value="The patient takes met for men 500 mg with breakfast.",
        height=130,
        help="Paste an ASR transcript. Original wording is preserved in the audit record.",
    )
with right:
    audio_file = st.file_uploader("Audio (optional)", type=["wav", "mp3", "m4a", "ogg", "flac"])
    locale = st.text_input(
        "Locale", value="en-US", help="Leave empty to use language identification."
    )
    if audio_file:
        st.audio(audio_file)

if st.button("Find terms for review", type="primary", disabled=not transcript.strip()):
    try:
        with st.spinner("Tracing pronunciations against the controlled dictionary…"):
            if audio_file:
                data = {"transcript": transcript, "locale": locale, "top_k": "5"}
                files = {"audio": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                result = api_request("POST", "/v1/match/audio", data=data, files=files)
            else:
                result = api_request(
                    "POST",
                    "/v1/match",
                    json={"text": transcript, "locale": locale or None, "top_k": 5},
                )
            st.session_state["match_result"] = result
            st.session_state["audio_bytes"] = audio_file.getvalue() if audio_file else None
    except RuntimeError as exc:
        st.error(str(exc))

result = st.session_state.get("match_result")
if result:
    spans = result["spans"]
    st.markdown("---")
    st.markdown(
        f'<div class="eyebrow">Case {html.escape(result["request_id"][:8])} · '
        f"{len(spans)} span{'s' if len(spans) != 1 else ''} · "
        f"{html.escape(result['artifact_version'])}</div>",
        unsafe_allow_html=True,
    )
    if not spans:
        st.markdown(
            '<div class="empty"><strong>No suspicious medical spans.</strong><br>'
            "The transcript stays unchanged; no review action is needed.</div>",
            unsafe_allow_html=True,
        )
    for index, span in enumerate(spans, start=1):
        st.markdown("<br>", unsafe_allow_html=True)
        context_col, candidate_col = st.columns([0.83, 1.17], gap="large")
        with context_col:
            status_class = "status-review" if span["decision"] == "review" else "status-abstain"
            st.markdown(
                f'<div class="case-card"><div class="eyebrow">Span {index} · '
                f'<span class="{status_class}">{html.escape(span["decision"]).upper()}</span></div>'
                f'<div class="context">…{highlighted_context(span)}…</div></div>',
                unsafe_allow_html=True,
            )
            phones = span["pronunciations"]
            phone_html = (
                '<div class="rail"></div>'.join(
                    f'<span class="phone">/{html.escape(item["value"])}/</span>' for item in phones
                )
                or '<span class="phone">No pronunciation</span>'
            )
            st.markdown(
                f'<div class="trace"><span class="trace-label">Evidence rail</span>'
                f'<div class="rail"></div>{phone_html}</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{span['language']} · {span['script']} · risk {span['evidence']['risk_score']:.2f} · "
                f"characters {span['char_start']}–{span['char_end']}"
            )
            safety_reasons = [
                reason
                for reason in span["decision_reason"]
                if reason in {"high_risk_medication", "dose_adjacent", "mixed_script", "low_margin"}
            ]
            if safety_reasons:
                st.markdown(
                    '<div class="warning"><strong>Safety review required</strong><br>'
                    + " · ".join(reason.replace("_", " ") for reason in safety_reasons)
                    + "</div>",
                    unsafe_allow_html=True,
                )
            if st.session_state.get("audio_bytes"):
                st.audio(st.session_state["audio_bytes"])

        with candidate_col:
            st.markdown(
                '<div class="eyebrow">Ranked dictionary candidates</div>', unsafe_allow_html=True
            )
            frame = candidate_frame(span)
            if frame.empty:
                st.info(
                    "No candidate reached the retrieval set. Keep the original or mark unresolved."
                )
            else:
                st.dataframe(
                    frame,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Phonetic": st.column_config.ProgressColumn(
                            min_value=0, max_value=1, format="%.2f"
                        ),
                        "Spelling": st.column_config.NumberColumn(format="%.2f"),
                        "Final": st.column_config.ProgressColumn(
                            min_value=0, max_value=1, format="%.2f"
                        ),
                    },
                )

            candidate_options = {
                f"{item['term']} · {item['concept_id']}": item["concept_id"]
                for item in span["candidates"]
            }
            action_labels = {
                "Accept candidate": "accept_candidate",
                "Keep original": "keep_original",
                "None of the above": "unresolved",
            }
            action_label = st.radio(
                "Decision",
                list(action_labels),
                horizontal=True,
                key=f"action-{span['span_id']}",
            )
            concept_label = None
            if action_labels[action_label] == "accept_candidate" and candidate_options:
                concept_label = st.selectbox(
                    "Chosen term", list(candidate_options), key=f"candidate-{span['span_id']}"
                )
            note = st.text_input("Review note (optional)", key=f"note-{span['span_id']}")
            if st.button(
                "Record decision", key=f"save-{span['span_id']}", disabled=not reviewer_id
            ):
                if action_labels[action_label] == "accept_candidate" and not concept_label:
                    st.error("Choose a candidate before recording this decision.")
                else:
                    try:
                        record = api_request(
                            "POST",
                            f"/v1/matches/{result['request_id']}/spans/{span['span_id']}/review",
                            json={
                                "reviewer_id": reviewer_id,
                                "action": action_labels[action_label],
                                "concept_id": candidate_options.get(concept_label)
                                if concept_label
                                else None,
                                "note": note or None,
                            },
                        )
                        st.success(f"Decision recorded · {record['review_id'][:8]}")
                    except RuntimeError as exc:
                        st.error(str(exc))
else:
    st.markdown(
        '<div class="empty"><strong>Start with a transcript.</strong><br>'
        "Suspicious phrases, pronunciations, and controlled-dictionary candidates will appear here.</div>",
        unsafe_allow_html=True,
    )
