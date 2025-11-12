import os
import re
import json
import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Bootstrapping
# =========================
st.set_page_config(page_title="Prepmate: Interview Practice", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found in .env / Secrets (OPENAI_API_KEY).")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Persona config
# =========================
PERSONA_GUIDES = {
    "Neutral": {
        "guide": "Objective, concise, evidence-focused. No fluff.",
        "badge": ("#e5e7eb", "#111827"),  # bg, text
        "quote": "Reasonable approach, clear logic.",
    },
    "Friendly coach": {
        "guide": "Warm encouragement, growth-minded, actionable nudges.",
        "badge": ("#dcfce7", "#065f46"),
        "quote": "You’ve got this—nice momentum!",
    },
    "Strict bar-raiser": {
        "guide": "Demanding precision, expects specifics, metrics, and trade-offs.",
        "badge": ("#fee2e2", "#991b1b"),
        "quote": "Details matter—excellence hides there.",
    },
    "Motivational mentor": {
        "guide": "Inspiring, reframes weaknesses as opportunities, urges reflection.",
        "badge": ("#ede9fe", "#5b21b6"),
        "quote": "You’re closer than you think—iterate with intent.",
    },
    "Calm psychologist": {
        "guide": "Analytical, empathetic, probes reasoning and self-awareness.",
        "badge": ("#e0f2fe", "#0c4a6e"),
        "quote": "Interesting—what informed that choice?",
    },
    "Playful mock interviewer": {
        "guide": "Light teasing, fun but insightful, still cares about substance.",
        "badge": ("#fff7ed", "#7c2d12"),
        "quote": "Bold move—now back it up. 😉",
    },
    "Corporate recruiter": {
        "guide": "Polished tone; values structure, clarity, stakeholder-friendliness.",
        "badge": ("#f1f5f9", "#0f172a"),
        "quote": "Good presence—tighten the phrasing a bit.",
    },
    "Sarcastic Interviewer": {
        "guide": "Wry, sharp; witty asides; still fair and substantive.",
        "badge": ("#fde68a", "#78350f"),
        "quote": "Stunning. If we graded on vibes.",
    },
}

DIFFICULTY_TIPS = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions involving reasoning and real examples.",
    "Hard": "Ask complex, open-ended or scenario-based questions testing depth and creativity.",
}


# =========================
# Utilities
# =========================
def misuse_guard(*texts: str) -> bool:
    lower = " ".join(t or "" for t in texts).lower()
    flags = ["cheat on", "bypass security", "malware", "phishing", "exploit", "ddos"]
    return any(f in lower for f in flags)


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    tokens = max(1, chars // 4)
    rates = {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "gpt-4o": 5.00 / 1_000_000,
        "gpt-4.1": 5.00 / 1_000_000,
        "gpt-4.1-mini": 0.30 / 1_000_000,
    }
    return round(tokens * rates.get(model, 0.15 / 1_000_000), 5)


def ask_text(
    prompt: str, *, model: str, temperature: float, top_p: float, max_tokens: int
) -> str:
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        input=prompt,
    )
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def supports_json_mode(m: str) -> bool:
    m = (m or "").lower()
    # Adjust allowlist for your account capabilities
    return "gpt-4.1" in m  # gpt-4.1 and gpt-4.1-mini support JSON mode


def ask_json(prompt: str, *, model: str, max_tokens: int) -> str:
    """Prefer strict JSON mode; fallback to normal mode."""
    if supports_json_mode(model):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.2,
                top_p=1,
                max_output_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            try:
                return resp.output_text
            except Exception:
                return str(resp)
        except Exception:
            pass
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
            top_p=1,
            max_output_tokens=max_tokens,
        )
        try:
            return resp.output_text
        except Exception:
            return str(resp)
    except Exception as e:
        return f"__FALLBACK_ERROR__: {e}"


def json_items_from_text(raw_text: str):
    """Extract {"items":[...]} robustly."""
    js = raw_text.strip()
    m = re.search(r"\{[\s\S]*\}", js)
    if m:
        js = m.group(0)
    js = re.sub(r",(\s*[}\]])", r"\1", js)  # strip trailing commas
    try:
        data = json.loads(js)
        return data.get("items", []), js
    except Exception:
        return [], js


def normalize_answer(a: str) -> str:
    return (a or "").strip()


def short_quote_for(persona: str) -> str:
    return PERSONA_GUIDES.get(persona, PERSONA_GUIDES["Neutral"]).get("quote", "")


def persona_pill(persona: str) -> str:
    bg, fg = PERSONA_GUIDES.get(persona, PERSONA_GUIDES["Neutral"])["badge"]
    return f"<span style='background:{bg};color:{fg};padding:3px 8px;border-radius:999px;font-size:12px;font-weight:700'>{persona}</span>"


def forced_zero_answer(ans: str) -> bool:
    s = (ans or "").strip().lower()
    bads = [
        "i don't know",
        "i dont know",
        "dont know",
        "do not know",
        "i don't care",
        "i dont care",
        "dont care",
        "do not care",
    ]
    return any(b in s for b in bads)


# =========================
# Session defaults
# =========================
if "started" not in st.session_state:
    st.session_state.started = False
if "qs" not in st.session_state:
    st.session_state.qs: List[str] = []
if "idx" not in st.session_state:
    st.session_state.idx = 0  # 0..9
if "results" not in st.session_state:
    st.session_state.results: List[Dict] = []  # per-question results
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    persona = st.selectbox("Interviewer persona", list(PERSONA_GUIDES.keys()), index=0)

    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p sampling", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 100, 2000, 700, 50)
    grade_tokens = max(1200, int(max_tokens * 1.2))
    USE_MOCK = st.toggle("🧪 Mock Mode (no API calls)", False)
    st.caption("🔒 Basic misuse guard is active.")

# =========================
# Header
# =========================
st.title("Prepmate: Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)


# =========================
# Input panel (hidden after start)
# =========================
def _normalize(s: str) -> str:
    return (s or "").replace("\r\n", "\n").strip()


if not st.session_state.started:
    topic = st.text_area(
        "What do you want to practice?",
        placeholder="e.g., SQL joins, system design, behavioral STAR…",
        height=90,
        key="topic_input",
    )

    with st.expander("📎 Optional: Upload Job Description (TXT, MD)"):
        jd_file = st.file_uploader(
            " ", type=["txt", "md"], label_visibility="collapsed"
        )
        jd_text = ""
        if jd_file is not None:
            if getattr(jd_file, "size", 0) > 200_000:
                st.warning("File too large (>200 KB). Please paste key parts instead.")
            else:
                raw = jd_file.read()
                try:
                    jd_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    jd_text = raw.decode("cp1252", errors="ignore")
                jd_text = jd_text[:10000]
    topic = _normalize(st.session_state.get("topic_input", ""))
    jd_text = _normalize(jd_text)

    # Start quiz button
    col_a, col_b = st.columns([2, 1])
    with col_a:
        start_btn = st.button("🧠 Generate Questions", use_container_width=True)
    with col_b:
        st.markdown("&nbsp;")

    if start_btn:
        if misuse_guard(topic, jd_text):
            st.error("This looks unsafe or out of scope. Please rephrase.")
        else:
            # Build prompt
            guide = DIFFICULTY_TIPS[difficulty]
            p = f"""
You are an experienced interviewer.

Difficulty: {difficulty}
Guideline: {guide}
Interviewer persona guideline: {PERSONA_GUIDES[persona]['guide']}
Focus topic(s): {topic or 'General interview readiness'}
Job Description (if any): {jd_text or 'N/A'}

Task:
Create a single set of interview questions that mixes technical and behavioral aspects
relevant to the topic. Return EXACTLY 10 questions, numbered 1 to 10. One per line.
Keep each question under 25 words and realistic.

Format:
1. ...
2. ...
...
10. ...
"""
            if USE_MOCK:
                out = "\n".join(
                    [
                        f"{i}. Mock question {i} about {topic or 'the topic'}"
                        for i in range(1, 11)
                    ]
                )
            else:
                out = ask_text(
                    p,
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )

            # Parse 10 numbered lines
            lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
            qs = []
            for ln in lines:
                head = ln.split(maxsplit=1)[0]
                if head.rstrip(".)").isdigit():
                    q = (
                        ln.split(".", 1)[-1]
                        if "." in ln
                        else ln.split(")", 1)[-1] if ")" in ln else ln[len(head) :]
                    )
                    qs.append(q.strip(" -–•\t"))
            qs = qs[:10]
            while len(qs) < 10:
                qs.append("(placeholder)")

            st.session_state.qs = qs
            st.session_state.idx = 0
            st.session_state.results = []
            st.session_state.started = True
            try:
                st.toast(
                    "✅ Questions ready. Click “Submit answer for scoring and feedback”.",
                    icon="✅",
                )
            except:
                pass
            st.rerun()


# =========================
# One-by-one Q&A
# =========================
def grade_single(question: str, user_answer: str, persona: str, topic: str) -> Dict:
    """
    Returns dict:
      {index, question, answer, verdict, points, comment, tip, scores{Clarity,Depth,Structure,Overall}, weighted}
    """
    # Forced 0 for "I don't know / don't care"
    if forced_zero_answer(user_answer):
        return {
            "verdict": "bad",
            "points": 0.0,
            "comment": "No answer provided.",
            "tip": "Offer a concise attempt, even if partial.",
            "scores": {"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1},
        }

    rubric = f"""
You are an interviewer. Apply this persona strictly: {PERSONA_GUIDES[persona]['guide']}
Topic focus: {topic or 'General readiness'}

Grade the candidate’s answer for the given question.

Rules:
- verdict ∈ {{good, in-between, bad}}.
- points: good=1.0, in-between=0.5, bad=0.0
- scores are integers 1..5
- comment: ≤ 12 words, specific
- tip: ≤ 12 words, actionable
- Make verdicts FAIR (not excessively strict).
- If the answer is vague but has some substance → in-between (0.5).
- If fully off-topic or blank → bad (0).
- If strong, specific, well-structured → good (1).
- Persona tone should influence comment & tip phrasing.

Return ONLY JSON:
{{
  "verdict": "good|in-between|bad",
  "points": 0|0.5|1,
  "comment": "short sentence",
  "tip": "short sentence",
  "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
}}
Question: {question}
Answer: {user_answer[:1200]}
"""

    if USE_MOCK:
        # Simple mock: decent if length > 140 chars, else in-between (unless empty which is forced above)
        L = len(user_answer.strip())
        if L == 0:
            return {
                "verdict": "bad",
                "points": 0.0,
                "comment": "No answer provided.",
                "tip": "Offer a concise attempt, even if partial.",
                "scores": {"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1},
            }
        if L > 240:
            return {
                "verdict": "good",
                "points": 1.0,
                "comment": "Clear, concrete, and structured.",
                "tip": "Keep focusing on specifics.",
                "scores": {"Clarity": 5, "Depth": 4, "Structure": 4, "Overall": 4},
            }
        return {
            "verdict": "in-between",
            "points": 0.5,
            "comment": "Some substance—needs specifics.",
            "tip": "Add metrics or examples.",
            "scores": {"Clarity": 3, "Depth": 2, "Structure": 3, "Overall": 3},
        }

    raw = ask_json(rubric, model=model, max_tokens=700)
    items, cleaned = json_items_from_text(raw)
    # Single-item schema; if model returned dict, parse differently
    parsed = None
    if not items:
        try:
            d = json.loads(cleaned)
            parsed = d
        except Exception:
            # Ultra fallback: in-between
            parsed = {
                "verdict": "in-between",
                "points": 0.5,
                "comment": "Partially clear, needs specifics.",
                "tip": "Add metrics/examples.",
                "scores": {"Clarity": 3, "Depth": 2, "Structure": 3, "Overall": 3},
            }
    else:
        parsed = items[0]

    # Sanitize
    verdict = str(parsed.get("verdict", "in-between")).lower().strip()
    if verdict not in ["good", "in-between", "bad"]:
        verdict = "in-between"
    pts = parsed.get("points", 0.5)
    try:
        pts = float(pts)
    except Exception:
        pts = 0.5
    scores = parsed.get("scores", {}) or {}
    clarity = int(scores.get("Clarity", 3))
    depth = int(scores.get("Depth", 3))
    structure = int(scores.get("Structure", 3))
    overall = int(scores.get("Overall", 3))
    comment = str(parsed.get("comment", "")).strip()[:120]
    tip = str(parsed.get("tip", "")).strip()[:120]

    # Weighted (Overall counts double)
    weighted = round((clarity + depth + structure + 2 * overall) / 5, 1)

    return {
        "verdict": verdict,
        "points": pts,
        "comment": comment or "Shorten, add specifics.",
        "tip": tip or "Add metrics or a concrete example.",
        "scores": {
            "Clarity": clarity,
            "Depth": depth,
            "Structure": structure,
            "Overall": overall,
        },
        "weighted": weighted,
    }


def color_for_verdict(v: str) -> str:
    return {"good": "green", "in-between": "orange", "bad": "red"}.get(v, "gray")


# =========================
# Quiz Flow
# =========================
if st.session_state.started and st.session_state.idx < 10:
    q = st.session_state.qs[st.session_state.idx]
    pill = persona_pill(persona)

    st.markdown(
        f"**Question {st.session_state.idx + 1} of 10** &nbsp; {pill}",
        unsafe_allow_html=True,
    )
    st.write(q)
    st.caption("Click **Submit answer for scoring and feedback**.")

    ans = st.text_area(
        "Your answer",
        key=f"ans_{st.session_state.idx}",
        height=140,
        placeholder="Type your answer here…",
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        submit = st.button(
            "💬 Submit answer for scoring and feedback",
            use_container_width=True,
            key=f"submit_{st.session_state.idx}",
        )
    with c2:
        next_btn = st.button(
            "➡️ Next question",
            use_container_width=True,
            key=f"next_{st.session_state.idx}",
        )

    # Handle submit
    if submit:
        if misuse_guard(ans):
            st.error("This looks unsafe or out of scope. Please rephrase.")
        else:
            result = grade_single(
                q,
                normalize_answer(ans),
                persona,
                st.session_state.get("topic_input", ""),
            )
            # Store result (attach question & answer for final table)
            store = {
                "q_idx": st.session_state.idx + 1,
                "question": q,
                "answer": normalize_answer(ans),
                **result,
            }
            # Replace existing result at this index if re-submitted
            if len(st.session_state.results) == st.session_state.idx:
                st.session_state.results.append(store)
            elif len(st.session_state.results) > st.session_state.idx:
                st.session_state.results[st.session_state.idx] = store

            # Show feedback immediately
            icon = {"good": "✅", "in-between": "⚠️", "bad": "❌"}[result["verdict"]]
            color = color_for_verdict(result["verdict"])
            st.markdown(
                f"<div style='color:{color};font-weight:700'>{icon} "
                f"Verdict: {result['verdict'].upper()} · +{result['points']:.1f} pts · Weighted: {result['weighted']:.1f}/5</div>",
                unsafe_allow_html=True,
            )

            sc = result["scores"]
            st.write(
                f"- **Scores:** Clarity: {sc['Clarity']}/5 · Depth: {sc['Depth']}/5 · "
                f"Structure: {sc['Structure']}/5 · Overall: {sc['Overall']}/5"
            )
            st.write(f"**Comment:** {result['comment']}")
            st.write(f"**Tip:** {result['tip']}")
            # Persona reaction
            st.caption(f"*{short_quote_for(persona)}*")

    # Handle next
    if next_btn:
        # Only allow next if we have a graded result for this index
        if len(st.session_state.results) <= st.session_state.idx:
            st.warning("Submit your answer to get feedback before proceeding.")
        else:
            st.session_state.idx += 1
            st.rerun()

# =========================
# Final Evaluation
# =========================
if st.session_state.started and st.session_state.idx >= 10:
    # Compute totals
    total_points = sum([r.get("points", 0.0) for r in st.session_state.results])
    # Status band
    if total_points < 3:
        band_color = "#fee2e2"
        band_text = "Not Ready for Interview"
        band_fg = "#991b1b"
    elif total_points < 7:
        band_color = "#fef9c3"
        band_text = "Almost Ready for Interview"
        band_fg = "#a16207"
    else:
        band_color = "#dcfce7"
        band_text = "Ready for Interview"
        band_fg = "#065f46"

    st.markdown(
        f"<div style='background:{band_color};color:{band_fg};padding:10px 14px;border-radius:12px;font-weight:700'>"
        f"🏁 {band_text} — Total Points: {total_points:.1f}/10"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write(" ")

    # Results table
    import pandas as pd

    rows = []
    for r in st.session_state.results:
        rows.append(
            {
                "Q": r["q_idx"],
                "Question": r["question"],
                "Answer": r["answer"][:300],
                "Overall (1–5)": r["scores"]["Overall"],
                "Weighted": f"{r['weighted']:.1f}",
                "Points": f"{r['points']:.1f}",
                "Tip": r["tip"],
            }
        )
    df = pd.DataFrame(
        rows,
        columns=[
            "Q",
            "Question",
            "Answer",
            "Overall (1–5)",
            "Weighted",
            "Points",
            "Tip",
        ],
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export + New quiz
    colx, coly = st.columns([1, 1])
    with colx:
        # Markdown export
        def build_md(res):
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            parts = [
                f"# Prepmate Session Export ({now})\n",
                f"**Total Points:** {total_points:.1f}/10\n",
            ]
            for r in res:
                parts.append(
                    f"""## Q{r['q_idx']}
**Question:** {r['question']}
**Answer:** {r['answer']}
**Verdict:** {r['verdict']} (+{r['points']:.1f} pts, weighted {r['weighted']:.1f}/5)
**Scores:** Clarity {r['scores']['Clarity']}/5 · Depth {r['scores']['Depth']}/5 · Structure {r['scores']['Structure']}/5 · Overall {r['scores']['Overall']}/5
**Comment:** {r['comment']}
**Tip:** {r['tip']}

---
"""
                )
            return "\n".join(parts)

        md = build_md(st.session_state.results)
        st.download_button(
            "⬇️ Download Session (Markdown)",
            data=md.encode("utf-8"),
            file_name="prepmate_session.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with coly:
        if st.button("🌀 Take new quiz", use_container_width=True):
            # Full reset
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

# =========================
# Footer
# =========================
st.caption("© 2025 Prepmate — Built with Streamlit & OpenAI API")
