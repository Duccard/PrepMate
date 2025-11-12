import os
import re
import json
import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Bootstrapping
# =========================
st.set_page_config(page_title="PrepMate: Interview Practice", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error(
        "❌ No OpenAI API key found. Add OPENAI_API_KEY to .env or Streamlit Secrets."
    )
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Session init
# =========================
SS = st.session_state


def _init():
    SS.setdefault("topic", "")
    SS.setdefault("difficulty", "Medium")
    SS.setdefault("persona", "Neutral")
    SS.setdefault("model", "gpt-4.1")
    SS.setdefault("max_tokens", 700)
    SS.setdefault("use_mock", False)

    SS.setdefault("jd_text", "")
    SS.setdefault("jd_pdf_name", None)

    SS.setdefault("questions", [])  # list[str] length 10
    SS.setdefault("idx", 0)  # 0..9
    SS.setdefault("graded", [])  # per-question results
    SS.setdefault("last_feedback", None)  # last question feedback dict
    SS.setdefault("finished", False)
    SS.setdefault(
        "feedback_ready_for_idx", -1
    )  # which index currently has feedback rendered


_init()

# =========================
# Configs
# =========================
difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions involving reasoning and real examples.",
    "Hard": "Ask complex, scenario-based questions testing depth and creativity.",
}

persona_guides = {
    "Neutral": "Professional, concise, unbiased phrasing. Focus on clarity and substance.",
    "Friendly Coach": "Warm, encouraging; highlight strengths first, then gentle suggestions.",
    "Strict Bar-Raiser": "Direct, demanding, precise. Expect clear evidence and reject vagueness.",
    "Motivational Mentor": "Inspiring; emphasize progress and concrete next steps.",
    "Calm Psychologist": "Analytical, reflective; emphasize reasoning and awareness.",
    "Playful Mock Interviewer": "Witty, teasing but fair. Mix humor with insight.",
    "Corporate Recruiter": "Professional and evaluative; focus on communication and business fit.",
    "Algorithmic Stickler": "Highly structured and precise; loves frameworks and metrics.",
    "Sarcastic Interviewer": "Dry, sharp, witty—but still helpful and actionable.",
}

persona_badge = {
    "Neutral": "🧭 Neutral",
    "Friendly Coach": "🤝 Friendly Coach",
    "Strict Bar-Raiser": "🧱 Bar-Raiser",
    "Motivational Mentor": "🚀 Mentor",
    "Calm Psychologist": "🧘 Psychologist",
    "Playful Mock Interviewer": "🎭 Playful",
    "Corporate Recruiter": "💼 Recruiter",
    "Algorithmic Stickler": "🧮 Stickler",
    "Sarcastic Interviewer": "😏 Sarcastic",
}

# Persona “quotes” (no brackets, no quotes; will be merged into feedback)
persona_quote = {
    "Neutral": "Clear, direct, and sufficiently evidenced.",
    "Friendly Coach": "Nice work—two tweaks and you’ll nail it.",
    "Strict Bar-Raiser": "Show the numbers—claims without proof won’t pass.",
    "Motivational Mentor": "You’re close—refine structure and finish strong.",
    "Calm Psychologist": "Notice your reasoning flow—tighten the links.",
    "Playful Mock Interviewer": "Spice level is mild—add heat with specifics.",
    "Corporate Recruiter": "Translate this into business impact.",
    "Algorithmic Stickler": "Enumerate assumptions, state invariants, cite metrics.",
    "Sarcastic Interviewer": "Decent—but let’s not write fortune-cookie wisdom.",
}

COLOR = {"good": "green", "in-between": "orange", "bad": "red"}
ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}

# =========================
# Hard-zero detector (improved)
# =========================
AUTO_ZERO_PATTERNS = [
    r"^\s*$",
    r"^\s*i\s*don'?t\s*know\s*\.?\s*$",
    r"^\s*don'?t\s*know\s*\.?\s*$",
    r"^\s*i\s*do\s*not\s*know\s*\.?\s*$",
    r"^\s*do\s*not\s*know\s*\.?\s*$",
    r"^\s*i\s*don'?t\s*care\s*\.?\s*$",
    r"^\s*don'?t\s*care\s*\.?\s*$",
    r"^\s*idk\s*\.?\s*$",
]
AUTO_ZERO_RE = re.compile("|".join(AUTO_ZERO_PATTERNS), re.IGNORECASE)


def is_auto_zero(ans: str) -> bool:
    a = (ans or "").strip()
    if AUTO_ZERO_RE.match(a):
        return True
    # Extremely low-effort answers (<= 2 words) = hard zero
    return len(a.split()) <= 2


def misuse_guard(*texts: str) -> bool:
    s = " ".join(t or "" for t in texts).lower()
    return any(
        x in s
        for x in ["phishing", "malware", "exploit", "cheat on", "bypass security"]
    )


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def ask_openai_json(prompt: str, *, model: str, max_tokens: int) -> str:
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
            top_p=1,
            max_output_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return resp.output_text
    except Exception as e:
        return f"__ERROR__: {e}"


def parse_json(text: str) -> Any:
    try:
        m = re.search(r"\{[\s\S]*\}\s*$", (text or "").strip())
        if not m:
            return None
        js = m.group(0)
        js = re.sub(r",(\s*[}\]])", r"\1", js)
        return json.loads(js)
    except Exception:
        return None


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    SS.difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value=SS.difficulty
    )
    SS.persona = st.selectbox(
        "Persona",
        list(persona_guides.keys()),
        index=list(persona_guides.keys()).index(SS.persona),
    )
    SS.model = st.selectbox(
        "Model", ["gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini"], index=0
    )
    SS.max_tokens = st.slider("Max tokens", 300, 2000, SS.max_tokens, 50)
    SS.use_mock = st.toggle("🧪 Mock mode", SS.use_mock)

# =========================
# Header
# =========================
st.title("PrepMate: Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)


# =========================
# Start panel (hidden after generation)
# =========================
def render_start():
    st.subheader("Practice Setup")

    SS.topic = st.text_area(
        "What do you want to practice?",
        value=SS.topic,
        placeholder="e.g., System design, SQL, data cleaning, leadership…",
        height=80,
    )

    with st.expander("📄 Optional Context (TXT/MD/PDF)"):
        jd_file = st.file_uploader(
            "Attach Job Description / Notes (TXT, MD, PDF)", type=["txt", "md", "pdf"]
        )
        if jd_file is not None:
            if jd_file.type in ("text/plain", "text/markdown"):
                try:
                    SS.jd_text = jd_file.read().decode("utf-8", errors="ignore")[:10000]
                    SS.jd_pdf_name = None
                    st.success("Text context loaded.")
                except Exception:
                    st.warning("Could not read text file; ignoring.")
            else:
                # PDF: keep only the filename note; do not parse PDF content here
                SS.jd_text = ""
                SS.jd_pdf_name = jd_file.name
                st.info(
                    f"PDF attached: {SS.jd_pdf_name} (not parsed; passed as context note)."
                )

    col1, _ = st.columns([1, 3])
    with col1:
        if st.button("🧠 Generate Questions", use_container_width=True):
            generate_questions()


def generate_questions():
    if misuse_guard(SS.topic, SS.jd_text or ""):
        st.error("This looks unsafe or out of scope. Please rephrase.")
        return

    persona = persona_guides[SS.persona]
    ctx_note = ""
    if SS.jd_text:
        ctx_note = f"\nJob Description (excerpt):\n{SS.jd_text[:1600]}\n"
    elif SS.jd_pdf_name:
        ctx_note = f"\nPDF attached: {SS.jd_pdf_name} (content not parsed)\n"

    q_prompt = f"""
You are an interviewer.

Topic: {SS.topic or 'General readiness'}
Difficulty: {SS.difficulty}
Persona style: {persona}

Task:
- Generate EXACTLY 10 interview questions mixing technical & behavioral.
- Number them 1..10, one per line.
- Each <= 25 words, realistic for the topic & difficulty.
{ctx_note}
Return plain text only.
"""

    if SS.use_mock:
        out = "\n".join(
            [
                f"{i}. Mock question {i} about {SS.topic or 'the role'}"
                for i in range(1, 11)
            ]
        )
    else:
        try:
            out = client.responses.create(
                model=SS.model,
                input=q_prompt,
                temperature=0.5,
                top_p=1,
                max_output_tokens=400,
            ).output_text
        except Exception as e:
            st.error(f"OpenAI error while generating: {e}")
            return

    # Parse into numbered questions
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    q10 = []
    for ln in lines:
        # Accept "1. ..." / "1) ..." / "1 - ..." / "1:" patterns
        if re.match(r"^\d+\s*[\.\)::-]\s*", ln):
            q10.append(re.sub(r"^\d+\s*[\.\)::-]\s*", "", ln).strip())
        elif re.match(r"^\d+\s+", ln):
            q10.append(re.sub(r"^\d+\s+", "", ln).strip())
        else:
            q10.append(ln)
    q10 = [q for q in q10 if q][:10]
    while len(q10) < 10:
        q10.append("(placeholder)")

    SS.questions = q10
    SS.idx = 0
    SS.graded = []
    SS.last_feedback = None
    SS.feedback_ready_for_idx = -1
    SS.finished = False
    rerun()


# =========================
# Grading helpers
# =========================
def normalize_item(item: dict, index: int) -> dict:
    """Make sure the item is well-formed and apply classic strictness mapping."""
    item = item or {}
    q = str(item.get("question", ""))
    a = str(item.get("answer", ""))
    cm = str(item.get("comment", ""))
    tip = str(item.get("tip", ""))
    sc = item.get("scores") or {}
    clarity = int(sc.get("Clarity", 0))
    depth = int(sc.get("Depth", 0))
    structure = int(sc.get("Structure", 0))
    overall = int(sc.get("Overall", 0))

    # Hard zero for don't-know-type answers (before any mapping)
    if is_auto_zero(a):
        verdict = "bad"
        points = 0.0
        clarity = depth = structure = overall = 0
        cm = "No useful answer provided."
        tip = "Prepare concrete examples and basic strategies before interviews."
        weighted = 0.0
    else:
        weighted = round((clarity + depth + structure + 2 * overall) / 5, 2)
        if weighted >= 4.0:
            verdict, points = "good", 1.0
        elif weighted >= 2.0:
            verdict, points = "in-between", 0.5
        else:
            verdict, points = "bad", 0.0

    return {
        "index": index,
        "question": q,
        "answer": a,
        "verdict": verdict,
        "points": round(points, 1),
        "weighted": weighted,
        "comment": cm,
        "tip": tip,
        "scores": {
            "Clarity": clarity,
            "Depth": depth,
            "Structure": structure,
            "Overall": overall,
        },
    }


def grade_one(question: str, answer: str) -> Dict:
    """Grade a single Q/A with persona & difficulty. JSON mode -> normalized."""
    # Fast path: auto-zero first
    if is_auto_zero(answer):
        return normalize_item(
            {
                "question": question,
                "answer": answer,
                "comment": "No useful answer provided.",
                "tip": "Give a concrete example and a short framework next time.",
                "scores": {"Clarity": 0, "Depth": 0, "Structure": 0, "Overall": 0},
            },
            index=SS.idx + 1,
        )

    persona = persona_guides[SS.persona]
    p_quote = persona_quote[SS.persona]

    g_prompt = f"""
You are grading ONE answer in the style of this persona:
{persona}

Question: "{question}"
Answer: "{answer}"

Rules:
- Score each 1..5: Clarity, Depth, Structure, Overall.
- weighted = (Clarity + Depth + Structure + 2*Overall)/5 (2 decimals).
- Classic strictness mapping to points:
    if weighted >= 4.0 → verdict="good", points=1.0
    elif weighted >= 2.0 → verdict="in-between", points=0.5
    else → verdict="bad", points=0.0
- comment ≤ 14 words, tip ≤ 16 words, both in persona tone.
- Merge the persona phrase into the comment naturally (no brackets, no quotes), e.g. "{p_quote} — then your comment".
- If answer is vague, name the missing detail (metric, tradeoff, example, assumption).
- Output valid JSON only, no code fences.

Return:
{{
  "items": [{{
    "question": "{question}",
    "answer": "{answer}",
    "comment": "",
    "tip": "",
    "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
  }}]
}}
"""
    if SS.use_mock:
        words = len((answer or "").split())
        if words > 30:
            sc = {"Clarity": 5, "Depth": 5, "Structure": 4, "Overall": 5}
            cm = f"{p_quote} — Strong, detailed and coherent."
            tp = "Excellent depth; keep explicit trade-offs."
        elif words > 12:
            sc = {"Clarity": 4, "Depth": 3, "Structure": 3, "Overall": 3}
            cm = f"{p_quote} — Add specifics and measurable outcomes."
            tp = "Include one metric and example."
        else:
            sc = {"Clarity": 2, "Depth": 2, "Structure": 2, "Overall": 2}
            cm = f"{p_quote} — Too brief to assess value."
            tp = "Use a 3-step structure and one result."
        js_item = {
            "question": question,
            "answer": answer,
            "comment": cm,
            "tip": tp,
            "scores": sc,
        }
        return normalize_item(js_item, index=SS.idx + 1)

    raw = ask_openai_json(g_prompt, model=SS.model, max_tokens=SS.max_tokens)
    js = parse_json(raw) or {}
    items = js.get("items") or []
    if not items:
        js_item = {
            "question": question,
            "answer": answer,
            "comment": f"{p_quote} — Partial credit; add specifics.",
            "tip": "State one metric and a concrete example.",
            "scores": {"Clarity": 3, "Depth": 2, "Structure": 3, "Overall": 2},
        }
        return normalize_item(js_item, index=SS.idx + 1)

    # Merge persona phrase into returned comment if the model forgot
    item = items[0]
    cmt = str(item.get("comment", "")).strip()
    if p_quote not in cmt:
        item["comment"] = f"{p_quote} — {cmt}" if cmt else p_quote

    return normalize_item(item, index=SS.idx + 1)


# =========================
# Per-question flow
# =========================
def persona_header():
    st.markdown(
        f"""
<div style="display:flex;gap:8px;align-items:center;margin:6px 0 12px 0;">
  <div style="background:#111;color:#fff;padding:6px 10px;border-radius:999px;font-weight:700">{persona_badge[SS.persona]}</div>
  <div style="background:#0b5;padding:6px 10px;border-radius:999px;color:#fff;font-weight:700">Difficulty: {SS.difficulty}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_quiz():
    if SS.idx >= len(SS.questions):
        render_final()
        return

    persona_header()

    q = SS.questions[SS.idx]
    st.subheader(f"Question {SS.idx+1}/10")
    st.markdown(f"**{q}**")

    # ------------- Answer form -------------
    with st.form(key=f"qa_form_{SS.idx}"):
        ans_key = f"ans_{SS.idx}"
        ans_val = st.text_area(
            "Your answer",
            value=SS.get(ans_key, ""),
            height=140,
            placeholder="Answer here…",
        )
        submit = st.form_submit_button("💬 Submit answer for scoring and feedback")

        if submit:
            SS[ans_key] = ans_val
            result = grade_one(q, ans_val)

            # Store/replace graded result for current idx
            if len(SS.graded) > SS.idx:
                SS.graded[SS.idx] = result
            else:
                SS.graded.append(result)

            SS.last_feedback = result
            SS.feedback_ready_for_idx = (
                SS.idx
            )  # mark feedback is ready for current index

    # ------------- Feedback block (after form) -------------
    fb = SS.last_feedback
    if fb and SS.feedback_ready_for_idx == SS.idx:
        verdict = fb["verdict"]
        icon = ICON[verdict]
        color = COLOR[verdict]
        st.markdown(
            f"<div style='padding:10px;border:1px solid {color};border-left:6px solid {color};border-radius:8px'>"
            f"<div style='font-weight:800;color:{color}'>{icon} Verdict: {verdict.upper()} · +{fb['points']:.1f} pts · Weighted: {fb['weighted']:.2f}/5</div>"
            f"<div><b>Scores:</b> Clarity: {fb['scores']['Clarity']}/5 · Depth: {fb['scores']['Depth']}/5 · "
            f"Structure: {fb['scores']['Structure']}/5 · Overall: {fb['scores']['Overall']}/5</div>"
            f"<div style='margin-top:6px'><b>Feedback:</b> {fb['comment']}</div>"
            f"<div><b>Tip:</b> {fb['tip']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(
            0.0 if verdict == "bad" else (0.5 if verdict == "in-between" else 1.0)
        )

        # Next button OUTSIDE the form, keyed to current index
        if st.button("➡️ Next question", key=f"next_{SS.idx}"):
            SS.idx += 1
            SS.last_feedback = None
            SS.feedback_ready_for_idx = -1
            if SS.idx >= 10:
                SS.finished = True
            rerun()


# =========================
# Final summary
# =========================
def readiness_badge(total_pts: float) -> str:
    if total_pts > 6.999:
        return "🟢 Ready For Interview"
    if total_pts > 2.999:
        return "🟠 Almost Ready For Interview"
    return "🔴 Not Ready For Interview"


def render_final():
    st.subheader("🏁 Final Evaluation")
    items = SS.graded
    total_points = round(sum(i.get("points", 0) for i in items), 2)
    avg_weighted = round(
        sum(i.get("weighted", 0) for i in items) / max(len(items), 1), 2
    )
    badge = readiness_badge(total_points)
    if "🟢" in badge:
        bg = "#1d7d46"
    elif "🟠" in badge:
        bg = "#b26a00"
    else:
        bg = "#a11"

    st.markdown(
        f"""
<div style="padding:12px;background:{bg};color:white;border-radius:10px;font-weight:800;display:inline-block">
  {badge} — Total Points: {total_points}/10 · Avg Weighted: {avg_weighted}/5
</div>
<div style="margin-top:8px;font-style:italic;opacity:0.9">
  <b>{persona_badge[SS.persona]}</b> — {persona_quote[SS.persona]}
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Feedback Table")
    rows = []
    for it in items:
        rows.append(
            {
                "Question": it["question"],
                "Answer": (it["answer"] or "")[:120]
                + ("…" if len(it["answer"] or "") > 120 else ""),
                "Overall score (1–5)": it["scores"]["Overall"],
                "Tip": it["tip"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    if st.button("🔄 Take new quiz"):
        SS.questions = []
        SS.idx = 0
        SS.graded = []
        SS.last_feedback = None
        SS.feedback_ready_for_idx = -1
        SS.finished = False
        rerun()


# =========================
# Router
# =========================
if not SS.questions or SS.finished:
    # Show setup only when no active quiz or finished
    render_start()
    if SS.questions and not SS.finished:
        pass
else:
    # Hide setup once questions exist
    render_quiz()
