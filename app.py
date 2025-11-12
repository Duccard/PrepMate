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
    SS.setdefault("max_tokens", 800)
    SS.setdefault("use_mock", False)

    SS.setdefault("jd_text", "")
    SS.setdefault("jd_pdf_name", None)

    SS.setdefault("questions", [])  # list[str] length 10
    SS.setdefault("idx", 0)  # 0..9
    SS.setdefault("graded", [])  # per-question results dicts
    SS.setdefault("last_feedback", None)  # dict for current Q
    SS.setdefault("feedback_ready_for_idx", -1)
    SS.setdefault("finished", False)  # True -> show final


_init()

# =========================
# Configs
# =========================
difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions involving reasoning and real examples.",
    "Hard": "Ask complex, scenario-based questions testing depth and creativity.",
}
difficulty_colors = {
    "Easy": "#1b8f4b",  # green
    "Medium": "#caa000",  # yellow
    "Hard": "#b21f2d",  # red
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

# First-line persona flavor (used if the model fails to return persona_comment)
persona_line = {
    "Neutral": "Clear, direct, and sufficiently evidenced.",
    "Friendly Coach": "Good energy—two small tweaks and you’ll nail it.",
    "Strict Bar-Raiser": "Show the numbers—claims without proof won’t pass.",
    "Motivational Mentor": "Strong direction—tighten structure and finish with impact.",
    "Calm Psychologist": "Observe your reasoning flow—align claims to evidence.",
    "Playful Mock Interviewer": "Spice level is mild—kick it up with specifics.",
    "Corporate Recruiter": "Translate this into business impact and stakeholder value.",
    "Algorithmic Stickler": "State assumptions, define invariants, cite metrics.",
    "Sarcastic Interviewer": "Nice. Let’s step beyond fortune-cookie wisdom, shall we?",
}

COLOR = {
    "very-good": "#167e2d",
    "good": "#3da83c",
    "intermediate": "#caa000",
    "bad": "#cc6d1a",
    "very-bad": "#b21f2d",
}
ICON = {
    "very-good": "🌟",
    "good": "✅",
    "intermediate": "🟡",
    "bad": "⚠️",
    "very-bad": "❌",
}

# =========================
# Hard-zero detector
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
# Start (hidden after generation)
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

    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    q10 = []
    for ln in lines:
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
# Scoring logic (5-tier)
# =========================
def tier_from_weighted(weighted: float) -> Dict[str, Any]:
    """
    Tiers:
      0–1  -> very-bad (0.00)
      1–2  -> bad (0.25)
      2–3  -> intermediate (0.50)
      3–4  -> good (0.75)
      4–5  -> very-good (1.00)
    """
    if weighted < 1.0:
        return {"tier": "very-bad", "points": 0.00}
    if weighted < 2.0:
        return {"tier": "bad", "points": 0.25}
    if weighted < 3.0:
        return {"tier": "intermediate", "points": 0.50}
    if weighted < 4.0:
        return {"tier": "good", "points": 0.75}
    return {"tier": "very-good", "points": 1.00}


def heuristic_strength_bonus(answer: str) -> float:
    """
    Light bump if model lowballs a clearly strong answer.
    Max +0.5 to weighted (capped at 5.0).
    """
    a = (answer or "").lower()
    score = 0
    if any(k in a for k in ["e.g.", "for example", "for instance", "example:"]):
        score += 1
    if any(
        k in a
        for k in [
            "%",
            "percent",
            "ms",
            "latency",
            "throughput",
            "downtime",
            "nps",
            "conversion",
            "f1",
            "auc",
        ]
    ):
        score += 1
    if any(
        k in a
        for k in ["trade-off", "tradeoff", "pros and cons", "cons", "constraints"]
    ):
        score += 1
    if any(k in a for k in ["first,", "second,", "third,", "step", "phase", "stage"]):
        score += 1
    if "because" in a or "therefore" in a:
        score += 1
    return 0.5 if score >= 3 else (0.25 if score == 2 else 0.0)


def normalize_item(item: dict, index: int) -> dict:
    """Finalize fields, apply tiering and heuristic bump; enforce hard-zero."""
    q = str(item.get("question", ""))
    a = str(item.get("answer", ""))
    cm = str(item.get("comment", ""))
    tip = str(item.get("tip", ""))
    pc = str(item.get("persona_comment", ""))
    sc = item.get("scores") or {}
    clarity = int(sc.get("Clarity", 0))
    depth = int(sc.get("Depth", 0))
    structure = int(sc.get("Structure", 0))
    overall = int(sc.get("Overall", 0))

    if is_auto_zero(a):
        # Hard zero everywhere
        clarity = depth = structure = overall = 0
        weighted = 0.0
        tier_info = {"tier": "very-bad", "points": 0.00}
        if not cm:
            cm = "No useful answer provided."
        if not tip:
            tip = "Prepare concrete examples and basic strategies before interviews."
    else:
        weighted = round((clarity + depth + structure + 2 * overall) / 5, 2)
        # bump if strong signals exist and weighted is lowish
        bump = heuristic_strength_bonus(a)
        if bump and weighted < 4.0:
            weighted = min(5.0, round(weighted + bump, 2))
        tier_info = tier_from_weighted(weighted)

    return {
        "index": index,
        "question": q,
        "answer": a,
        "tier": tier_info["tier"],
        "points": round(tier_info["points"], 2),
        "weighted": weighted,
        "comment": cm,
        "tip": tip,
        "persona_comment": pc if pc.strip() else persona_line[SS.persona],
        "scores": {
            "Clarity": clarity,
            "Depth": depth,
            "Structure": structure,
            "Overall": overall,
        },
    }


def grade_one(question: str, answer: str) -> Dict:
    """Grade a single Q/A with persona & difficulty. Returns normalized dict."""
    # Hard zero path
    if is_auto_zero(answer):
        base = {
            "question": question,
            "answer": answer,
            "persona_comment": f"{persona_line[SS.persona]} “Start with one example and one metric.”",
            "comment": "No useful answer provided.",
            "tip": "Give one concrete example and a short framework next time.",
            "scores": {"Clarity": 0, "Depth": 0, "Structure": 0, "Overall": 0},
        }
        return normalize_item(base, index=SS.idx + 1)

    persona = persona_guides[SS.persona]
    g_prompt = f"""
You are grading ONE answer in the style of this persona:
{persona}

Question: "{question}"
Answer: "{answer}"

Rules:
- Score each 1..5: Clarity, Depth, Structure, Overall.
- weighted = (Clarity + Depth + Structure + 2*Overall)/5 (2 decimals).
- Create a single natural-language paragraph (30–60 words) as persona_comment:
  * It should sound like a human interviewer.
  * Merge personality and substance.
  * Include one short quote (with standard quotes) inside the paragraph.
  * Do not use brackets. Do not label it.
- comment ≤ 14 words (concise critique).
- tip ≤ 16 words (one actionable suggestion).

Return ONLY valid JSON:
{{
  "items": [{{
    "question": "{question}",
    "answer": "{answer}",
    "persona_comment": "",
    "comment": "",
    "tip": "",
    "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
  }}]
}}
"""
    if SS.use_mock:
        words = len((answer or "").split())
        if words > 40:
            sc = {"Clarity": 5, "Depth": 5, "Structure": 4, "Overall": 5}
            pc = "Strong structure and evidence; I appreciate the trade-offs you name. “Own your decisions.” Keep the metrics explicit and your sequence crisp; that’s what makes leadership tangible."
            cm = "Rigorous, structured, outcome-focused."
            tp = "Keep explicit metrics and trade-offs."
        elif words > 16:
            sc = {"Clarity": 4, "Depth": 3, "Structure": 3, "Overall": 3}
            pc = "Nice arc; you’re nearly there—“specifics beat slogans.” Add one metric and a tighter step-wise flow to convert this from solid to persuasive."
            cm = "Good direction—add metrics and tradeoffs."
            tp = "Include one metric and example."
        else:
            sc = {"Clarity": 2, "Depth": 2, "Structure": 2, "Overall": 2}
            pc = "Sketchy outline. “Say less, mean more.” Give a concrete example, quantify impact, and state one constraint you managed."
            cm = "Too brief to assess value."
            tp = "Use a 3-step structure and one result."
        js_item = {
            "question": question,
            "answer": answer,
            "persona_comment": pc,
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
            "persona_comment": f"{persona_line[SS.persona]} “Specifics beat slogans.” Add a concrete metric and one explicit trade-off to anchor your claim.",
            "comment": "Partial credit—add specifics and measurable outcomes.",
            "tip": "State one metric and a concrete example.",
            "scores": {"Clarity": 3, "Depth": 2, "Structure": 3, "Overall": 2},
        }
        return normalize_item(js_item, index=SS.idx + 1)

    item = items[0]
    if (
        not isinstance(item.get("persona_comment", ""), str)
        or not item["persona_comment"].strip()
    ):
        item["persona_comment"] = (
            f"{persona_line[SS.persona]} “Specifics beat slogans.”"
        )
    return normalize_item(item, index=SS.idx + 1)


# =========================
# UI helpers
# =========================
def persona_header():
    diff = SS.difficulty
    diff_color = difficulty_colors[diff]
    st.markdown(
        f"""
<div style="display:flex;gap:8px;align-items:center;margin:6px 0 12px 0;">
  <div style="background:#111;color:#fff;padding:6px 10px;border-radius:999px;font-weight:700">{SS.persona}</div>
  <div style="background:{diff_color};padding:6px 10px;border-radius:999px;color:#fff;font-weight:700">Difficulty: {diff}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def item_or_default(d: dict, key: str, default: str) -> str:
    try:
        v = d.get(key)
        if isinstance(v, str) and v.strip():
            return v
    except Exception:
        pass
    return default


# =========================
# Per-question flow
# =========================
def render_quiz():
    if SS.finished:
        render_final()
        return

    if not SS.questions:
        return

    if SS.idx >= len(SS.questions):
        SS.finished = True
        render_final()
        return

    persona_header()

    q = SS.questions[SS.idx]
    st.subheader(f"Question {SS.idx+1}/10")
    st.markdown(f"**{q}**")

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
            if len(SS.graded) > SS.idx:
                SS.graded[SS.idx] = result
            else:
                SS.graded.append(result)
            SS.last_feedback = result
            SS.feedback_ready_for_idx = SS.idx

    fb = SS.last_feedback
    if fb and SS.feedback_ready_for_idx == SS.idx:
        tier = fb["tier"]
        icon = ICON[tier]
        color = COLOR[tier]

        # 1) Persona paragraph FIRST (no label)
        st.markdown(
            f"<div style='padding:12px;border:1px solid {color};border-left:6px solid {color};border-radius:8px;margin-bottom:8px'>"
            f"{fb['persona_comment']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # 2) Verdict + Metrics
        st.markdown(
            f"<div style='padding:10px;border:1px dashed {color};border-radius:8px;margin-bottom:10px'>"
            f"<div style='font-weight:800;color:{color}'>{icon} Verdict: {tier.replace('-', ' ').upper()} · +{fb['points']:.2f} pts · Weighted: {fb['weighted']:.2f}/5</div>"
            f"<div><b>Scores:</b> Clarity: {fb['scores']['Clarity']}/5 · Depth: {fb['scores']['Depth']}/5 · "
            f"Structure: {fb['scores']['Structure']}/5 · Overall: {fb['scores']['Overall']}/5</div>"
            f"<div style='margin-top:6px'><b>Comment:</b> {fb['comment']}</div>"
            f"<div><b>Tip:</b> {fb['tip']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Progress bar roughly scaled to tier points (0.0..1.0)
        st.progress(fb["points"])

        colA, colB = st.columns([1, 1])
        with colA:
            if SS.idx < 9:
                if st.button("➡️ Next question", key=f"next_{SS.idx}"):
                    SS.idx += 1
                    SS.last_feedback = None
                    SS.feedback_ready_for_idx = -1
                    rerun()
        with colB:
            if SS.idx == 9:
                if st.button("🏁 Finish & view results", key="finish_btn"):
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
  {badge} — Total Points: {total_points:.2f}/10 · Avg Weighted: {avg_weighted:.2f}/5
</div>
""",
        unsafe_allow_html=True,
    )

    # Table (wide Answer column)
    st.markdown("### Feedback Table")
    rows = []
    for it in items:
        rows.append(
            {
                "Question": it["question"],
                "Answer": it["answer"],  # full text; we will widen this column
                "Overall (1–5)": it["scores"]["Overall"],
                "Tip": it["tip"],
            }
        )
    df = pd.DataFrame(rows)

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Answer": st.column_config.TextColumn(width="large"),
            "Question": st.column_config.TextColumn(width="medium"),
            "Tip": st.column_config.TextColumn(width="medium"),
            "Overall (1–5)": st.column_config.NumberColumn(format="%d"),
        },
        hide_index=True,
    )

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
if SS.questions and not SS.finished:
    render_quiz()
elif SS.finished:
    render_final()
else:
    render_start()
