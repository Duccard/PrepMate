import os
import re
import json
import math
import datetime
from typing import List, Dict

import pandas as pd
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
    st.error(
        "❌ No OpenAI API key found in .env or Streamlit Secrets (OPENAI_API_KEY)."
    )
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Session
# =========================
ss = st.session_state


def _ensure_state():
    ss.setdefault("quiz_running", False)
    ss.setdefault("questions", [])
    ss.setdefault("current_idx", 0)
    ss.setdefault("results", [])
    ss.setdefault("topic", "")
    ss.setdefault("jd_text", "")
    ss.setdefault("persona", "Neutral")
    ss.setdefault("difficulty", "Medium")
    ss.setdefault("model", "gpt-4.1-mini")
    ss.setdefault("use_mock", False)


_ensure_state()

# =========================
# Personas & helpers
# =========================
persona_guides = {
    "Neutral": "Professional, concise, objective.",
    "Friendly Coach": "Warm, supportive, suggests concrete improvements.",
    "Strict Bar-Raiser": "Demanding, expects structured reasoning.",
    "Calm Psychologist": "Empathic, reflective tone.",
    "Motivational Mentor": "Encouraging, goal-oriented tone.",
    "Corporate Recruiter": "Polished, business-outcome focus.",
    "Playful Mock Interviewer": "Witty, probing but friendly.",
    "Algorithmic Stickler": "Highly structured and precise.",
    "Sarcastic Interviewer": "Dry, sharp, witty feedback.",
}

persona_badges = {
    "Neutral": "🧭 Neutral Evaluator",
    "Friendly Coach": "🤝 Friendly Coach",
    "Strict Bar-Raiser": "🧱 Strict Bar-Raiser",
    "Calm Psychologist": "🧘 Calm Psychologist",
    "Motivational Mentor": "🚀 Motivational Mentor",
    "Corporate Recruiter": "💼 Corporate Recruiter",
    "Playful Mock Interviewer": "🎭 Playful Mock Interviewer",
    "Algorithmic Stickler": "🧮 Algorithmic Stickler",
    "Sarcastic Interviewer": "😏 Sarcastic Interviewer",
}

persona_signoff = {
    "Neutral": "Balanced performance – focus on weaker areas.",
    "Friendly Coach": "Great effort! Small tweaks will unlock more progress.",
    "Strict Bar-Raiser": "Raise your bar. Add specifics and metrics.",
    "Calm Psychologist": "Be aware of your thinking patterns when answering.",
    "Motivational Mentor": "Strong start! Keep sharpening those details.",
    "Corporate Recruiter": "Polish your storytelling with business outcomes.",
    "Playful Mock Interviewer": "Good show! Add sharper examples next time.",
    "Algorithmic Stickler": "Quantify and structure everything next round.",
    "Sarcastic Interviewer": "Better than silence, but let's aim higher next time.",
}

difficulty_tips = {
    "Easy": "Ask basic, direct questions.",
    "Medium": "Ask questions requiring reasoning and examples.",
    "Hard": "Ask open-ended, complex scenario questions.",
}

DN_PAT = re.compile(
    r"\b(i\s*don'?t\s*know|dont\s*know|don\s*t\s*know|i\s*don'?t\s*care|dont\s*care|idk|no\s*idea|pass)\b",
    re.I,
)


# =========================
# Utility functions
# =========================
def misuse_guard(*texts: str) -> bool:
    lower = " ".join((t or "") for t in texts).lower()
    for f in ("cheat on", "bypass security", "malware", "phishing", "exploit", "ddos"):
        if f in lower:
            return True
    return False


def _force_bad_if_unknown(answer: str) -> bool:
    a = (answer or "").strip()
    if not a:
        return True
    return DN_PAT.search(a) is not None


def normalize_item(it: dict) -> dict:
    it = it or {}
    idx = int(it.get("index", 0))
    qtext = str(it.get("question", ""))
    ans = str(it.get("answer", ""))
    verdict = str(it.get("verdict", "bad")).lower()
    comment = str(it.get("comment", ""))
    tip = str(it.get("tip", ""))
    scores = it.get("scores") or {}

    clarity = int(scores.get("Clarity", 1))
    depth = int(scores.get("Depth", 1))
    structure = int(scores.get("Structure", 1))
    overall = int(scores.get("Overall", 1))

    # STRICT zero-score for unknowns
    if _force_bad_if_unknown(ans):
        verdict = "bad"
        points = 0.0
        clarity = depth = structure = overall = 0
        comment = "No answer provided."
        tip = "Prepare examples before your interview."
    else:
        if verdict == "good":
            points = 1.0
        elif verdict == "in-between":
            points = 0.5
        else:
            points = 0.0

    weighted = round((clarity + depth + structure + 2 * overall) / 5, 2)

    return {
        "index": idx,
        "question": qtext,
        "answer": ans,
        "verdict": verdict,
        "points": round(points, 1),
        "comment": comment,
        "tip": tip,
        "scores": {
            "Clarity": clarity,
            "Depth": depth,
            "Structure": structure,
            "Overall": overall,
        },
        "weighted": weighted,
    }


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    ss.difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value=ss.difficulty
    )
    ss.persona = st.selectbox(
        "Persona",
        list(persona_guides.keys()),
        index=list(persona_guides.keys()).index(ss.persona),
    )
    st.markdown(f"**{persona_badges[ss.persona]}**")
    ss.model = st.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini"], index=0)
    ss.use_mock = st.toggle("🧪 Mock Mode", ss.use_mock)


# =========================
# Start screen
# =========================
def render_start():
    st.subheader("Practice Setup")
    ss.topic = st.text_area("What do you want to practice?", value=ss.topic, height=80)

    with st.expander("📎 Optional: Upload Job Description (TXT, MD)"):
        jd = st.file_uploader("Upload file", type=["txt", "md"])
        if jd:
            ss.jd_text = jd.read().decode("utf-8")

    if st.button("🧠 Generate Questions"):
        generate_questions()


def generate_questions():
    ss.questions = [
        f"Q{i+1}. Mock question about {ss.topic or 'data analysis'}" for i in range(10)
    ]
    ss.results = []
    ss.current_idx = 0
    ss.quiz_running = True
    st.rerun()


# =========================
# Grading mock / example
# =========================
def grade_one(question: str, answer: str) -> Dict:
    # simulate LLM grading
    if _force_bad_if_unknown(answer):
        return normalize_item(
            {
                "index": 1,
                "question": question,
                "answer": answer,
                "verdict": "bad",
                "points": 0.0,
                "comment": "No answer provided.",
                "tip": "Prepare basic data cleaning strategies before interviews.",
                "scores": {"Clarity": 0, "Depth": 0, "Structure": 0, "Overall": 0},
            }
        )
    return normalize_item(
        {
            "index": 1,
            "question": question,
            "answer": answer,
            "verdict": "good",
            "points": 1.0,
            "comment": "Strong, detailed explanation.",
            "tip": "Keep your structure clear.",
            "scores": {"Clarity": 5, "Depth": 5, "Structure": 5, "Overall": 5},
        }
    )


# =========================
# Quiz flow
# =========================
def render_quiz():
    qs = ss.questions
    i = ss.current_idx
    total = len(qs)
    if i >= total:
        render_final()
        return

    st.markdown(
        f"**{persona_badges[ss.persona]}**  •  Difficulty: **{ss.difficulty}**  •  Q {i+1}/{total}"
    )
    q = qs[i]
    st.subheader(q)

    with st.form(key=f"form_{i}"):
        ans_key = f"ans_{i}"
        ans = st.text_area("Your answer", value=ss.get(ans_key, ""), height=120)
        submitted = st.form_submit_button("💬 Submit answer for scoring and feedback")

        if submitted:
            ss[ans_key] = ans
            result = grade_one(q, ans)
            ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
            COLOR = {"good": "green", "in-between": "orange", "bad": "red"}
            icon = ICON[result["verdict"]]
            color = COLOR[result["verdict"]]
            st.markdown(
                f"<span style='color:{color};font-weight:700'>{icon} Verdict: {result['verdict'].upper()} · +{result['points']} pts · Weighted: {result['weighted']}/5</span>",
                unsafe_allow_html=True,
            )
            sc = result["scores"]
            st.write(
                f"**Scores:** Clarity: {sc['Clarity']} · Depth: {sc['Depth']} · Structure: {sc['Structure']} · Overall: {sc['Overall']}"
            )
            st.write(f"**Comment:** {result['comment']}")
            st.write(f"**Tip:** {result['tip']}")

            if len(ss.results) > i:
                ss.results[i] = result
            else:
                ss.results.append(result)

            if st.form_submit_button("➡️ Next question"):
                ss.current_idx += 1
                st.rerun()


# =========================
# Final evaluation
# =========================
def render_final():
    st.subheader("🏁 Final Evaluation")
    items = ss.results
    total_points = round(sum(i["points"] for i in items), 1)
    if total_points >= 7:
        band, color = "Ready for Interview", "green"
    elif total_points >= 3:
        band, color = "Almost Ready", "orange"
    else:
        band, color = "Not Ready", "red"
    st.markdown(
        f"<div style='background:{color};color:white;padding:5px;border-radius:6px;width:fit-content'><b>{band}</b> — Total Points: {total_points}/10</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{persona_badges[ss.persona]} says:** {persona_signoff[ss.persona]}")
    df = pd.DataFrame(
        [
            {
                "Question": i["question"],
                "Answer": i["answer"],
                "Score": i["scores"]["Overall"],
                "Tip": i["tip"],
            }
            for i in items
        ]
    )
    st.dataframe(df, use_container_width=True)

    if st.button("🔁 Take another quiz"):
        ss.quiz_running = False
        ss.questions, ss.results = [], []
        ss.current_idx = 0
        st.rerun()


# =========================
# Router
# =========================
if not ss.quiz_running:
    render_start()
else:
    render_quiz()
