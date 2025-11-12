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
st.set_page_config(page_title="PrepMate", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ No OpenAI API key found. Add OPENAI_API_KEY to .env or Streamlit Secrets."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# Session State
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("q10", [])
    ss.setdefault("idx", 0)
    ss.setdefault("graded", [])
    ss.setdefault("last_feedback", None)
    ss.setdefault("run_id", 0)
    ss.setdefault("finished", False)


init_state()

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
    "Friendly coach": "Warm, encouraging tone; highlight strengths first, then gentle suggestions.",
    "Strict bar-raiser": "Direct, demanding, precise. Expect clear evidence and reject vagueness.",
    "Motivational mentor": "Inspiring tone; emphasize progress and next steps.",
    "Calm psychologist": "Analytical, thoughtful; emphasize reasoning and awareness.",
    "Playful mock interviewer": "Witty, teasing but fair. Mix humor with insight.",
    "Corporate recruiter": "Professional and evaluative; care about communication and fit.",
    "Sarcastic": "Dry, slightly snarky tone, but still helpful and clever.",
}

AUTO_ZERO_PATTERNS = [
    r"^\s*$",
    r"^\s*(don'?t|do not)\s+know\s*$",
    r"^\s*idk\s*$",
    r"^\s*(don'?t|do not)\s+care\s*$",
]


def misuse_guard(*texts: str) -> bool:
    s = " ".join(t or "" for t in texts).lower()
    return any(
        x in s
        for x in ["phishing", "malware", "exploit", "cheat on", "bypass security"]
    )


def is_auto_zero(ans: str) -> bool:
    if len(ans.split()) <= 2:
        return True
    for pat in AUTO_ZERO_PATTERNS:
        if re.match(pat, ans, re.IGNORECASE):
            return True
    return False


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


def parse_json(text):
    try:
        js = re.search(r"\{[\s\S]*\}", text).group(0)
        js = re.sub(r",(\s*[}\]])", r"\1", js)
        return json.loads(js)
    except Exception:
        return None


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    persona = st.selectbox("Persona", list(persona_guides.keys()), index=0)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    max_tokens = st.slider("Max tokens", 200, 1500, 600)
    USE_MOCK = st.toggle("🧪 Mock mode", False)
    st.caption("Sarcastic persona adds witty feedback 😉")

# =========================
# UI Header
# =========================
st.title("🎯 PrepMate — One-by-One Interview Practice")
st.caption("Get a new question each time, with instant AI feedback and strict scoring.")

topic = st.text_area(
    "💬 What do you want to practice?",
    placeholder="e.g., System design, SQL, leadership...",
    height=80,
)

with st.expander("📄 Optional Context"):
    jd = st.text_area("Job Description", height=100)
    resume = st.text_area("Your Resume Highlights", height=100)

# =========================
# Generate 10 Questions
# =========================
if st.button("🧠 Generate 10 Questions", use_container_width=True):
    prompt = f"""
Generate EXACTLY 10 interview questions about "{topic or 'general readiness'}".
They should mix technical and behavioral ones.
Number them 1 to 10, one per line.
Keep them under 25 words.
Difficulty: {difficulty}
Persona style: {persona_guides[persona]}
"""
    if USE_MOCK:
        q10 = [f"Mock question {i}" for i in range(1, 11)]
    else:
        with st.spinner("Generating questions..."):
            out = client.responses.create(
                model=model, input=prompt, max_output_tokens=400
            ).output_text
            q10 = [
                re.sub(r"^\d+[\).:-]*\s*", "", l.strip())
                for l in out.splitlines()
                if l.strip()
            ][:10]

    st.session_state.q10 = q10
    st.session_state.idx = 0
    st.session_state.graded = []
    st.session_state.finished = False
    st.success("✅ Questions ready! Scroll down to start.")

# =========================
# Question Flow
# =========================
if st.session_state.q10 and not st.session_state.finished:
    q = st.session_state.q10[st.session_state.idx]
    idx = st.session_state.idx + 1
    st.divider()
    st.markdown(f"### Question {idx}/10")
    st.markdown(f"**{q}**")

    ans = st.text_area("Your Answer", key=f"answer_{idx}", height=120)

    if st.button("💬 Get Feedback", key=f"grade_{idx}"):
        if is_auto_zero(ans):
            feedback = {
                "index": idx,
                "question": q,
                "answer": ans,
                "verdict": "bad",
                "points": 0.0,
                "weighted": 0.0,
                "comment": "No useful answer provided.",
                "tip": "Try to give a concrete example next time.",
                "scores": {"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1},
            }
        else:
            grading_prompt = f"""
Persona: {persona_guides[persona]}

Strictly grade this single answer:
Question: "{q}"
Answer: "{ans}"

Rules:
- Give 1–5 for Clarity, Depth, Structure, Overall.
- weighted = (Clarity + Depth + Structure + 2*Overall)/5
- points = weighted/5 (e.g., 4.25 → 0.85)
- verdict = "good" if points ≥ 0.8, "in-between" if 0.4–0.79, else "bad"
- comment ≤ 12 words (persona tone)
- tip ≤ 16 words (persona tone)
If perfect (points ≥ 0.98), tip = "Excellent answer — keep it up."

Return JSON:
{{
 "items": [{{
   "index": 1, "question": "...", "answer": "...",
   "verdict": "...", "points": 0.00, "weighted": 0.00,
   "comment": "...", "tip": "...",
   "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
 }}]
}}
"""
            if USE_MOCK:
                feedback = {
                    "index": idx,
                    "question": q,
                    "answer": ans,
                    "verdict": "good" if len(ans.split()) > 10 else "in-between",
                    "points": 0.9 if len(ans.split()) > 10 else 0.6,
                    "weighted": 4.5 if len(ans.split()) > 10 else 3.0,
                    "comment": (
                        "Well explained." if len(ans.split()) > 10 else "Add clarity."
                    ),
                    "tip": (
                        "Add a real-world example."
                        if len(ans.split()) > 10
                        else "Expand your reasoning."
                    ),
                    "scores": {"Clarity": 4, "Depth": 4, "Structure": 4, "Overall": 4},
                }
            else:
                raw = ask_openai_json(grading_prompt, model=model, max_tokens=800)
                js = parse_json(raw)
                if js and "items" in js:
                    feedback = js["items"][0]
                    feedback["index"] = idx
                else:
                    feedback = {
                        "index": idx,
                        "question": q,
                        "answer": ans,
                        "verdict": "in-between",
                        "points": 0.5,
                        "weighted": 2.5,
                        "comment": "Partial credit, unclear answer.",
                        "tip": "Be concise and structured.",
                        "scores": {
                            "Clarity": 2,
                            "Depth": 3,
                            "Structure": 3,
                            "Overall": 3,
                        },
                    }

        st.session_state.last_feedback = feedback
        st.session_state.graded.append(feedback)
        st.success("✅ Feedback received!")

    fb = st.session_state.last_feedback
    if fb and fb["index"] == idx:
        color = {"good": "green", "in-between": "orange", "bad": "red"}[fb["verdict"]]
        st.markdown(
            f"**Verdict:** <span style='color:{color}'>{fb['verdict'].upper()}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"**Points:** {fb['points']:.2f} · **Weighted:** {fb['weighted']:.2f}/5"
        )
        st.markdown(f"**Comment:** {fb['comment']}")
        st.markdown(f"**Tip:** {fb['tip']}")
        st.progress(fb["points"])

        if st.button("➡️ Next Question"):
            st.session_state.idx += 1
            st.session_state.last_feedback = None
            if st.session_state.idx >= 10:
                st.session_state.finished = True
            rerun()

# =========================
# Final Summary
# =========================
if st.session_state.finished:
    st.divider()
    st.subheader("🏁 Final Summary")

    items = st.session_state.graded
    total_points = sum(i["points"] for i in items)
    avg_weighted = round(sum(i["weighted"] for i in items) / len(items), 2)

    if total_points <= 3:
        note = "❌ Not Ready For Interview"
        color = "red"
    elif total_points <= 6:
        note = "🟠 Almost Ready For The Interview"
        color = "orange"
    else:
        note = "🟢 Ready For The Interview"
        color = "green"

    st.markdown(
        f"<div style='padding:10px;background:{color};color:white;border-radius:8px;font-weight:700'>"
        f"Total Points: {total_points:.2f}/10 · Avg Weighted: {avg_weighted:.2f}/5 · {note}</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Feedback Table")
    st.write("Each row shows your question, shortened answer, overall score, and tip.")
    rows = []
    for i in items:
        rows.append(
            {
                "Question": i["question"],
                "Answer": i["answer"][:60] + ("..." if len(i["answer"]) > 60 else ""),
                "Overall (Weighted)": f"{i['weighted']:.2f}/5",
                "Tip": i["tip"],
            }
        )
    st.dataframe(rows, use_container_width=True)

    if st.button("🔄 Restart Session"):
        st.session_state.q10 = []
        st.session_state.idx = 0
        st.session_state.graded = []
        st.session_state.last_feedback = None
        st.session_state.finished = False
        st.session_state.run_id += 1
        rerun()
