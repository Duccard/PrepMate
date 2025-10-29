import os
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------- Bootstrapping ----------
st.set_page_config(
    page_title="PrepMate • Interview Practice", page_icon="🎤", layout="wide"
)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found in .env / Secrets. Set OPENAI_API_KEY.")
    st.stop()
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------- Session ----------
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "last_questions" not in st.session_state:
    st.session_state.last_questions = ""

# ---------- Helpers ----------
difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions that involve reasoning and real examples.",
    "Hard": "Ask complex, open-ended or scenario-based questions testing depth and creativity.",
}


def misuse_guard(*texts: str) -> bool:
    lower = " ".join(t or "" for t in texts).lower()
    flags = ["cheat on", "bypass security", "malware", "phishing", "exploit", "ddos"]
    return any(f in lower for f in flags)


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    tokens = max(1, chars // 4)
    rates = {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "gpt-4.1": 5.00 / 1_000_000,
        "gpt-4.1-mini": 0.30 / 1_000_000,
    }
    return tokens * rates.get(model, 0.15 / 1_000_000)


def ask_openai(
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


# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎛️ Settings")
    role = st.selectbox(
        "Role / domain",
        ["General", "Data Science", "Backend", "Frontend", "Product", "HR"],
    )
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )

    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p sampling", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 100, 2000, 600, 50)

    USE_MOCK = st.toggle("🧪 Mock Mode (no API calls)", False)
    st.caption("🔒 Basic misuse guard is active.")

# ---------- Header ----------
st.title("sPrepMate — Interview Practice")
st.caption(
    "Generate tailored questions, get STAR feedback, track history, and tune model behavior."
)

# ---------- Inputs ----------
topic = st.text_area(
    "What do you want to practice?",
    placeholder="e.g., SQL joins, system design, behavioral STAR…",
    height=90,
)

with st.expander("📄 Optional context"):
    job_desc = st.text_area("Job description (paste text)", height=140)
    resume = st.text_area("Your resume bullets (paste text)", height=120)

colL, colR = st.columns([2, 1])
with colL:
    gen_btn = st.button("🧠 Generate Questions", use_container_width=True)
with colR:
    st.write("")


# ---------- Safe ask wrapper (supports Mock Mode) ----------
def safe_ask(prompt: str) -> str:
    if USE_MOCK:
        return (
            "Technical:\n"
            "1) Explain the difference between INNER and LEFT JOIN with an example.\n"
            "2) When would you use ROW_NUMBER() vs RANK()?\n"
            "3) Design a rate-limited API; outline components.\n"
            "Behavioral:\n"
            "4) Tell me about a time you handled conflicting priorities.\n"
            "5) Give an example of turning feedback into improvement."
        )
    return ask_openai(
        prompt, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )


# ---------- Generate Questions ----------
if gen_btn:
    if misuse_guard(topic, job_desc, resume):
        st.error("This looks unsafe or out of scope. Please rephrase.")
    else:
        guideline = difficulty_tips[difficulty]
        prompt = f"""
You are an experienced interviewer. Generate tailored interview questions.

Role: {role}
Difficulty: {difficulty}
Guideline: {guideline}
Focus topic(s): {topic or 'General interview readiness'}
Job Description: {job_desc or 'N/A'}
Candidate Resume Bullets: {resume or 'N/A'}

Requirements:
- Return 8 questions total.
- Split: 5 technical, 3 behavioral.
- Keep questions concise and realistic for the role & difficulty.
"""
        with st.spinner("Generating questions…"):
            try:
                out = safe_ask(prompt)
                st.session_state.last_questions = out
                st.session_state.history.append({"type": "questions", "text": out})
                st.subheader("📋 Suggested Questions")
                st.write(out)
                st.caption(
                    f"💰 Estimated prompt cost (rough): ${estimate_cost(len(prompt), model):.5f}"
                )
                try:
                    st.toast("✅ Questions ready!", icon="✅")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                st.info("Tip: turn on 🧪 Mock Mode if you’re out of quota.")

st.divider()

# ---------- Critique ----------
st.subheader("🧩 Critique My Answer")
user_answer = st.text_area(
    "Paste your answer (to any question) for feedback:", height=140
)
crit_btn = st.button("💬 Get Feedback")

if crit_btn:
    if not user_answer.strip():
        st.warning("Please paste your answer above.")
    elif misuse_guard(user_answer):
        st.error("This content looks unsafe or out of scope.")
    else:
        critique_prompt = f"""
You are an expert interviewer. Evaluate the candidate's answer using the STAR framework.
Rate on 1–5 for Clarity, Depth, and Structure.
Then provide a 2–3 sentence strengths summary and ONE concrete improvement tip.
Keep it under ~150 words.

Candidate answer:
{user_answer}
"""
        with st.spinner("Scoring your answer…"):
            try:
                critique = safe_ask(critique_prompt)
                st.write(critique)
                st.session_state.history.append({"type": "critique", "text": critique})
                st.caption(
                    f"💰 Estimated prompt cost (rough): ${estimate_cost(len(critique_prompt), model):.5f}"
                )
                try:
                    st.toast("📝 Feedback ready!", icon="📝")
                except Exception:
                    pass
            except Exception as e:
                st.error(f"OpenAI error: {e}")

# ---------- History ----------
with st.expander("🕓 Session history"):
    if not st.session_state.history:
        st.caption("No history yet — generate questions or request a critique.")
    else:
        for item in st.session_state.history:
            tag = "🧠 Questions" if item["type"] == "questions" else "🧩 Critique"
            st.markdown(f"**{tag}**\n\n{item['text']}\n\n---")
