import os
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# --- Bootstrapping & config ---
st.set_page_config(page_title="Interview Practice", page_icon="🎤", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found in .env (OPENAI_API_KEY=sk-...).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = (
        []
    )  # [{"type": "question"|"critique", "text": "..."}]
if "last_questions" not in st.session_state:
    st.session_state.last_questions: str = ""


# --- Helpers ---
def ask_openai(prompt: str, model: str, temperature: float) -> str:
    """Call OpenAI Responses API and return text; raise on error."""
    resp = client.responses.create(model=model, temperature=temperature, input=prompt)
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    """
    VERY rough estimate: ~4 chars per token.
    Update prices if needed.
    """
    tokens = max(1, chars // 4)
    # Placeholder rates per 1K tokens (USD) — adjust to your plan if needed
    rates = {
        "gpt-4o-mini": 0.15 / 1_000,  # $0.15 / 1K tokens (example)
        "gpt-4o": 0.50 / 1_000,  # example number
        "gpt-4.1": 5.00 / 1_000,  # example number
        "gpt-4.1-mini": 0.30 / 1_000,  # example number
    }
    price = rates.get(model, 0.15 / 1_000)
    return tokens * price


def misuse_guard(*texts: str) -> bool:
    lower = " ".join(t or "" for t in texts).lower()
    flags = ["cheat on", "bypass security", "malware", "phishing", "exploit", "ddos"]
    return any(f in lower for f in flags)


# --- Sidebar controls ---
with st.sidebar:
    st.header("🎛️ Settings")
    role = st.selectbox(
        "Role / domain",
        ["General", "Data Science", "Backend", "Frontend", "Product", "HR"],
    )
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    st.caption("🔒 Basic misuse guard is active.")

# --- Main UI ---
st.title("🎤 Interview Practice App")
st.caption("Hour 2: Critiques, session history, and pricing estimates.")

colL, colR = st.columns([2, 1])

with colL:
    topic = st.text_area(
        "What do you want to practice?",
        placeholder="e.g., SQL joins, system design, behavioral STAR…",
    )
    job_desc = st.text_area("Optional: Job description (paste text)")
    resume = st.text_area("Optional: Your resume bullets (paste text)")

with colR:
    st.subheader("Actions")
    gen_btn = st.button("🧠 Generate Questions", use_container_width=True)

# --- Generate Questions ---
if gen_btn:
    if misuse_guard(topic, job_desc, resume):
        st.error(
            "Your request looks unsafe or out of scope for interview practice. Please rephrase."
        )
    else:
        prompt = f"""
You are an experienced interviewer. Generate tailored interview questions.

Role: {role}
Difficulty: {difficulty}
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
                out = ask_openai(prompt, model=model, temperature=temperature)
                st.session_state.last_questions = out
                st.session_state.history.append({"type": "question", "text": out})
                st.subheader("📋 Suggested Questions")
                st.write(out)
                est = estimate_cost(len(prompt), model)
                st.caption(f"💰 Estimated prompt cost (very rough): ${est:.5f}")
                st.success("Done!")
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                st.info(
                    "Tip: If you see a quota error, add billing or switch to mock mode."
                )

st.divider()

# --- Critique My Answer ---
st.subheader("🧩 Critique My Answer")
user_answer = st.text_area(
    "Paste your answer here (to one of the questions above, or any answer you want feedback on):",
    height=140,
)
crit_btn = st.button("💬 Get Feedback")

if crit_btn:
    if not user_answer.strip():
        st.warning("Please paste your answer above.")
    elif misuse_guard(user_answer):
        st.error("This content looks unsafe or out of scope for interview practice.")
    else:
        critique_prompt = f"""
You are an expert interviewer. Evaluate the candidate's answer using the STAR framework.

Rate the answer on a 1–5 scale for:
- Clarity
- Depth
- Structure

Then provide:
- A 2–3 sentence summary of strengths.
- One concrete improvement tip.

Return a short, readable response (max ~150 words).

Candidate answer:
{user_answer}
"""
        with st.spinner("Scoring your answer…"):
            try:
                critique = ask_openai(critique_prompt, model=model, temperature=0.7)
                st.write(critique)
                st.session_state.history.append({"type": "critique", "text": critique})
                est = estimate_cost(len(critique_prompt), model)
                st.caption(f"💰 Estimated prompt cost (very rough): ${est:.5f}")
                st.success("Feedback ready!")
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                st.info("Tip: Add billing or use a different key if you hit quota.")

# --- Session History ---
with st.expander("🕓 Session history"):
    if not st.session_state.history:
        st.caption(
            "No history yet. Generate questions or request a critique to populate this section."
        )
    else:
        for item in st.session_state.history:
            tag = "🧠 Questions" if item["type"] == "question" else "🧩 Critique"
            st.markdown(f"**{tag}**\n\n{item['text']}\n\n---")

st.caption(
    "✅ Hour 2 complete — you now have Q generation, critiques, session history, and rough pricing."
)
