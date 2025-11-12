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
    st.error("❌ No OpenAI API key found. Add it to your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Session
# =========================
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "q10" not in st.session_state:
    st.session_state.q10 = []
if "current_q" not in st.session_state:
    st.session_state.current_q = 0
if "answers" not in st.session_state:
    st.session_state.answers = []
if "grading" not in st.session_state:
    st.session_state.grading = []

# =========================
# Persona guides (revamped)
# =========================
persona_guides = {
    "Neutral": "Professional and balanced. Gives objective, straightforward feedback.",
    "Friendly coach": "Warm and supportive. Offers encouragement and constructive tips with a kind tone.",
    "Strict bar-raiser": "Demanding and exact. Values precision, confidence, and well-structured answers.",
    "Motivational mentor": "Inspiring, positive tone. Pushes you to see growth and improvement.",
    "Calm psychologist": "Analytical and empathetic. Focuses on emotional intelligence and self-awareness.",
    "Playful mock interviewer": "Uses humor and teasing to make feedback memorable but insightful.",
    "Corporate recruiter": "Polished, HR-oriented. Focuses on professionalism and communication clarity.",
    "Sarcastic Interviewer": "Witty, sharp, and slightly playful. Gives biting but honest feedback 😉",
}

difficulty_tips = {
    "Easy": "Ask simple questions that test basic reasoning or self-awareness.",
    "Medium": "Ask realistic interview questions mixing behavioral and technical insight.",
    "Hard": "Ask open-ended or scenario-based questions that test depth, logic, and creativity.",
}

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    persona = st.selectbox("Interviewer Persona", list(persona_guides.keys()), index=0)

    st.markdown("### ⚙️ Model")
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 100, 2000, 800, 50)
    USE_MOCK = st.toggle("🧪 Mock mode (no API)", False)


# =========================
# Helper Functions
# =========================
def misuse_guard(*texts: str) -> bool:
    bad = ["bypass security", "phishing", "exploit", "cheat on", "ddos"]
    return any(b in " ".join(texts).lower() for b in bad)


def ask_openai(prompt: str) -> str:
    if USE_MOCK:
        return "1. Mock Q1\n2. Mock Q2\n3. Mock Q3\n4. Mock Q4\n5. Mock Q5\n6. Mock Q6\n7. Mock Q7\n8. Mock Q8\n9. Mock Q9\n10. Mock Q10"
    resp = client.responses.create(
        model=model, temperature=temperature, max_output_tokens=max_tokens, input=prompt
    )
    return resp.output_text


def ask_openai_json(prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.3,
        max_output_tokens=max_tokens,
    )
    return resp.output_text


# =========================
# Header
# =========================
st.title("PrepMate — Interactive Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)

# =========================
# Input Section
# =========================
topic = st.text_area(
    "What do you want to practice?",
    placeholder="e.g. Cooking, Data analysis, Team leadership...",
)
jd_file = st.file_uploader(
    "📎 Optional: Upload Job Description (TXT, MD)", type=["txt", "md"]
)

job_desc = ""
if jd_file:
    job_desc = jd_file.read().decode("utf-8", errors="ignore")[:8000]

# =========================
# Generate Questions
# =========================
if st.button("🧠 Generate 10 Questions"):
    if not topic:
        st.warning("Please enter a topic first.")
    else:
        prompt = f"""
You are an interviewer creating realistic {difficulty.lower()}-level interview questions.

Persona: {persona_guides[persona]}
Topic: {topic}
Job Description: {job_desc or 'N/A'}

Generate exactly 10 questions mixing technical and behavioral aspects.
Each question under 25 words. Number them 1 to 10.
"""
        with st.spinner("Generating questions..."):
            out = ask_openai(prompt)
        q10 = [
            re.sub(r"^\d+[\).\-]*", "", q).strip()
            for q in out.splitlines()
            if q.strip()
        ]
        st.session_state.q10 = q10[:10]
        st.session_state.current_q = 0
        st.session_state.answers = []
        st.session_state.grading = []
        st.success("✅ 10 questions ready! Click 'Next' to start answering.")

# =========================
# Interactive Q&A Flow
# =========================
if st.session_state.q10:
    qlist = st.session_state.q10
    idx = st.session_state.current_q

    if idx < len(qlist):
        st.subheader(f"Question {idx+1}/10")
        st.markdown(f"**{qlist[idx]}**")
        ans = st.text_area("Your answer:", key=f"ans_{idx}", height=100)
        if st.button("Submit Answer"):
            if not ans.strip():
                st.warning("Please type an answer.")
            else:
                grading_prompt = f"""
You are an expert interviewer using persona: {persona_guides[persona]}.
Evaluate the candidate's single answer.

Rules:
- If answer includes "don't care", "don't know", or is empty → 0 pts, verdict = "bad"
- Otherwise, judge naturally: be fair but not overly harsh
- Use verdict: "good", "in-between", "bad"
- Give points: 1, 0.5, or 0
- Weighted score = average(clarity, depth, structure, overall)
- Add one short tip (<=12 words) to help improve
- Use personality tone matching persona strictly

Return JSON:
{{
 "index": {idx+1},
 "question": "{qlist[idx]}",
 "answer": "{ans}",
 "verdict": "...",
 "points": ...,
 "comment": "...",
 "tip": "...",
 "scores": {{"Clarity": 1-5, "Depth": 1-5, "Structure": 1-5, "Overall": 1-5}}
}}
"""
                with st.spinner("Evaluating..."):
                    raw = ask_openai_json(grading_prompt)
                try:
                    match = re.search(r"\{[\s\S]*\}", raw)
                    data = json.loads(match.group(0)) if match else {}
                except Exception:
                    data = {
                        "verdict": "in-between",
                        "points": 0.5,
                        "comment": "Unclear structure or detail.",
                        "tip": "Be more specific.",
                        "scores": {
                            "Clarity": 3,
                            "Depth": 3,
                            "Structure": 3,
                            "Overall": 3,
                        },
                    }

                st.session_state.answers.append(ans)
                st.session_state.grading.append(data)

                verdict = data.get("verdict", "in-between")
                pts = data.get("points", 0.5)
                comment = data.get("comment", "")
                tip = data.get("tip", "")
                sc = data.get("scores", {})

                ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
                COLOR = {"good": "green", "in-between": "orange", "bad": "red"}

                st.markdown(
                    f"<span style='color:{COLOR.get(verdict)};font-weight:700'>{ICON.get(verdict)} {verdict.upper()} — +{pts} pts</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Comment:** {comment}")
                st.markdown(f"**Tip:** {tip}")
                st.markdown(
                    f"**Scores:** Clarity: {sc.get('Clarity',0)}/5 · Depth: {sc.get('Depth',0)}/5 · Structure: {sc.get('Structure',0)}/5 · Overall: {sc.get('Overall',0)}/5"
                )
                st.session_state.current_q += 1
                st.experimental_rerun()

    else:
        # Final summary
        st.subheader("🏁 Final Evaluation")
        df = []
        total_pts = 0
        for g, q, a in zip(
            st.session_state.grading, st.session_state.q10, st.session_state.answers
        ):
            w = round(
                sum(
                    [
                        g["scores"].get(k, 0)
                        for k in ["Clarity", "Depth", "Structure", "Overall"]
                    ]
                )
                / 4,
                2,
            )
            total_pts += g.get("points", 0)
            df.append({"Question": q, "Answer": a, "Score": w, "Tip": g.get("tip", "")})

        ready_color = "red"
        readiness = "Not Ready for Interview"
        if total_pts >= 7:
            ready_color, readiness = "green", "Ready for Interview"
        elif total_pts >= 3:
            ready_color, readiness = "orange", "Almost Ready for Interview"

        st.markdown(
            f"<h3 style='color:{ready_color}'>{readiness} — Total Points: {total_pts}/10</h3>",
            unsafe_allow_html=True,
        )
        st.table(df)
