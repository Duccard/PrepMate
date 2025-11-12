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
    st.error("❌ No OpenAI API key found. Add it to your .env file (OPENAI_API_KEY).")
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
# Personas (stronger character)
# =========================
persona_guides = {
    "Neutral": "Professional and balanced. Gives objective, straightforward feedback.",
    "Friendly coach": "Warm and supportive. Offers encouragement and constructive tips with a kind tone.",
    "Strict bar-raiser": "Demanding and exact. Values precision, confidence, and well-structured answers.",
    "Motivational mentor": "Inspiring, positive tone. Pushes you to see growth and improvement.",
    "Calm psychologist": "Analytical and empathetic. Focuses on emotional intelligence and self-awareness.",
    "Playful mock interviewer": "Uses humor and teasing to make feedback memorable but insightful.",
    "Corporate recruiter": "Polished, HR-oriented. Focuses on professionalism and communication clarity.",
    "Sarcastic Interviewer": "Witty, sharp, and slightly playful. Gives biting but honest feedback.",
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
# Helpers
# =========================
def misuse_guard(*texts: str) -> bool:
    bad = ["bypass security", "phishing", "exploit", "cheat on", "ddos"]
    return any(b in " ".join(texts).lower() for b in bad)


def ask_openai(prompt: str) -> str:
    if USE_MOCK:
        return (
            "Sure! Here are 10 interview questions:\n"
            "1. Mock Q1\n2. Mock Q2\n3. Mock Q3\n4. Mock Q4\n5. Mock Q5\n"
            "6. Mock Q6\n7. Mock Q7\n8. Mock Q8\n9. Mock Q9\n10. Mock Q10"
        )
    resp = client.responses.create(
        model=model, temperature=temperature, max_output_tokens=max_tokens, input=prompt
    )
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def ask_openai_json(prompt: str) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.3,
        top_p=1,
        max_output_tokens=max_tokens,
    )
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def extract_numbered_questions(text: str, want=10) -> List[str]:
    """
    Robustly extract up to `want` numbered questions from arbitrary model text.
    Handles preambles like 'Sure! Here are 10...' and various numbering styles.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    qs = []

    # Prefer lines that look like "1. ..." / "1) ..." / "1 - ..."
    for ln in lines:
        m = re.match(r"^\s*(\d{1,2})\s*[\.\)\-]\s*(.+)$", ln)
        if m:
            qs.append(m.group(2).strip())

    # If not enough, try lines that end with a question mark
    if len(qs) < want:
        for ln in lines:
            if ln.endswith("?"):
                qs.append(re.sub(r"^\s*(\d{1,2})\s*[\.\)\-]\s*", "", ln))

    # If still not enough, try splitting by question marks in big paragraphs
    if len(qs) < want:
        blob = " ".join(lines)
        parts = [p.strip() + "?" for p in blob.split("?") if p.strip()]
        for p in parts:
            if len(p) > 3:
                qs.append(re.sub(r"^\s*(\d{1,2})\s*[\.\)\-]\s*", "", p))

    # Deduplicate, trim, cap to want, and ensure each under 25 words-ish
    seen = set()
    cleaned = []
    for q in qs:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            cleaned.append(q)
        if len(cleaned) >= want:
            break

    # Fallback placeholders if needed
    while len(cleaned) < want:
        cleaned.append("Placeholder question — describe a relevant scenario briefly?")

    return cleaned[:want]


# =========================
# Header
# =========================
st.title("Prepmate: Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)

# =========================
# Topic & Optional Context (file-only)
# =========================
topic = st.text_area(
    "What do you want to practice?",
    placeholder="e.g. Cooking, Data analysis, Team leadership...",
    height=90,
)
jd_file = st.file_uploader(
    "📎 Optional: Upload Job Description (TXT, MD)", type=["txt", "md"]
)
job_desc = ""
if jd_file:
    try:
        job_desc = jd_file.read().decode("utf-8", errors="ignore")[:8000]
    except Exception:
        job_desc = ""

# =========================
# Generate Questions
# =========================
if st.button("🧠 Generate Questions"):
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        prompt = f"""
You are an interviewer creating realistic {difficulty.lower()}-level interview questions.

Persona: {persona_guides[persona]}
Topic: {topic}
Job Description: {job_desc or 'N/A'}

Generate exactly 10 questions mixing technical and behavioral aspects.
RETURN ONLY the numbered list (1..10). No preamble, no closing text.
Each question under 25 words. Number them 1 to 10.
"""
        with st.spinner("Generating questions..."):
            out = ask_openai(prompt)
        q10 = extract_numbered_questions(out, want=10)

        st.session_state.q10 = q10
        st.session_state.current_q = 0
        st.session_state.answers = []
        st.session_state.grading = []
        st.success(
            "✅ 10 questions ready! Click “Submit Answer” to get scoring and feedback."
        )

# =========================
# Interactive Q&A Flow (one-by-one)
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
                # Grade single answer with persona tone; fair, not hyper-strict
                grading_prompt = f"""
You are an expert interviewer using persona: {persona_guides[persona]}.
Evaluate the candidate's single answer to ONE question.

Rules:
- If the answer contains phrases like "don't care", "dont care", "don't know", "dont know"
  or is blank → verdict "bad", points 0, comment "No answer provided.", scores all 1.
- Otherwise be FAIR (not overly harsh). Use:
  verdict ∈ {{good, in-between, bad}} ; points ∈ {{1, 0.5, 0}}
- Provide a short, specific tip (<= 12 words) aligned with the persona's tone.
- Scores are integers 1..5 for Clarity, Depth, Structure, Overall.
- Keep "answer" in output <= 18 words (shortened summary of what the candidate wrote).

Return ONLY JSON:
{{
 "index": {idx+1},
 "question": "{qlist[idx]}",
 "answer": "{ans.replace('"','\\\"')}",
 "verdict": "good|in-between|bad",
 "points": 1|0.5|0,
 "comment": "one sentence, <= 18 words",
 "tip": "one short tip, <= 12 words",
 "scores": {{"Clarity": 1-5, "Depth": 1-5, "Structure": 1-5, "Overall": 1-5}}
}}
"""
                with st.spinner("Evaluating..."):
                    raw = ask_openai_json(grading_prompt)

                # Parse JSON robustly
                try:
                    m = re.search(r"\{[\s\S]*\}", raw)
                    data = json.loads(m.group(0)) if m else {}
                except Exception:
                    data = {
                        "index": idx + 1,
                        "question": qlist[idx],
                        "answer": ans[:80],
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

                # Strict zeroing for don't-know/don't-care even if model missed it
                if (
                    re.search(r"\b(don['’]?t\s+(know|care))\b", ans.lower())
                    or ans.strip() == ""
                ):
                    data["verdict"] = "bad"
                    data["points"] = 0
                    data["comment"] = "No answer provided."
                    data["scores"] = {
                        "Clarity": 1,
                        "Depth": 1,
                        "Structure": 1,
                        "Overall": 1,
                    }
                    data.setdefault("tip", "Try a concrete example.")

                st.session_state.answers.append(ans)
                st.session_state.grading.append(data)

                # Show feedback
                verdict = data.get("verdict", "in-between")
                pts = data.get("points", 0.5)
                comment = data.get("comment", "")
                tip = data.get("tip", "")
                sc = data.get("scores", {})

                ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
                COLOR = {"good": "green", "in-between": "orange", "bad": "red"}

                st.markdown(
                    f"<span style='color:{COLOR.get(verdict)};font-weight:700'>"
                    f"{ICON.get(verdict)} {verdict.upper()} — +{pts} pts</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Comment:** {comment}")
                st.markdown(f"**Tip:** {tip}")
                st.markdown(
                    f"**Scores:** Clarity: {sc.get('Clarity',0)}/5 · Depth: {sc.get('Depth',0)}/5 · "
                    f"Structure: {sc.get('Structure',0)}/5 · Overall: {sc.get('Overall',0)}/5"
                )

                # Advance to next question automatically after feedback is shown
                st.session_state.current_q += 1
                st.rerun()

    else:
        # Final summary
        st.subheader("🏁 Final Evaluation")
        rows = []
        total_pts = 0.0
        for g, q, a in zip(
            st.session_state.grading, st.session_state.q10, st.session_state.answers
        ):
            sc = g.get("scores", {})
            weighted = round(
                (
                    sc.get("Clarity", 0)
                    + sc.get("Depth", 0)
                    + sc.get("Structure", 0)
                    + sc.get("Overall", 0)
                )
                / 4,
                2,
            )
            total_pts += float(g.get("points", 0))
            rows.append(
                {
                    "Question": q,
                    "Answer": a,
                    "Overall score (1–5)": weighted,
                    "Tip": g.get("tip", ""),
                }
            )

        readiness = "Not Ready for Interview"
        color = "red"
        if total_pts >= 7:
            readiness, color = "Ready for Interview", "green"
        elif total_pts >= 3:
            readiness, color = "Almost Ready for Interview", "orange"

        st.markdown(
            f"<h3 style='color:{color}'>"
            f"{readiness} — Total Points: {total_pts:.1f}/10"
            f"</h3>",
            unsafe_allow_html=True,
        )
        st.table(rows)
