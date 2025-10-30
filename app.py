import os
import json, io, datetime
from typing import List, Dict
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ---------- Bootstrapping ----------
st.set_page_config(page_title="PrepMate", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ No OpenAI API key found in .env or Streamlit Secrets (OPENAI_API_KEY)."
    )
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

persona_guides = {
    "Neutral": "Professional, concise, one question at a time.",
    "Friendly coach": "Supportive tone, encourages reflection, gives hints.",
    "Strict bar-raiser": "Terse, challenging, asks for evidence and metrics.",
}


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
    persona = st.selectbox(
        "Interviewer persona",
        ["Neutral", "Friendly coach", "Strict bar-raiser"],
        index=0,
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
st.title("PrepMate")
st.caption(
    "Your personal AI interview practice companion — generate questions, get STAR feedback, and track progress."
)


# ---------- Inputs ----------
topic = st.text_area(
    "What do you want to practice?",
    placeholder="e.g., SQL joins, system design, behavioral STAR…",
    height=90,
)

with st.expander("📄 Optional context"):
    jd_file = st.file_uploader(
        "Upload Job Description (.txt or .md)", type=["txt", "md"]
    )
    job_desc_manual = st.text_area("Or paste job description", height=140)
    resume = st.text_area("Your resume bullets (paste text)", height=120)


def _normalize(s: str) -> str:
    return (s or "").replace("\r\n", "\n").strip()


job_desc = ""
if jd_file is not None:
    if getattr(jd_file, "size", 0) > 200_000:
        st.warning("JD file is too large (>200 KB). Please paste key parts instead.")
    else:
        raw = jd_file.read()
        try:
            job_desc = raw.decode("utf-8")
        except UnicodeDecodeError:
            job_desc = raw.decode("cp1252", errors="ignore")
        job_desc = job_desc[:10000]

if not job_desc:
    job_desc = job_desc_manual

topic = _normalize(topic)
job_desc = _normalize(job_desc)
resume = _normalize(resume)


def too_long(s: str, limit=5000) -> bool:
    return len(s) > limit


if too_long(topic) or too_long(job_desc) or too_long(resume):
    st.error("Inputs are too long (limit ~5000 chars per field). Trim and try again.")
    st.stop()


# ---------- Buttons ----------
colL, colR = st.columns([2, 1])
with colL:
    gen_btn = st.button("🧠 Generate Questions", use_container_width=True)
with colR:
    st.write("")


# ---------- Safe ask (supports mock mode) ----------
def safe_ask(prompt: str) -> str:
    if USE_MOCK:
        return (
            "Technical:\n"
            "1) Explain the difference between INNER and LEFT JOIN.\n"
            "2) When would you use ROW_NUMBER() vs RANK()?\n"
            "Behavioral:\n"
            "3) Tell me about a time you handled conflicting priorities."
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
You are an experienced interviewer.

Role: {role}
Difficulty: {difficulty}
Guideline: {guideline}
Interviewer persona guideline: {persona_guides[persona]}
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
                except:
                    pass
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                st.info("Tip: Turn on Mock Mode if you're out of quota.")

st.divider()


# ---------- Critique ----------
st.subheader("🧩 Critique My Answer")
user_answer = st.text_area("Paste your answer for feedback:", height=140)
crit_btn = st.button("💬 Get Feedback")

if crit_btn:
    if not user_answer.strip():
        st.warning("Please paste your answer above.")
    elif misuse_guard(user_answer):
        st.error("This content looks unsafe or out of scope.")
    else:
        critique_prompt = f"""
You are an expert interviewer.

Interviewer persona guideline: {persona_guides[persona]}
Evaluate the candidate's answer using STAR.

Return ONLY a compact JSON object with this schema:
{{
  "scores": {{"Clarity": 1-5, "Depth": 1-5, "Structure": 1-5}},
  "overall_comment": "one paragraph, <= 80 words",
  "improvement_tip": "one concrete tip, one sentence"
}}

Candidate answer:
{user_answer}
"""
        with st.spinner("Scoring your answer…"):
            try:
                raw = safe_ask(critique_prompt)
                json_str = raw.strip()
                start = json_str.find("{")
                end = json_str.rfind("}")
                if start != -1 and end != -1:
                    json_str = json_str[start : end + 1]
                data = json.loads(json_str)

                scores = data.get("scores", {})
                clarity = int(scores.get("Clarity", 0))
                depth = int(scores.get("Depth", 0))
                structure = int(scores.get("Structure", 0))
                avg = (
                    round((clarity + depth + structure) / 3, 2)
                    if any([clarity, depth, structure])
                    else 0
                )

                st.subheader("Scores")
                st.write(
                    f"- **Clarity:** {clarity}/5\n- **Depth:** {depth}/5\n- **Structure:** {structure}/5\n- **Overall:** **{avg}/5**"
                )
                st.subheader("Comment")
                st.write(data.get("overall_comment", ""))
                st.subheader("Improvement Tip")
                st.write("• " + data.get("improvement_tip", ""))

                st.session_state.history.append({"type": "critique", "text": raw})
                st.caption(
                    f"💰 Estimated prompt cost (rough): ${estimate_cost(len(critique_prompt), model):.5f}"
                )
                try:
                    st.toast("📝 Structured feedback ready!", icon="📝")
                except:
                    pass
            except Exception as e:
                st.error("Parsing error — showing raw output:")
                st.write(raw)


# ---------- LEVEL MODE ----------
st.divider()
st.subheader("🏔️ Level Mode (1–5)")

if "lvl_current" not in st.session_state:
    st.session_state.lvl_current = 1
if "lvl_questions" not in st.session_state:
    st.session_state.lvl_questions = []
if "lvl_feedback" not in st.session_state:
    st.session_state.lvl_feedback = []
if "lvl_status" not in st.session_state:
    st.session_state.lvl_status = ""

LEVEL_GUIDES = {
    1: "Very basic, foundational questions.",
    2: "Basic applied questions. Simple scenarios.",
    3: "Intermediate reasoning. Tradeoffs.",
    4: "Advanced problem-solving and edge cases.",
    5: "Expert-level open-ended and strategic questions.",
}


def gen_level_questions(level: int) -> list[str]:
    guide = LEVEL_GUIDES.get(level, LEVEL_GUIDES[5])
    lvl_prompt = f"""
Generate 5 interview questions for Level {level} of 5.

Role: {role}
Topic: {topic or 'General readiness'}
Difficulty guide: {guide}
Interviewer persona: {persona_guides[persona]}

Requirements:
- 5 concise questions, each <25 words.
- Each on a new line.
"""
    raw = safe_ask(lvl_prompt)
    qs = [q.strip(" -•\t") for q in raw.split("\n") if q.strip()]
    return qs[:5] if len(qs) >= 5 else (qs + ["(placeholder)"] * (5 - len(qs)))


def grade_level(level: int, questions: list[str], answers: list[str]) -> dict:
    grading_prompt = f"""
You are an expert interviewer.

Interviewer persona guideline: {persona_guides[persona]}

Evaluate answers for Level {level}.
For each question, output JSON with:
"q","a","correct":true/false,"comment":"one-sentence reason"

Questions:
{json.dumps(questions)}

Answers:
{json.dumps(answers)}

Return ONLY JSON:
{{"items":[{{"q":"...","a":"...","correct":true,"comment":"..."}}, ...]}}
"""
    raw = safe_ask(grading_prompt)
    js = raw.strip()
    s, e = js.find("{"), js.rfind("}")
    if s != -1 and e != -1:
        js = js[s : e + 1]
    try:
        data = json.loads(js)
    except Exception:
        data = {
            "items": [
                {"q": q, "a": a, "correct": False, "comment": "Parse error."}
                for q, a in zip(questions, answers)
            ]
        }
    return data


# Controls
colA, colB = st.columns([1, 1])
with colA:
    st.markdown(f"**Current level:** {st.session_state.lvl_current}")
with colB:
    if st.button("🔁 Reset Levels"):
        (
            st.session_state.lvl_current,
            st.session_state.lvl_questions,
            st.session_state.lvl_feedback,
            st.session_state.lvl_status,
        ) = (1, [], [], "")
        st.experimental_rerun()

if not st.session_state.lvl_questions:
    if st.button(
        "▶️ Start Level" if st.session_state.lvl_current == 1 else "▶️ Start Next Level"
    ):
        st.session_state.lvl_questions = gen_level_questions(
            st.session_state.lvl_current
        )
        st.session_state.lvl_feedback, st.session_state.lvl_status = [], ""
        st.experimental_rerun()

if st.session_state.lvl_questions:
    st.markdown("### Level Questions")
    with st.form(key=f"lvl_form_{st.session_state.lvl_current}"):
        lvl_answers = []
        for i, q in enumerate(st.session_state.lvl_questions):
            st.markdown(f"**Q{i+1}.** {q}")
            ans = st.text_area(f"Your answer to Q{i+1}", key=f"ans_{i}", height=80)
            lvl_answers.append(ans.strip())
        submitted = st.form_submit_button("✅ Submit Level")
    if submitted:
        result = grade_level(
            st.session_state.lvl_current, st.session_state.lvl_questions, lvl_answers
        )
        items = result.get("items", [])
        all_correct = all(x.get("correct") for x in items)
        st.session_state.lvl_feedback, st.session_state.lvl_status = items, (
            "pass" if all_correct else "fail"
        )
        st.experimental_rerun()

if st.session_state.lvl_feedback:
    st.markdown("### Results")
    if st.session_state.lvl_status == "pass":
        st.success("🎉 Level passed! All answers correct.")
        if st.session_state.lvl_current < 5:
            if st.button("➡️ Continue to next level"):
                st.session_state.lvl_current += 1
                (
                    st.session_state.lvl_questions,
                    st.session_state.lvl_feedback,
                    st.session_state.lvl_status,
                ) = ([], [], "")
                st.experimental_rerun()
        else:
            st.balloons()
            st.info("🏁 You completed Level 5 — congratulations!")
    else:
        st.error("❌ Level failed. See feedback below.")
    for i, it in enumerate(st.session_state.lvl_feedback, 1):
        icon = "✅" if it.get("correct") else "❌"
        st.markdown(f"**{icon} Q{i}. {it.get('q','')}**")
        st.markdown(f"- *Your answer:* {it.get('a','')}")
        st.markdown(f"- *Comment:* {it.get('comment','')}")
        st.markdown("---")
    if st.session_state.lvl_status == "fail":
        if st.button("🔄 Retry this level"):
            (
                st.session_state.lvl_questions,
                st.session_state.lvl_feedback,
                st.session_state.lvl_status,
            ) = (gen_level_questions(st.session_state.lvl_current), [], "")
            st.experimental_rerun()


# ---------- History / Export / Reset ----------
st.divider()
c1, c2 = st.columns([1, 1])
with c1:
    if st.button("🧹 Clear History"):
        st.session_state.history.clear()
        st.experimental_rerun()
with c2:

    def build_md(hist):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [f"# PrepMate Session Export ({now})\n"]
        for item in hist:
            tag = "Questions" if item["type"] == "questions" else "Critique"
            parts.append(f"## {tag}\n\n{item['text']}\n")
        return "\n---\n".join(parts)

    md = build_md(st.session_state.history)
    st.download_button(
        "⬇️ Download Session (Markdown)",
        data=md.encode("utf-8"),
        file_name="prepmate_session.md",
        mime="text/markdown",
    )

st.caption("© 2025 PrepMate — Built with Streamlit & OpenAI API")
