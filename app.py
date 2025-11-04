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
        "❌ No OpenAI API key found in .env or Streamlit Secrets (OPENAI_API_KEY)."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Session
# =========================
if "history" not in st.session_state:
    st.session_state.history: List[Dict] = []
if "q10" not in st.session_state:
    st.session_state.q10: List[str] = []
if "last_questions" not in st.session_state:
    st.session_state.last_questions = ""

# =========================
# Helpers
# =========================
difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions that involve reasoning and real examples.",
    "Hard": "Ask complex, open-ended or scenario-based questions testing depth and creativity.",
}

persona_guides = {
    "Neutral": "Professional, concise, and unbiased.",
    "Friendly coach": "Supportive tone, encourages reflection, offers gentle hints.",
    "Strict bar-raiser": "Challenging, expects precise answers and metrics.",
    "Motivational mentor": "Inspiring tone, focuses on growth and encouragement.",
    "Calm psychologist": "Analytical and empathetic, probes for self-awareness.",
    "Playful mock interviewer": "Light tone, humorous but insightful feedback.",
    "Corporate recruiter": "Evaluates professionalism, fit, and clarity in communication.",
}


def misuse_guard(*texts: str) -> bool:
    lower = " ".join(t or "" for t in texts).lower()
    flags = ["cheat on", "bypass security", "malware", "phishing", "exploit", "ddos"]
    return any(f in lower for f in flags)


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    tokens = max(1, chars // 4)  # very rough char->token
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


def ask_openai_json(prompt: str, *, model: str, max_tokens: int) -> str:
    """
    Strict JSON helper for grading.
    - Use JSON mode only on models that support it (gpt-4.1 family is a safe bet).
    - Otherwise/fallback: low-temp normal response; rely on our JSON-only prompt + parser.
    """

    def _supports_json_mode(m: str) -> bool:
        m = (m or "").lower()
        return "gpt-4.1" in m  # gpt-4.1 and gpt-4.1-mini

    # Try JSON mode first if supported
    if _supports_json_mode(model):
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
            pass  # fall through

    # Fallback normal call (still low temp)
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


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    persona = st.selectbox("Interviewer persona", list(persona_guides.keys()), index=0)

    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p sampling", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 100, 2000, 800, 50)
    # More headroom for grading JSON:
    grade_tokens = max(1200, int(max_tokens * 1.2))
    USE_MOCK = st.toggle("🧪 Mock Mode (no API calls)", False)
    st.caption("🔒 Basic misuse guard is active.")

# =========================
# Header
# =========================
st.title("PrepMate")
st.caption(
    "Your AI interview practice companion — 10 questions per round, per-question verdicts, and weighted scoring."
)

# =========================
# Inputs
# =========================
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

# =========================
# Buttons
# =========================
colL, colR = st.columns([2, 1])
with colL:
    gen_btn = st.button("🧠 Generate 10 Questions", use_container_width=True)
with colR:
    st.write("")


# =========================
# Mock-safe ask
# =========================
def safe_ask(prompt: str) -> str:
    if USE_MOCK:
        if "Return EXACTLY 10 questions" in prompt:
            return "\n".join(
                [
                    f"{i}. Mock question {i} about {topic or 'the role'}"
                    for i in range(1, 11)
                ]
            )
        if '"items"' in prompt and "verdict" in prompt:
            items = []
            for i in range(1, 11):
                items.append(
                    {
                        "index": i,
                        "question": f"Mock question {i}",
                        "answer": "Mock answer",
                        "verdict": "in-between",
                        "points": 0.5,
                        "comment": "Decent, but missing specifics.",
                        "scores": {
                            "Clarity": 3,
                            "Depth": 2,
                            "Structure": 3,
                            "Overall": 3,
                        },
                    }
                )
            return json.dumps({"items": items}, ensure_ascii=False)
        return "Mock output."
    return ask_openai(
        prompt, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )


# =========================
# Generate 10 merged questions (stored, not printed twice)
# =========================
if gen_btn:
    if misuse_guard(topic, job_desc, resume):
        st.error("This looks unsafe or out of scope. Please rephrase.")
    else:
        guideline = difficulty_tips[difficulty]
        prompt = f"""
You are an experienced interviewer.

Difficulty: {difficulty}
Guideline: {guideline}
Interviewer persona guideline: {persona_guides[persona]}
Focus topic(s): {topic or 'General interview readiness'}
Job Description: {job_desc or 'N/A'}
Candidate Resume Bullets: {resume or 'N/A'}

Task:
Create a single set of interview questions that mixes technical and behavioral aspects
relevant to the topic. Return EXACTLY 10 questions, numbered 1 to 10. One per line.
Keep each question under 25 words and realistic.

Example format:
1. ...
2. ...
...
10. ...
"""
        with st.spinner("Generating questions…"):
            try:
                out = safe_ask(prompt)

                # Parse 10 numbered lines
                lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                q10 = []
                for ln in lines:
                    # Accept "1. ..." / "1) ..." / "1 - ..."
                    head = ln.split(maxsplit=1)[0]
                    if head.rstrip(".)").isdigit():
                        q = (
                            ln.split(".", 1)[-1]
                            if "." in ln
                            else ln.split(")", 1)[-1] if ")" in ln else ln[len(head) :]
                        )
                        q10.append(q.strip(" -–•\t"))
                q10 = q10[:10]
                while len(q10) < 10:
                    q10.append("(placeholder)")

                st.session_state.q10 = q10
                st.session_state.last_questions = "\n".join(
                    [f"{i+1}. {q}" for i, q in enumerate(q10)]
                )

                st.caption(
                    f"💰 Estimated prompt cost (rough): ${estimate_cost(len(prompt), model):.5f}"
                )
                try:
                    st.toast("✅ 10 questions ready!", icon="✅")
                except:
                    pass
            except Exception as e:
                st.error(f"OpenAI error: {e}")
                st.info("Tip: Turn on Mock Mode if you're out of quota.")

# =========================
# Answer & Feedback
# =========================
st.divider()
st.subheader("🧩 Answer & Get Feedback (10 questions)")

if not st.session_state.q10:
    st.info("Generate questions first, then answer below.")
else:
    q10 = st.session_state.q10

    with st.form(key="answers_form_10"):
        answers = []
        for i, q in enumerate(q10, 1):
            st.markdown(f"**{i}. {q}**")
            a = st.text_area(
                f"Your answer {i}",
                key=f"ans10_{i}",
                height=80,
                placeholder="Type your answer here…",
            )
            answers.append((q, (a or "").strip()))
        crit_btn = st.form_submit_button("💬 Submit Answers for Grading")

    if crit_btn:
        if any(misuse_guard(a) for _, a in answers):
            st.error(
                "Some answers look unsafe or out of scope. Please revise and resubmit."
            )
        else:
            questions = [q for q, _ in answers]
            safe_answers = [(a or "").replace('"', "'") for _, a in answers]

            grading_prompt = f"""
You are an expert interviewer and strict grader.

Interviewer persona guideline: {persona_guides[persona]}
Topic focus: {topic or 'General readiness'}

You will receive 10 questions and the candidate's 10 answers (aligned by index).

Rules per item:
- If the candidate answer is empty/blank, set:
  "answer": "", "verdict": "bad", "points": 0,
  "comment": "No answer provided.",
  "scores": {{"Clarity":1,"Depth":1,"Structure":1,"Overall":1}}
- Keep "answer" <= 12 words (shortened from user answer).
- Keep "comment" <= 12 words, one sentence, specific.
- Use ONLY these verdicts: "good", "in-between", "bad".
- Points: good=1, in-between=0.5, bad=0.
- Scores are integers 1..5.
- Return EXACTLY 10 items, index 1..10.
- Output ONLY valid JSON (no prose, no code fences).
- Use double quotes for strings; escape internal quotes as \\".
- Do not use smart quotes.

Return JSON with this schema:
{{
  "items": [
    {{
      "index": 1,
      "question": "...",
      "answer": "...",
      "verdict": "good|in-between|bad",
      "points": 1|0.5|0,
      "comment": "...",
      "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
    }},
    ... 10 items total ...
  ]
}}

Questions:
{json.dumps(questions, ensure_ascii=False)}

Answers:
{json.dumps(safe_answers, ensure_ascii=False)}
"""

            with st.spinner("Grading your 10 answers…"):
                # First try: strict JSON mode
                raw = ask_openai_json(
                    grading_prompt, model=model, max_tokens=grade_tokens
                )

            def _parse_items(raw_text: str):
                js = raw_text.strip()
                # Keep only innermost JSON block if wrappers exist
                match = re.search(r"\{[\s\S]*\}", js)
                if match:
                    js = match.group(0)
                # Clean trailing commas, but DO NOT touch apostrophes
                js = re.sub(r",(\s*[}\]])", r"\1", js)
                try:
                    data = json.loads(js)
                    return data.get("items", []), js
                except json.JSONDecodeError:
                    return [], js

            items, cleaned = _parse_items(raw)

            # Retry once if fewer than 10 items
            if len(items) < 10:
                retry_prompt = (
                    grading_prompt
                    + """

REMINDER:
- Return ONLY a JSON object with key "items" that contains EXACTLY 10 entries.
- No prose, no code fences.
- Each item must include: index, question, answer, verdict, points, comment, scores.
"""
                )
                raw = ask_openai_json(
                    retry_prompt, model=model, max_tokens=grade_tokens
                )
                items, cleaned = _parse_items(raw)

            if not items:
                st.warning("Could not parse results. Showing raw output below:")
                st.code(raw[:2000])
            else:
                if len(items) < 10:
                    st.warning(
                        f"Received only {len(items)}/10 graded items. Showing what we have."
                    )

                ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
                COLOR = {"good": "green", "in-between": "orange", "bad": "red"}

                total_points = 0.0
                weighted_scores = []

                st.markdown("### Results")
                for it in items:
                    idx = it.get("index", "?")
                    verdict = (it.get("verdict") or "").lower()
                    points = float(it.get("points", 0))
                    comment = it.get("comment", "")
                    sc = it.get("scores", {}) or {}
                    clarity = float(sc.get("Clarity", 0))
                    depth = float(sc.get("Depth", 0))
                    structure = float(sc.get("Structure", 0))
                    overall = float(sc.get("Overall", 0))
                    weighted = round((clarity + depth + structure + 2 * overall) / 5, 2)

                    total_points += points
                    weighted_scores.append(weighted)

                    icon = ICON.get(verdict, "❔")
                    color = COLOR.get(verdict, "gray")

                    st.markdown(
                        f"<span style='color:{color}; font-weight:700'>{icon} Q{idx} — {verdict.upper()} · +{points} pts</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**Question:** {it.get('question','')}")
                    st.markdown(f"**Your answer:** {it.get('answer','')}")
                    st.markdown(f"**Comment:** {comment}")
                    st.markdown(
                        f"**Scores:** Clarity: {int(clarity)}/5 · Depth: {int(depth)}/5 · "
                        f"Structure: {int(structure)}/5 · Overall: {int(overall)}/5 "
                        f"(weighted: **{weighted}/5**)"
                    )
                    st.markdown("---")

                avg_weighted = (
                    round(sum(weighted_scores) / len(weighted_scores), 2)
                    if weighted_scores
                    else 0.0
                )
                st.success(
                    f"🏁 **Round summary:** {total_points}/10 points · "
                    f"Average weighted score: **{avg_weighted}/5** (Overall counted double)"
                )

                # Log to history
                summary_block = "\n".join(
                    [
                        f"Q{it.get('index')}: verdict={it.get('verdict')}, pts={it.get('points')}, "
                        f"scores={it.get('scores')}, comment={it.get('comment')}"
                        for it in items
                    ]
                )
                st.session_state.history.append(
                    {"type": "critique", "text": summary_block}
                )

# =========================
# History / Export / Reset
# =========================
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
            tag = "Grading" if item["type"] == "critique" else "Questions/Answers"
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
