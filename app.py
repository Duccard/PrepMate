import os
import re
import json
import datetime
from typing import List, Dict, Any

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
# Helpers
# =========================
persona_guides = {
    "Neutral": "Professional, concise, and unbiased.",
    "Friendly coach": "Supportive tone, encourages reflection, offers gentle hints.",
    "Strict bar-raiser": "Challenging, expects precise answers and metrics.",
    "Motivational mentor": "Inspiring tone, focuses on growth and encouragement.",
    "Calm psychologist": "Analytical and empathetic, probes for self-awareness.",
    "Playful mock interviewer": "Light tone, humorous but insightful feedback.",
    "Corporate recruiter": "Evaluates professionalism, fit, and clarity in communication.",
}

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
    tokens = max(1, chars // 4)  # rough char->token
    rates = {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "gpt-4o": 5.00 / 1_000_000,
        "gpt-4.1": 5.00 / 1_000_000,
        "gpt-4.1-mini": 0.30 / 1_000_000,
    }
    return tokens * rates.get(model, 0.15 / 1_000_000)


def _supports_json_mode(m: str) -> bool:
    m = (m or "").lower()
    return "gpt-4.1" in m  # JSON mode supported for gpt-4.1 / gpt-4.1-mini


def ask_openai_text(
    prompt: str, *, model: str, temperature: float, top_p: float, max_tokens: int
) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
    )
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def ask_openai_json(prompt: str, *, model: str, max_tokens: int) -> str:
    # Try strict JSON mode if available
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
            pass
    # Fallback to normal call (still low temp)
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


def clean_and_parse_json(raw_text: str) -> Any:
    """Extract innermost JSON and parse. Also remove trailing commas."""
    js = (raw_text or "").strip()
    m = re.search(r"\{[\s\S]*\}", js)
    if m:
        js = m.group(0)
    # Remove trailing commas BEFORE closing } or ]
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    # Don't alter apostrophes; the prompt enforces double quotes
    return json.loads(js)


# =========================
# Session (single-question flow)
# =========================
def ensure_state():
    s = st.session_state
    s.setdefault("history", [])  # past exports
    s.setdefault("topic", "")
    s.setdefault("difficulty", "Medium")
    s.setdefault("persona", "Neutral")
    s.setdefault("model", "gpt-4o-mini")
    s.setdefault("temperature", 0.7)
    s.setdefault("top_p", 1.0)
    s.setdefault("max_tokens", 800)
    s.setdefault("grade_tokens", 1200)
    s.setdefault("USE_MOCK", False)

    # round state
    s.setdefault("questions", [])  # list[str], len==10 when generated
    s.setdefault("cur_idx", 0)  # 0..9
    s.setdefault("answers", [""] * 10)  # user answers
    s.setdefault("results", [None] * 10)  # per-Q grading dicts
    s.setdefault("total_points", 0.0)  # sum of per-Q points
    s.setdefault("done", False)  # finished 10/10


ensure_state()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    st.session_state.difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value=st.session_state.difficulty
    )
    st.session_state.persona = st.selectbox(
        "Interviewer persona",
        list(persona_guides.keys()),
        index=list(persona_guides.keys()).index(st.session_state.persona),
    )

    st.markdown("### ⚙️ Model")
    st.session_state.model = st.selectbox(
        "Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"],
        index=["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"].index(
            st.session_state.model
        ),
    )
    st.session_state.temperature = st.slider(
        "Temperature", 0.0, 1.5, float(st.session_state.temperature), 0.1
    )
    st.session_state.top_p = st.slider(
        "Top-p sampling", 0.0, 1.0, float(st.session_state.top_p), 0.05
    )
    st.session_state.max_tokens = st.slider(
        "Max output tokens", 100, 2000, int(st.session_state.max_tokens), 50
    )
    st.session_state.grade_tokens = max(1200, int(st.session_state.max_tokens * 1.2))
    st.session_state.USE_MOCK = st.toggle(
        "🧪 Mock Mode (no API calls)", st.session_state.USE_MOCK
    )
    st.caption("🔒 Basic misuse guard is active.")

# =========================
# Header
# =========================
st.title("PrepMate")
st.caption(
    "Single-question flow: answer → instant feedback → next. 10 questions total, then a graded summary."
)

# =========================
# Inputs (topic + optional context)
# =========================
topic = st.text_area(
    "What do you want to practice?",
    value=st.session_state.topic,
    placeholder="e.g., SQL joins, system design, behavioral STAR…",
    height=90,
)
with st.expander("📄 Optional context"):
    jd_file = st.file_uploader(
        "Upload Job Description (.txt or .md)", type=["txt", "md"]
    )
    job_desc_manual = st.text_area("Or paste job description", height=140, value="")
    resume = st.text_area("Your resume bullets (paste text)", height=120, value="")


def _norm(s: str) -> str:
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

topic = _norm(topic)
job_desc = _norm(job_desc)
resume = _norm(resume)


def too_long(s: str, limit=5000) -> bool:
    return len(s) > limit


if any([too_long(topic), too_long(job_desc), too_long(resume)]):
    st.error("Inputs are too long (limit ~5000 chars per field). Trim and try again.")
    st.stop()


# =========================
# Generation (10 questions upfront)
# =========================
def safe_generate_questions() -> List[str]:
    if st.session_state.USE_MOCK:
        return [
            f"{i}. Mock question {i} about {topic or 'the role'}" for i in range(1, 11)
        ]

    guideline = difficulty_tips[st.session_state.difficulty]
    prompt = f"""
You are an experienced interviewer.

Difficulty: {st.session_state.difficulty}
Guideline: {guideline}
Interviewer persona guideline: {persona_guides[st.session_state.persona]}
Focus topic(s): {topic or 'General interview readiness'}
Job Description: {job_desc or 'N/A'}
Candidate Resume Bullets: {resume or 'N/A'}

Task:
Create a single set of interview questions that mixes technical and behavioral aspects relevant to the topic.
Return EXACTLY 10 questions, numbered 1 to 10. One per line.
Keep each question under 25 words and realistic.

Example format:
1. ...
2. ...
...
10. ...
"""
    out = ask_openai_text(
        prompt,
        model=st.session_state.model,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        max_tokens=st.session_state.max_tokens,
    )
    # Parse 10 numbered lines
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    q10 = []
    for ln in lines:
        head = ln.split(maxsplit=1)[0]
        if head.rstrip(".)").isdigit():
            q = (
                ln.split(".", 1)[-1]
                if "." in ln
                else (ln.split(")", 1)[-1] if ")" in ln else ln[len(head) :])
            )
            q10.append(q.strip(" -–•\t"))
    q10 = q10[:10]
    while len(q10) < 10:
        q10.append("(placeholder)")
    return q10


# Controls
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("🧠 Start / Regenerate 10 Questions", use_container_width=True):
        if misuse_guard(topic, job_desc, resume):
            st.error("This looks unsafe or out of scope. Please rephrase.")
        else:
            st.session_state.topic = topic
            st.session_state.questions = safe_generate_questions()
            st.session_state.cur_idx = 0
            st.session_state.answers = [""] * 10
            st.session_state.results = [None] * 10
            st.session_state.total_points = 0.0
            st.session_state.done = False
            try:
                st.toast("✅ 10 questions ready!", icon="✅")
            except:
                pass
with c2:
    if st.button("🔁 Reset Round", use_container_width=True):
        st.session_state.questions = []
        st.session_state.cur_idx = 0
        st.session_state.answers = [""] * 10
        st.session_state.results = [None] * 10
        st.session_state.total_points = 0.0
        st.session_state.done = False
        st.experimental_rerun()

st.divider()

# =========================
# Single-question flow UI
# =========================
if not st.session_state.questions:
    st.info("Click **Start / Regenerate 10 Questions** to begin.")
else:
    idx = st.session_state.cur_idx
    q = st.session_state.questions[idx]
    st.markdown(f"### Question {idx+1}/10")
    st.write(q)

    # input
    st.session_state.answers[idx] = st.text_area(
        "Your answer",
        key=f"ans_{idx}",
        value=st.session_state.answers[idx],
        height=130,
        placeholder="Type your answer here…",
    )

    # ----- grader -----
    def grade_one(question: str, answer: str) -> Dict:
        if st.session_state.USE_MOCK:
            # simple deterministic mock
            verdict = (
                "good"
                if len(answer.strip()) > 40
                else ("in-between" if len(answer.strip()) > 10 else "bad")
            )
            overall = 5 if verdict == "good" else (3 if verdict == "in-between" else 1)
            clarity = overall
            depth = overall
            structure = overall
            tip = (
                "Excellent answer, keep it up!"
                if verdict == "good"
                else (
                    "Add a concrete example and metrics."
                    if verdict == "in-between"
                    else "State the problem, your actions, and the result."
                )
            )
            return {
                "index": idx + 1,
                "question": question,
                "answer": (answer or "")[:80],
                "verdict": verdict,
                "comment": "Mock grade.",
                "tip": tip,
                "scores": {
                    "Clarity": clarity,
                    "Depth": depth,
                    "Structure": structure,
                    "Overall": overall,
                },
            }

        safe_answer = (answer or "").replace('"', '\\"')
        grading_prompt = f"""
You are an expert interviewer and strict grader.

Interviewer persona guideline: {persona_guides[st.session_state.persona]}
Topic focus: {st.session_state.topic or 'General readiness'}

Grade ONE answer for ONE question.

Rules:
- If candidate answer is blank/empty -> set:
  "answer": "", "verdict": "bad",
  "comment": "No answer provided.",
  "tip": "Start with a concrete example; outline Situation, Actions, Result.",
  "scores": {{"Clarity":1,"Depth":1,"Structure":1,"Overall":1}}
- Keep "answer" <= 25 words (shorten if longer).
- "comment" <= 15 words, one sentence, specific.
- "tip" <= 18 words; if perfect, write "Excellent answer, keep it up!".
- Verdict in ["good","in-between","bad"].
- Scores are integers 1..5.
- Output ONLY valid JSON (no prose, no code fences) in this schema:

{{
  "index": {idx+1},
  "question": "{question.replace('"','\\"')}",
  "answer": "...",
  "verdict": "good|in-between|bad",
  "comment": "...",
  "tip": "...",
  "scores": {{"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1}}
}}

Candidate answer (raw): "{safe_answer}"
"""
        raw = ask_openai_json(
            grading_prompt,
            model=st.session_state.model,
            max_tokens=st.session_state.grade_tokens,
        )
        try:
            data = clean_and_parse_json(raw)
            return data
        except Exception:
            # One retry with a simpler reminder
            retry = (
                grading_prompt
                + "\nREMINDER: Output ONLY valid JSON exactly matching the schema."
            )
            raw2 = ask_openai_json(
                retry,
                model=st.session_state.model,
                max_tokens=st.session_state.grade_tokens,
            )
            try:
                return clean_and_parse_json(raw2)
            except Exception:
                # last resort basic outcome
                return {
                    "index": idx + 1,
                    "question": question,
                    "answer": (answer or "")[:80],
                    "verdict": "in-between" if answer.strip() else "bad",
                    "comment": "Parser fallback.",
                    "tip": "Tighten structure and add metrics.",
                    "scores": {
                        "Clarity": 3 if answer.strip() else 1,
                        "Depth": 3 if answer.strip() else 1,
                        "Structure": 3 if answer.strip() else 1,
                        "Overall": 3 if answer.strip() else 1,
                    },
                }

    # submit/grade
    gcol1, gcol2 = st.columns([1, 1])
    with gcol1:
        grade_btn = st.button("💬 Submit & Grade", use_container_width=True)
    with gcol2:
        next_disabled = st.session_state.results[idx] is None
        next_label = "➡️ Next Question" if idx < 9 else "🏁 Finish"
        next_btn = st.button(
            next_label, use_container_width=True, disabled=next_disabled
        )

    # on grade
    if grade_btn:
        if misuse_guard(st.session_state.answers[idx]):
            st.error("This looks unsafe or out of scope. Please rephrase.")
        else:
            with st.spinner("Grading…"):
                result = grade_one(q, st.session_state.answers[idx])
            # compute weighted + points (continuous)
            sc = result.get("scores", {}) or {}
            clarity = float(sc.get("Clarity", 0))
            depth = float(sc.get("Depth", 0))
            structure = float(sc.get("Structure", 0))
            overall = float(sc.get("Overall", 0))
            weighted = round((clarity + depth + structure + 2 * overall) / 5, 2)
            # points from weighted (0..1)
            points = min(1.0, round(weighted / 5.0, 2))
            result["_weighted"] = weighted
            result["_points"] = points

            st.session_state.results[idx] = result
            st.session_state.total_points = round(
                sum((r or {}).get("_points", 0.0) for r in st.session_state.results), 2
            )

    # show feedback (if graded)
    res = st.session_state.results[idx]
    if res is not None:
        ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
        COLOR = {"good": "green", "in-between": "orange", "bad": "red"}
        verdict = (res.get("verdict") or "").lower()
        icon = ICON.get(verdict, "❔")
        color = COLOR.get(verdict, "gray")
        points = res.get("_points", 0.0)
        weighted = res.get("_weighted", 0.0)

        st.markdown(
            f"<span style='color:{color};font-weight:700'>{icon} Q{idx+1} — {verdict.upper()} · +{points:.2f} pts</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Your (shortened) answer:** {res.get('answer','')}")
        st.markdown(f"**Comment:** {res.get('comment','')}")
        st.markdown(f"**Tip:** {res.get('tip','')}")
        sc = res.get("scores", {}) or {}
        st.markdown(
            f"**Scores:** Clarity: {int(sc.get('Clarity',0))}/5 · "
            f"Depth: {int(sc.get('Depth',0))}/5 · "
            f"Structure: {int(sc.get('Structure',0))}/5 · "
            f"Overall: {int(sc.get('Overall',0))}/5 "
            f"(weighted: **{weighted}/5**)"
        )
        st.caption(f"Total so far: **{st.session_state.total_points:.2f}/10**")

    # on next
    if next_btn and res is not None:
        if idx < 9:
            st.session_state.cur_idx += 1
            st.experimental_rerun()
        else:
            st.session_state.done = True
            st.experimental_rerun()

st.divider()

# =========================
# Final Summary (after 10)
# =========================
if st.session_state.questions and st.session_state.done:
    st.subheader("🏁 Final Summary")
    items = st.session_state.results
    ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
    rows = []
    for i, it in enumerate(items, 1):
        if not it:
            continue
        rows.append(
            {
                "Q#": i,
                "Verdict": it.get("verdict", ""),
                "Points": f"{it.get('_points',0.0):.2f}",
                "Weighted": f"{it.get('_weighted',0.0):.2f}/5",
                "Tip": it.get("tip", ""),
            }
        )
    # print as markdown table (simple)
    if rows:
        st.markdown("| Q# | Verdict | Points | Weighted | Tip |")
        st.markdown("|---:|:-------:|------:|:--------:|:----|")
        for r in rows:
            st.markdown(
                f"| {r['Q#']} | {r['Verdict']} | {r['Points']} | {r['Weighted']} | {r['Tip']} |"
            )

    total = round(sum((it or {}).get("_points", 0.0) for it in items), 2)
    avg_weighted = round(
        sum((it or {}).get("_weighted", 0.0) for it in items) / max(1, len(items)), 2
    )
    st.success(
        f"**Final grade:** {total:.2f}/10 · Average weighted: **{avg_weighted}/5**"
    )

    # export block
    def export_md() -> str:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [f"# PrepMate Summary ({now})\n", f"Topic: {st.session_state.topic}\n"]
        for i, (q, a, it) in enumerate(
            zip(st.session_state.questions, st.session_state.answers, items), 1
        ):
            parts.append(f"## Q{i}: {q}\n")
            parts.append(f"**Your answer:**\n{a}\n")
            if it:
                parts.append(
                    f"**Verdict:** {it.get('verdict')} · **Points:** {it.get('_points',0.0):.2f} · "
                    f"**Weighted:** {it.get('_weighted',0.0):.2f}/5\n"
                    f"**Comment:** {it.get('comment','')}\n**Tip:** {it.get('tip','')}\n"
                )
            parts.append("---\n")
        parts.append(
            f"**Final grade:** {total:.2f}/10 · Average weighted: **{avg_weighted}/5**\n"
        )
        return "\n".join(parts)

    md = export_md()
    st.download_button(
        "⬇️ Download Session (Markdown)",
        data=md.encode("utf-8"),
        file_name="prepmate_session.md",
        mime="text/markdown",
    )
