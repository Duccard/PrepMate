import os
import re
import json
import datetime
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# Setup
# =========================
st.set_page_config(page_title="PrepMate", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Missing OpenAI API key in .env or Streamlit Secrets (OPENAI_API_KEY).")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# --- Safe rerun for all versions ---
def _rerun():
    """Call rerun compatibly across Streamlit versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# =========================
# Utility functions
# =========================
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


def ask_openai_text(
    prompt: str, *, model: str, temperature: float, top_p: float, max_tokens: int
) -> str:
    resp = client.responses.create(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_tokens,
        input=prompt,
    )
    return getattr(resp, "output_text", str(resp))


def ask_openai_json(prompt: str, *, model: str, max_tokens: int) -> str:
    """Use JSON mode for gpt-4.1 family, fallback otherwise."""

    def supports_json(m):
        return "gpt-4.1" in (m or "").lower()

    if supports_json(model):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.2,
                top_p=1,
                max_output_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return getattr(resp, "output_text", str(resp))
        except Exception:
            pass
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=0.2,
        top_p=1,
        max_output_tokens=max_tokens,
    )
    return getattr(resp, "output_text", str(resp))


def parse_json_object(raw: str) -> Dict[str, Any]:
    js = raw.strip()
    m = re.search(r"\{[\s\S]*\}", js)
    if m:
        js = m.group(0)
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    return json.loads(js)


# =========================
# Sidebar / Settings
# =========================
with st.sidebar:
    st.header("🎛️ Settings")

    persona_guides = {
        "Neutral": "Professional, concise, unbiased.",
        "Friendly coach": "Supportive, encourages reflection, gentle hints.",
        "Strict bar-raiser": "Challenging, expects precision and depth.",
        "Calm psychologist": "Analytical, empathetic, self-aware focus.",
        "Corporate recruiter": "Evaluates professionalism and communication.",
    }
    persona = st.selectbox("Interviewer persona", list(persona_guides.keys()), index=0)

    st.markdown("### ⚙️ Model")
    model = st.selectbox(
        "Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini"], index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p sampling", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 200, 2000, 700, 50)
    grade_tokens = max(1200, int(max_tokens * 1.3))
    USE_MOCK = st.toggle("🧪 Mock Mode (no API calls)", False)
    st.caption("🔒 Basic misuse guard active.")


# =========================
# Session
# =========================
def init_state():
    if "round_started" not in st.session_state:
        st.session_state.round_started = False
    if "questions" not in st.session_state:
        st.session_state.questions: List[str] = []
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "answers" not in st.session_state:
        st.session_state.answers: List[str] = [""] * 10
    if "results" not in st.session_state:
        st.session_state.results: List[Dict[str, Any] | None] = [None] * 10
    if "total_points" not in st.session_state:
        st.session_state.total_points = 0.0


init_state()


# =========================
# UI — Topic
# =========================
st.title("PrepMate")
st.caption(
    "Answer 10 interview questions one-by-one. Get instant feedback and a final summary."
)

topic = st.text_input(
    "What do you want to practice?", value=st.session_state.get("topic", "")
)
st.session_state.topic = topic


# =========================
# Question generator
# =========================
def mock_questions(t: str) -> List[str]:
    return [
        "Describe a system you've designed. What were its primary goals and challenges?",
        "How do you ensure scalability in your designs? Give an example.",
        "Explain consistency vs availability trade-offs with a real case.",
        "Tell me about a design that failed. What did you learn?",
        "Which architectural patterns do you prefer and why?",
        "How do you handle failures and retries in your design?",
        "What documentation artifacts do you maintain and how?",
        "How do you collaborate with non-technical stakeholders?",
        "What metrics do you track to assess system health?",
        "How do you keep up with evolving technologies?",
    ]


def generate_questions(t: str) -> List[str]:
    if USE_MOCK or not t.strip():
        return mock_questions(t)
    prompt = f"""
You are an expert interviewer. Persona: {persona_guides[persona]}
Topic: {t or 'General readiness'}

Return EXACTLY 10 mixed technical + behavioral questions, each under 20 words, numbered 1..10.
"""
    out = ask_openai_text(
        prompt, model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    qs = []
    for ln in lines:
        head = ln.split(maxsplit=1)[0]
        if head.rstrip(".)").isdigit():
            q = (
                ln.split(".", 1)[-1]
                if "." in ln
                else ln.split(")", 1)[-1] if ")" in ln else ln[len(head) :]
            ).strip(" -–•\t")
            qs.append(q)
    return qs[:10] or mock_questions(t)


# =========================
# Start / Reset
# =========================
colL, colR = st.columns([2, 1])
with colL:
    if st.button("🧠 Start 10-Question Round", use_container_width=True):
        st.session_state.questions = generate_questions(topic)
        st.session_state.idx = 0
        st.session_state.answers = [""] * 10
        st.session_state.results = [None] * 10
        st.session_state.total_points = 0.0
        st.session_state.round_started = True
        _rerun()
with colR:
    if st.button("🧹 Reset", use_container_width=True):
        for k in [
            "round_started",
            "questions",
            "idx",
            "answers",
            "results",
            "total_points",
        ]:
            if k in st.session_state:
                del st.session_state[k]
        init_state()
        _rerun()


if not st.session_state.round_started:
    st.info("Click **Start 10-Question Round** to begin.")
    st.stop()


# =========================
# Question display
# =========================
qs = st.session_state.questions
idx = st.session_state.idx
q = qs[idx]
st.subheader(f"Question {idx+1}/10")
st.write(q)

st.markdown("**Your answer**")
answer_key = f"answer_{idx}"
user_answer = st.text_area(
    "", key=answer_key, value=st.session_state.answers[idx], height=130
)
st.session_state.answers[idx] = user_answer


# =========================
# Grading logic
# =========================
def grade_one(question: str, answer: str) -> Dict[str, Any]:
    """Return JSON with grading results."""
    if USE_MOCK:
        ans_clean = (answer or "").strip()
        if not ans_clean:
            verdict, scores = "bad", {
                "Clarity": 1,
                "Depth": 1,
                "Structure": 1,
                "Overall": 1,
            }
        elif len(ans_clean) > 100:
            verdict, scores = "good", {
                "Clarity": 4,
                "Depth": 5,
                "Structure": 4,
                "Overall": 4,
            }
        else:
            verdict, scores = "in-between", {
                "Clarity": 3,
                "Depth": 3,
                "Structure": 3,
                "Overall": 3,
            }
        weighted = round(
            (
                scores["Clarity"]
                + scores["Depth"]
                + scores["Structure"]
                + 2 * scores["Overall"]
            )
            / 5,
            2,
        )
        points = round(weighted / 5, 2)
        tip = (
            "Excellent answer—keep it up!"
            if points > 0.95
            else (
                "Add metrics and specific examples."
                if points < 0.4
                else "Expand slightly on challenges or trade-offs."
            )
        )
        return {
            "question": question,
            "answer": ans_clean[:200],
            "verdict": verdict,
            "comment": "Mock feedback.",
            "tip": tip,
            "scores": scores,
            "_weighted": weighted,
            "_points": points,
        }

    safe_ans = (answer or "").replace('"', '\\"')
    prompt = f"""
You are an expert interviewer.

Persona: {persona_guides[persona]}
Topic: {topic or 'General readiness'}

Grade one question/answer pair.

Return ONLY JSON (no prose) like:
{{
 "question": "...",
 "answer": "...",
 "verdict": "good|in-between|bad",
 "comment": "...",
 "tip": "...",
 "scores": {{"Clarity":1-5,"Depth":1-5,"Structure":1-5,"Overall":1-5}}
}}

Question: {question}
Answer: "{safe_ans}"
"""
    raw = ask_openai_json(prompt, model=model, max_tokens=grade_tokens)
    data = parse_json_object(raw)
    sc = data.get("scores", {}) or {}
    c, d, s, o = (
        float(sc.get(k, 0)) for k in ["Clarity", "Depth", "Structure", "Overall"]
    )
    weighted = round((c + d + s + 2 * o) / 5, 2)
    data["_weighted"] = weighted
    data["_points"] = round(weighted / 5, 2)
    return data


# =========================
# Buttons
# =========================
colL, colR = st.columns([1, 1])
with colL:
    grade_btn = st.button("💬 Submit & Grade", use_container_width=True)
with colR:
    next_disabled = st.session_state.results[idx] is None
    next_btn = st.button(
        "➡️ Next Question", disabled=next_disabled, use_container_width=True
    )


# =========================
# Grading + rerun
# =========================
if grade_btn:
    if misuse_guard(st.session_state.answers[idx]):
        st.error("This looks unsafe. Please rephrase.")
    else:
        with st.spinner("Grading…"):
            result = grade_one(q, st.session_state.answers[idx])
        st.session_state.results[idx] = result
        st.session_state.total_points = round(
            sum((r or {}).get("_points", 0.0) for r in st.session_state.results if r), 2
        )
        _rerun()


# =========================
# Show feedback
# =========================
res = st.session_state.results[idx]
if res:
    ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
    COLOR = {"good": "green", "in-between": "orange", "bad": "red"}
    icon, color = ICON.get(res["verdict"], "❔"), COLOR.get(res["verdict"], "gray")
    pts, weighted = res["_points"], res["_weighted"]
    sc = res.get("scores", {})

    st.markdown(
        f"<span style='color:{color};font-weight:700'>{icon} Q{idx+1} — {res['verdict'].upper()} · +{pts:.2f} pts</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Your (shortened) answer:** {res['answer']}")
    st.markdown(f"**Comment:** {res['comment']}")
    st.markdown(f"**Tip:** {res['tip']}")
    st.markdown(
        f"**Scores:** Clarity: {int(sc.get('Clarity',0))}/5 · Depth: {int(sc.get('Depth',0))}/5 · "
        f"Structure: {int(sc.get('Structure',0))}/5 · Overall: {int(sc.get('Overall',0))}/5 "
        f"(weighted: **{weighted}/5**)"
    )
    st.caption(f"Total so far: **{st.session_state.total_points:.2f}/10**")


# =========================
# Next Question
# =========================
if next_btn and not next_disabled:
    st.session_state.idx = min(9, st.session_state.idx + 1)
    _rerun()


# =========================
# Final summary
# =========================
if st.session_state.idx == 9 and st.session_state.results[9] is not None:
    st.subheader("🏁 Final Summary")
    rows = []
    for i, (qq, rr) in enumerate(
        zip(st.session_state.questions, st.session_state.results), start=1
    ):
        rr = rr or {}
        sc = rr.get("scores", {}) or {}
        rows.append(
            {
                "Q": i,
                "Verdict": rr.get("verdict", ""),
                "Points": rr.get("_points", 0.0),
                "Weighted": rr.get("_weighted", 0.0),
                "Clarity": sc.get("Clarity", 0),
                "Depth": sc.get("Depth", 0),
                "Structure": sc.get("Structure", 0),
                "Overall": sc.get("Overall", 0),
                "Tip": rr.get("tip", ""),
            }
        )
    st.table(rows)
    st.success(f"Total: **{st.session_state.total_points:.2f}/10**")

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# PrepMate Summary ({now})\n", f"**Topic:** {topic}\n"]
    for i, (qq, rr) in enumerate(
        zip(st.session_state.questions, st.session_state.results), start=1
    ):
        rr = rr or {}
        md += [
            f"## Q{i}. {qq}",
            f"- Verdict: {rr.get('verdict','')}, Points: {rr.get('_points',0.0)}",
            f"- Scores: {rr.get('scores',{})}, Weighted: {rr.get('_weighted',0.0)}",
            f"- Comment: {rr.get('comment','')}",
            f"- Tip: {rr.get('tip','')}",
            "",
        ]
    st.download_button(
        "⬇️ Download Summary (Markdown)",
        data="\n".join(md).encode("utf-8"),
        file_name="prepmate_summary.md",
        mime="text/markdown",
    )

st.caption("© 2025 PrepMate — Built with Streamlit & OpenAI API")
