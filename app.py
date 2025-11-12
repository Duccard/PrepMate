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
    ss.setdefault("questions", [])  # list[str]
    ss.setdefault("current_idx", 0)  # 0..9
    ss.setdefault("results", [])  # list[normalized item dicts]
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
    "Neutral": "Professional, concise, objective. Balanced tone; fair but firm.",
    "Friendly Coach": "Warm, supportive, suggests concrete improvements. Encourages reflection.",
    "Strict Bar-Raiser": "Demanding and precise; challenges weak claims; expects metrics and rigor.",
    "Calm Psychologist": "Empathic, metacognitive prompts; probes self-awareness and decision rationale.",
    "Motivational Mentor": "Inspires growth; applauds progress; pushes for ambitious clarity.",
    "Corporate Recruiter": "Focus on business impact, communication polish, stakeholder alignment.",
    "Playful Mock Interviewer": "Wry, light but insightful; uses playful framing without being flippant.",
    "Algorithmic Stickler": "Hyper-structured, rubric-driven, cites criteria, nitpicks rigor.",
    "Sarcastic Interviewer": "Dry, pointed, witty; demands substance without fluff.",
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

# Persona-flavored final sign-off
persona_signoff = {
    "Neutral": "Balanced performance – focus next on shoring up weaker areas.",
    "Friendly Coach": "Great effort! Small tweaks will unlock the next level.",
    "Strict Bar-Raiser": "Raise the bar again. Specifics and metrics win offers.",
    "Calm Psychologist": "Notice your patterns: clarity grows with deliberate structure.",
    "Motivational Mentor": "You’re close. Double down on precision and you'll shine.",
    "Corporate Recruiter": "Translate effort into outcomes; hiring managers notice results.",
    "Playful Mock Interviewer": "Not bad! Now make it sing with sharper examples.",
    "Algorithmic Stickler": "Calibrate to the rubric; quantify, compare, conclude.",
    "Sarcastic Interviewer": "Better than a shrug. Now bring receipts, not vibes.",
}

difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions requiring reasoning, tradeoffs, and examples.",
    "Hard": "Ask complex, open-ended, scenario-based questions testing depth and creativity.",
}

DN_PAT = re.compile(
    r"\b(i\s*don'?t\s*know|dont\s*know|don\s*t\s*know|i\s*don'?t\s*care|dont\s*care|idk|no\s*idea|pass)\b",
    re.I,
)


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


def _coerce_int(x, lo=0, hi=5, default=1):
    try:
        v = int(round(float(x)))
        return max(lo, min(hi, v))
    except Exception:
        return default


def _coerce_float(x, lo=0.0, hi=10.0, default=0.0):
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default


def normalize_item(it: dict) -> dict:
    """Ensure all keys, auto-bad for unknown, compute weighted."""
    it = it or {}
    idx = _coerce_int(it.get("index", 0), lo=0, hi=100, default=0)
    qtext = str(it.get("question", "") or "")
    ans = str(it.get("answer", "") or "")
    verdict = str(it.get("verdict", "") or "").lower().strip()
    points = _coerce_float(it.get("points", 0.0), lo=0.0, hi=1.0, default=0.0)
    comment = str(it.get("comment", "") or "")
    tip = str(it.get("tip", "") or "")
    scores = it.get("scores") or {}

    clarity = _coerce_int(scores.get("Clarity", 1), 1, 5, 1)
    depth = _coerce_int(scores.get("Depth", 1), 1, 5, 1)
    structure = _coerce_int(scores.get("Structure", 1), 1, 5, 1)
    overall = _coerce_int(scores.get("Overall", 1), 1, 5, 1)

    if _force_bad_if_unknown(ans):
        verdict = "bad"
        points = 0.0
        clarity = depth = structure = overall = 1
        if not comment:
            comment = "No answer provided."
        if not tip:
            tip = "Offer a concise, concrete example next time."

    if verdict not in ("good", "in-between", "bad"):
        verdict = "in-between"
    if verdict == "good":
        points = 1.0
    elif verdict == "in-between":
        points = 0.5
    else:
        points = 0.0

    weighted = round((clarity + depth + structure + 2 * overall) / 5.0, 2)

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


def ask_openai_text(
    prompt: str, *, model: str, temperature: float, top_p: float, max_tokens: int
) -> str:
    if ss.use_mock:
        # Simple mock questions
        if "EXACTLY 10" in prompt and "Number them 1..10" in prompt:
            return "\n".join([f"{i}. Mock question {i}" for i in range(1, 11)])
        # Grader mock: one-item JSON
        if '"items"' in prompt:
            fake = {
                "items": [
                    {
                        "index": 1,
                        "question": "Mock?",
                        "answer": "demo",
                        "verdict": "in-between",
                        "points": 0.5,
                        "comment": "Decent but vague.",
                        "tip": "Be concrete.",
                        "scores": {
                            "Clarity": 3,
                            "Depth": 3,
                            "Structure": 3,
                            "Overall": 3,
                        },
                    }
                ]
            }
            return json.dumps(fake)
        return "Mock output."
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
    """Try JSON mode on 4.1*; else fall back to low-temp text mode."""

    def _supports_json(m: str) -> bool:
        m = (m or "").lower()
        return "gpt-4.1" in m  # includes gpt-4.1-mini

    if ss.use_mock:
        return ask_openai_text(
            prompt, model=model, temperature=0.2, top_p=1.0, max_tokens=max_tokens
        )
    if _supports_json(model):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_tokens,
                temperature=0.2,
                top_p=1,
                response_format={"type": "json_object"},
            )
            try:
                return resp.output_text
            except Exception:
                return str(resp)
        except Exception:
            pass
    # fallback
    return ask_openai_text(
        prompt, model=model, temperature=0.2, top_p=1.0, max_tokens=max_tokens
    )


def parse_items(raw_text: str) -> List[dict]:
    """Extract JSON {items:[...]}; tolerate minor trailing commas."""
    js = (raw_text or "").strip()
    m = re.search(r"\{[\s\S]*\}", js)
    if m:
        js = m.group(0)
    js = re.sub(r",(\s*[}\]])", r"\1", js)
    try:
        data = json.loads(js)
        items = data.get("items", [])
        if isinstance(items, dict):  # rare mis-shape
            items = [items]
        return items
    except Exception:
        return []


def estimate_cost(chars: int, model: str = "gpt-4o-mini") -> float:
    tokens = max(1, int(chars / 4))
    rates = {
        "gpt-4o-mini": 0.15 / 1_000_000,
        "gpt-4o": 5 / 1_000_000,
        "gpt-4.1": 5 / 1_000_000,
        "gpt-4.1-mini": 0.30 / 1_000_000,
    }
    return tokens * rates.get(model, 0.15 / 1_000_000)


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.header("🎛️ Settings")
    ss.difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value=ss.difficulty
    )
    ss.persona = st.selectbox(
        "Interviewer persona",
        list(persona_guides.keys()),
        index=list(persona_guides.keys()).index(ss.persona),
    )
    st.markdown(f"**Persona:** {persona_badges[ss.persona]}")
    st.markdown("### ⚙️ Model")
    model_choices = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
    default_idx = model_choices.index(ss.model) if ss.model in model_choices else 0
    ss.model = st.selectbox("Model", model_choices, index=default_idx)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    top_p = st.slider("Top-p sampling", 0.0, 1.0, 1.0, 0.05)
    max_tokens = st.slider("Max output tokens", 200, 2000, 800, 50)
    ss.use_mock = st.toggle("🧪 Mock Mode (no API calls)", ss.use_mock)
    st.caption("🔒 Basic misuse guard is active.")

# =========================
# Header
# =========================
st.title("Prepmate: Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)


# =========================
# Start screen (hidden when quiz running)
# =========================
def render_start():
    st.subheader("Practice Setup")
    ss.topic = st.text_area(
        "What do you want to practice?",
        value=ss.topic,
        placeholder="e.g., system design tradeoffs, SQL joins, behavioral STAR…",
        height=90,
    )

    with st.expander("📎 Optional: Upload Job Description (TXT, MD)"):
        jd_file = st.file_uploader("Choose a file", type=["txt", "md"])
        if jd_file is not None:
            if getattr(jd_file, "size", 0) > 200_000_000:  # 200MB UI limit
                st.warning("File too large; please upload a smaller TXT/MD.")
            else:
                raw = jd_file.read()
                try:
                    ss.jd_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    ss.jd_text = raw.decode("cp1252", errors="ignore")
                ss.jd_text = ss.jd_text[:200000]  # hard cap

    if st.button("🧠 Generate Questions", use_container_width=True):
        if misuse_guard(ss.topic, ss.jd_text):
            st.error("This looks unsafe or out of scope. Please rephrase.")
            return
        generate_questions()


def generate_questions():
    guideline = difficulty_tips[ss.difficulty]
    prompt = f"""
You are an interview question generator.

Persona: {persona_guides[ss.persona]}
Difficulty: {ss.difficulty} — {guideline}
Topic focus: {ss.topic or 'General interview readiness'}
Job description context (optional, may ignore if not useful):
{ss.jd_text or 'N/A'}

Task:
Produce a single mixed set of **EXACTLY 10** interview questions (technical + behavioral) relevant to the topic/context.
Rules:
- Number them 1..10, one per line.
- Each question ≤ 20 words.
- Be realistic and role-appropriate.

Format:
1. ...
2. ...
...
10. ...
"""
    with st.spinner("Generating questions…"):
        raw = ask_openai_text(
            prompt, model=ss.model, temperature=0.6, top_p=1.0, max_tokens=max_tokens
        )

    # Parse 10 lines
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    qs: List[str] = []
    for ln in lines:
        head = ln.split(maxsplit=1)[0]
        if head.rstrip(".)").isdigit():
            q = (
                ln.split(".", 1)[-1]
                if "." in ln
                else (ln.split(")", 1)[-1] if ")" in ln else ln[len(head) :])
            )
            qs.append(q.strip(" -–•\t"))
    qs = qs[:10]
    while len(qs) < 10:
        qs.append("(placeholder)")

    ss.questions = qs
    ss.current_idx = 0
    ss.results = []
    ss.quiz_running = True

    try:
        st.toast("✅ Questions ready.", icon="✅")
    except Exception:
        pass

    st.rerun()


# =========================
# Grading (one question)
# =========================
def grade_one(question: str, answer: str) -> Dict:
    safe_answer = (answer or "").replace('"', '\\"').strip()
    rules = f"""
You are an expert interviewer and strict grader.
Persona (tone for comments & tip): {persona_guides[ss.persona]}
Topic: {ss.topic or 'General readiness'}

Grade the SINGLE Q&A below and return ONLY valid JSON with key "items" holding exactly one object.

Hard rules:
- If the candidate answer is empty OR contains phrases like "don't know", "dont know", "idk", "don't care", "no idea", mark:
  "answer":"", "verdict":"bad", "points":0, "comment":"No answer provided.", "scores":{{"Clarity":1,"Depth":1,"Structure":1,"Overall":1}}, and include a brief "tip".
- Otherwise:
  * verdict ∈ {{"good","in-between","bad"}}
  * points: good=1.0, in-between=0.5, bad=0.0
  * scores are integers 1..5
  * limit "answer" (your short paraphrase) to ≤ 18 words
  * keep "comment" ≤ 14 words, specific; keep "tip" ≤ 14 words, actionable
- Comments/tips should reflect the persona tone, but no emojis.

JSON Schema (single item only):
{{
  "items": [
    {{
      "index": 1,
      "question": "...",
      "answer": "...",
      "verdict": "good|in-between|bad",
      "points": 1|0.5|0,
      "comment": "...",
      "tip": "...",
      "scores": {{"Clarity":1,"Depth":1,"Structure":1,"Overall":1}}
    }}
  ]
}}

Question: {json.dumps(question, ensure_ascii=False)}
Answer: {json.dumps(safe_answer, ensure_ascii=False)}
"""
    raw = ask_openai_json(rules, model=ss.model, max_tokens=600)
    items = parse_items(raw)
    if not items:
        retry = (
            rules
            + "\nREMINDER: Return ONLY the JSON object as specified, with exactly one item."
        )
        raw = ask_openai_json(retry, model=ss.model, max_tokens=600)
        items = parse_items(raw)
    if not items:
        # last-resort synthetic bad
        items = [
            {
                "index": 1,
                "question": question,
                "answer": answer,
                "verdict": "bad",
                "points": 0,
                "comment": "No parsable grading.",
                "tip": "Answer with specifics.",
                "scores": {"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1},
            }
        ]
    item = items[0]
    item["question"] = question
    item["answer"] = answer.strip()
    return normalize_item(item)


# =========================
# Quiz view (one by one)
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
    st.subheader(f"Q{i+1}. {q}")

    # Answer form
    with st.form(key=f"form_q_{i}", clear_on_submit=False):
        ans_key = f"answer_{i}"
        default_val = ss.get(ans_key, "")
        ans = st.text_area(
            "Your answer",
            value=default_val,
            height=140,
            placeholder="Type your answer here…",
        )
        submit = st.form_submit_button("💬 Submit answer for scoring and feedback")
        if submit:
            ss[ans_key] = ans
            if misuse_guard(ans):
                st.error("This looks unsafe or out of scope. Please rephrase.")
            else:
                with st.spinner("Scoring…"):
                    result = grade_one(q, ans)

                ICON = {"good": "✅", "in-between": "⚠️", "bad": "❌"}
                COLOR = {"good": "green", "in-between": "orange", "bad": "red"}
                icon = ICON.get(result["verdict"], "❔")
                color = COLOR.get(result["verdict"], "gray")

                st.markdown(
                    f"<span style='color:{color};font-weight:700'>{icon} "
                    f"Verdict: {result['verdict'].upper()} · +{result['points']:.1f} pts "
                    f"· Weighted: {result['weighted']:.2f}/5</span>",
                    unsafe_allow_html=True,
                )
                sc = result["scores"]
                st.write(
                    f"**Scores:** Clarity: {sc['Clarity']}/5 · Depth: {sc['Depth']}/5 · "
                    f"Structure: {sc['Structure']}/5 · Overall: {sc['Overall']}/5"
                )
                if result["comment"]:
                    st.write(f"**Comment:** {result['comment']}")
                if result["tip"]:
                    st.write(f"**Tip:** {result['tip']}")

                # store/replace this question's result
                if len(ss.results) > i:
                    ss.results[i] = result
                else:
                    ss.results.append(result)

                # Next question button
                if st.form_submit_button("➡️ Next question"):
                    ss.current_idx += 1
                    st.rerun()

    # Allow quitting mid-quiz
    st.button("⏹️ End quiz & show summary", on_click=lambda: _end_quiz_now())


def _end_quiz_now():
    ss.current_idx = len(ss.questions)
    st.rerun()


# =========================
# Final summary
# =========================
def render_final():
    st.subheader("🏁 Final Evaluation")

    items = [normalize_item(it) for it in ss.results]
    # If user skipped some questions, pad them as bad
    while len(items) < len(ss.questions):
        idx = len(items) + 1
        items.append(
            normalize_item(
                {
                    "index": idx,
                    "question": ss.questions[idx - 1],
                    "answer": "",
                    "verdict": "bad",
                    "points": 0,
                    "comment": "No answer provided.",
                    "tip": "Answer each question.",
                    "scores": {"Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1},
                }
            )
        )

    total_points = round(sum(it["points"] for it in items), 1)

    # readiness band
    band = "Not Ready for Interview"
    color = "red"
    if total_points > 7.0:
        band, color = "Ready for Interview", "green"
    elif total_points >= 3.0:
        band, color = "Almost Ready for Interview", "orange"

    st.markdown(
        f"<div style='padding:10px;border-radius:8px;background:{'rgba(0,128,0,0.08)' if color=='green' else ('rgba(255,165,0,0.12)' if color=='orange' else 'rgba(255,0,0,0.10)')};'>"
        f"<strong style='color:{color};font-size:1.1rem'>{band} — Total Points: {total_points:.1f}/10</strong>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Persona sign-off
    st.markdown(f"**{persona_badges[ss.persona]} says:** {persona_signoff[ss.persona]}")

    # Build final table
    rows = []
    for it in items:
        rows.append(
            {
                "Question": it["question"],
                "Answer": it["answer"] if it["answer"] else "—",
                "Overall (1–5)": it["scores"]["Overall"],
                "Tip": it["tip"] or it["comment"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Export
    def build_md():
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [
            f"# Prepmate Session Export ({now})\n",
            f"**Topic:** {ss.topic or 'N/A'}\n",
            f"**Persona:** {ss.persona}\n",
            f"**Difficulty:** {ss.difficulty}\n",
            f"**Total Points:** {total_points:.1f}/10 — {band}\n\n",
        ]
        for i, it in enumerate(items, 1):
            sc = it["scores"]
            parts.append(
                f"## Q{i}. {it['question']}\n"
                f"**Answer:** {it['answer'] or '—'}\n\n"
                f"- Verdict: {it['verdict']} (+{it['points']:.1f} pts), Weighted: {it['weighted']:.2f}/5\n"
                f"- Scores: C:{sc['Clarity']}/5 D:{sc['Depth']}/5 S:{sc['Structure']}/5 O:{sc['Overall']}/5\n"
                f"- Comment: {it['comment']}\n"
                f"- Tip: {it['tip']}\n"
            )
        parts.append(
            f"\n**{persona_badges[ss.persona]}**: {persona_signoff[ss.persona]}\n"
        )
        return "\n".join(parts)

    st.download_button(
        "⬇️ Download Summary (Markdown)",
        data=build_md().encode("utf-8"),
        file_name="prepmate_summary.md",
        mime="text/markdown",
    )

    # Restart option
    def _reset_all():
        ss.quiz_running = False
        ss.questions, ss.results = [], []
        ss.current_idx = 0
        # keep topic & jd_text for convenience
        st.rerun()

    st.button("🔁 Take another quiz", on_click=_reset_all)


# =========================
# Router
# =========================
if not ss.quiz_running:
    render_start()
else:
    render_quiz()
