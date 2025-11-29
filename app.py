import os
import re
import json
import random
import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Bootstrapping
st.set_page_config(page_title="PrepMate: Interview Practice", layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(
        "❌ No OpenAI API key found. Add OPENAI_API_KEY to .env or Streamlit Secrets."
    )
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# Helpers & Config
def init_state():
    ss = st.session_state
    ss.setdefault("q10", [])
    ss.setdefault("idx", 0)
    ss.setdefault("graded", [])
    ss.setdefault("last_feedback", None)
    ss.setdefault("finished", False)
    ss.setdefault("started", False)
    ss.setdefault("run_id", 0)


init_state()

difficulty_tips = {
    "Easy": "Ask straightforward, entry-level questions testing basic understanding.",
    "Medium": "Ask mid-level questions involving reasoning and concrete examples.",
    "Hard": "Ask complex, scenario-based questions testing depth and creativity.",
}

persona_guides = {
    "Neutral": "Professional, concise, unbiased phrasing. Focus on clarity and substance.",
    "Friendly coach": "Warm, encouraging, highlight strengths first, then gentle suggestions.",
    "Strict bar-raiser": "Direct, demanding, precise. Expect evidence and reject vagueness.",
    "Motivational mentor": "Inspiring, emphasize progress and growth, push for specifics.",
    "Calm psychologist": "Analytical, reflective; probe reasoning and self-awareness.",
    "Playful mock interviewer": "Witty, teasing yet fair; mix humor with actionable insight.",
    "Corporate recruiter": "Professional and evaluative; care about communication and impact.",
    "Sarcastic Interviewer": "Dry, a bit snarky, incisive; cutting but helpful observations.",
}

persona_lines = {
    "Neutral": [
        "Clear thinking starts with clear structure—tighten one example and one metric.",
        "Good bones—add a specific trade-off and a measured outcome.",
        "Anchor claims with a before/after metric and a concrete decision.",
    ],
    "Friendly coach": [
        "You’re on the right track! Add one tangible metric to make it pop.",
        "Nice direction—tighten with a crisp example and outcome.",
        "Love the energy—ground it with one trade-off you considered.",
    ],
    "Strict bar-raiser": [
        "Assertions aren’t evidence. Provide metrics and trade-offs, or it doesn’t pass.",
        "Cut slogans. Demonstrate mechanism, constraints, and a measured outcome.",
        "Benchmark, constrain, justify. Precision beats platitudes.",
    ],
    "Motivational mentor": [
        "You’ve got the framework—now land a sharp metric to seal it.",
        "Strong start—push yourself to name the trade-off you accepted.",
        "You can elevate this: one number, one decision, one lesson.",
    ],
    "Calm psychologist": [
        "Surface your reasoning chain—assumptions, constraints, and outcome.",
        "Name the uncertainty and how you tested it; include one measure.",
        "Turn intuition into evidence—pick a metric that reflects your claim.",
    ],
    "Playful mock interviewer": [
        "Spice level is mild—kick it up with specifics. ‘Specifics beat slogans.’",
        "Fun vibe—now feed me numbers, not vibes. One metric, one trade-off.",
        "Cute story; the plot twist is data. Add numbers and a decision.",
    ],
    "Corporate recruiter": [
        "Translate this into business impact and stakeholder value with one metric.",
        "Tie to OKRs and risk; include cost or timeline trade-off.",
        "Hiring panel cares about impact—quantify it and cite a decision.",
    ],
    "Sarcastic Interviewer": [
        "Nice postcard. Now send the package: metric, constraint, outcome.",
        "Lovely brochure copy. Add a number and admit one real trade-off.",
        "Great bedtime story; I slept well. Wake me with data and a decision.",
    ],
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
    a = (ans or "").strip()
    if len(a.split()) <= 1:
        return True
    for pat in AUTO_ZERO_PATTERNS:
        if re.match(pat, a, re.IGNORECASE):
            return True
    return False


def ask_openai_text(
    prompt: str, *, model: str, max_tokens: int, temperature: float = 0.4
) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
    )
    try:
        return resp.output_text
    except Exception:
        return str(resp)


def ask_openai_json(prompt: str, *, model: str, max_tokens: int) -> str:
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_tokens,
            temperature=0.2,
            top_p=1,
            response_format={"type": "json_object"},
        )
        return resp.output_text
    except Exception:
        return ask_openai_text(
            prompt, model=model, max_tokens=max_tokens, temperature=0.2
        )


def parse_first_json(text: str):
    try:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        js = m.group(0)
        js = re.sub(r",(\s*[}\]])", r"\1", js)
        return json.loads(js)
    except Exception:
        return None


def band_from_weighted(w: float) -> str:
    if w < 1:
        return "very bad"
    if w < 2:
        return "bad"
    if w < 3:
        return "intermediate"
    if w < 4:
        return "good"
    return "very good"


def badge(label: str, color: str):
    st.markdown(
        f"<span style='display:inline-block;background:{color};color:#111;padding:4px 10px;"
        f"border-radius:999px;font-weight:700;font-size:12px;margin-right:6px;'>{label}</span>",
        unsafe_allow_html=True,
    )


# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.select_slider(
        "Difficulty", ["Easy", "Medium", "Hard"], value="Medium"
    )
    persona = st.selectbox("Interviewer Persona", list(persona_guides.keys()), index=0)
    model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"], index=0)
    max_tokens = st.slider("Max tokens (per call)", 300, 1600, 800)
    USE_MOCK = st.toggle("🧪 Mock mode", False)

# Header
st.title("PrepMate: Interview Practice")
st.caption(
    "Answer one question at a time, get instant feedback, and a final evaluation at the end."
)

difficulty_color = {"Easy": "#21c55d", "Medium": "#fbbf24", "Hard": "#ef4444"}[
    difficulty
]
badge(f"Difficulty: {difficulty}", difficulty_color)
badge(f"Interviewer: {persona}", "#a5b4fc")

# Topic & Optional Context (hidden after start)
if not st.session_state.started:
    st.markdown("### What do you want to practice?")
    topic = st.text_area(
        "",
        placeholder="e.g., System design, SQL, leadership, data analysis…",
        height=90,
        key="topic_input",
        label_visibility="collapsed",
    )

    files = None
    jd_text = ""
    resume_text = ""

    with st.expander("📎 Optional: Upload Job Description or Resume (TXT, MD, PDF)"):
        st.markdown(
            "Use these fields if you want more tailored questions based on a job ad or your CV."
        )
        files = st.file_uploader(
            "Attach files (optional)",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )
        jd_text = st.text_area("Job Description (optional)", height=120, key="jd_paste")
        resume_text = st.text_area(
            "Resume highlights (optional)", height=120, key="resume_paste"
        )

    def read_file_safe(f):
        try:
            raw = f.read()
            if f.type in ("text/plain", "text/markdown"):
                try:
                    return raw.decode("utf-8")
                except Exception:
                    return raw.decode("cp1252", errors="ignore")
            else:
                return f"[Attached file: {f.name}]"
        except Exception:
            return f"[Attached file: {getattr(f, 'name', 'unknown')}]"

    attached_blobs = []
    if files:
        for f in files:
            attached_blobs.append(read_file_safe(f))
    attached_context = "\n".join(attached_blobs).strip()

    if st.button("🧠 Generate Questions", use_container_width=True):
        if misuse_guard(topic, jd_text, resume_text, attached_context):
            st.error("This looks unsafe or out of scope. Please rephrase.")
        else:
            if USE_MOCK:
                q10 = [
                    f"Mock question {i} about {topic or 'general readiness'}"
                    for i in range(1, 11)
                ]
            else:
                guide = difficulty_tips[difficulty]
                prompt = f"""
You are an experienced interviewer.

Difficulty: {difficulty}
Guideline: {guide}
Persona style (for tone consistency only): {persona_guides[persona]}
Focus topic(s): {topic or 'General interview readiness'}

Optional context (may inform relevance):
{(jd_text or '').strip()}
{(resume_text or '').strip()}
{attached_context}

Task:
Create EXACTLY 10 interview questions that MIX technical and behavioral angles about the topic.
- Number them 1..10, one per line.
- Keep each question under 25 words.
- No extra commentary.
"""
                out = ask_openai_text(
                    prompt, model=model, max_tokens=700, temperature=0.4
                )
                lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
                q10 = []
                for ln in lines:
                    head = ln.split(maxsplit=1)[0]
                    if head.rstrip(".)").isdigit():
                        if "." in ln:
                            q = ln.split(".", 1)[1]
                        elif ")" in ln:
                            q = ln.split(")", 1)[1]
                        else:
                            q = ln[len(head) :]
                        q10.append(q.strip(" -–•\t"))
                q10 = q10[:10]
                while len(q10) < 10:
                    q10.append(f"Follow-up question {len(q10)+1}")

            st.session_state.q10 = q10
            st.session_state.idx = 0
            st.session_state.graded = []
            st.session_state.last_feedback = None
            st.session_state.finished = False
            st.session_state.started = True

            # Immediately rerun so topic/optional inputs disappear
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()


# One-by-one flow
def grade_one(q_idx: int, question: str, answer: str) -> Dict:
    if is_auto_zero(answer):
        return {
            "index": q_idx + 1,
            "question": question,
            "answer": answer.strip(),
            "verdict": "very bad",
            "points": 0.0,
            "weighted": 0.0,
            "scores": {"Clarity": 0, "Depth": 0, "Structure": 0, "Overall": 0},
            "comment": "No useful answer provided.",
            "tip": "Prepare concrete examples and basic strategies before interviews.",
            "persona_note": random.choice(persona_lines[persona]),
        }

    grading_prompt = f"""
Persona (use for tone in comments only): {persona_guides[persona]}

Strictly grade ONE answer with these rules:

- Score each: Clarity, Depth, Structure, Overall ∈ [1..5] (integers).
- weighted = (Clarity + Depth + Structure + 2*Overall)/5  (range 1..5)
- points = round(weighted/5, 2)
- Verdict bands:
    0–1  -> "very bad"
    1–2  -> "bad"
    2–3  -> "intermediate"
    3–4  -> "good"
    4–5  -> "very good"
- Comment: 1–2 sentences, persona voice, specific to THIS answer. No brackets.
- Tip: ≤ 16 words, actionable.
- If answer is near-perfect (weighted ≥ 4.8), tip = "Excellent answer — keep it up."

Return ONLY JSON:
{{
  "item": {{
    "Clarity": 1, "Depth": 1, "Structure": 1, "Overall": 1,
    "weighted": 1.0,
    "points": 0.20,
    "verdict": "bad",
    "comment": "....",
    "tip": "...."
  }}
}}

Question: "{question}"
Answer: "{answer.replace('"', '\\"')}"
"""
    if USE_MOCK:
        wc = len(answer.split())
        clarity = min(5, max(1, 1 + wc // 12))
        depth = min(5, max(1, 1 + wc // 15))
        structure = min(5, max(1, 1 + wc // 14))
        overall = min(5, max(1, 1 + wc // 13))
        weighted = (clarity + depth + structure + 2 * overall) / 5
        points = round(weighted / 5, 2)
        verdict = band_from_weighted(weighted)
        return {
            "index": q_idx + 1,
            "question": question,
            "answer": answer.strip(),
            "verdict": verdict,
            "points": points,
            "weighted": round(weighted, 2),
            "scores": {
                "Clarity": clarity,
                "Depth": depth,
                "Structure": structure,
                "Overall": overall,
            },
            "comment": random.choice(persona_lines[persona]),
            "tip": "Anchor with one metric and trade-off.",
            "persona_note": random.choice(persona_lines[persona]),
        }
    raw = ask_openai_json(grading_prompt, model=model, max_tokens=700)
    js = parse_first_json(raw)
    if not js or "item" not in js:
        return {
            "index": q_idx + 1,
            "question": question,
            "answer": answer.strip(),
            "verdict": "intermediate",
            "points": 0.5,
            "weighted": 2.5,
            "scores": {"Clarity": 3, "Depth": 2, "Structure": 3, "Overall": 2},
            "comment": random.choice(persona_lines[persona]),
            "tip": "State one metric and a concrete example.",
            "persona_note": random.choice(persona_lines[persona]),
        }

    item = js["item"]
    c = int(item.get("Clarity", 1))
    d = int(item.get("Depth", 1))
    s = int(item.get("Structure", 1))
    o = int(item.get("Overall", 1))
    weighted = float(item.get("weighted", (c + d + s + 2 * o) / 5))
    weighted = max(0.0, min(5.0, weighted))
    points = round(weighted / 5, 2)
    verdict = band_from_weighted(weighted)

    return {
        "index": q_idx + 1,
        "question": question,
        "answer": answer.strip(),
        "verdict": verdict,
        "points": points,
        "weighted": round(weighted, 2),
        "scores": {"Clarity": c, "Depth": d, "Structure": s, "Overall": o},
        "comment": str(
            item.get("comment", random.choice(persona_lines[persona]))
        ).strip(),
        "tip": str(item.get("tip", "Add one metric and a trade-off.")).strip(),
        "persona_note": random.choice(persona_lines[persona]),
    }


if st.session_state.started and not st.session_state.finished:
    q_list = st.session_state.q10
    idx = st.session_state.idx
    question = q_list[idx]

    st.divider()
    st.subheader(f"Question {idx+1}/10")
    st.markdown(f"**{question}**")

    answer = st.text_area(
        "Your answer",
        key=f"ans_{idx}",
        height=140,
        placeholder="Type your answer here…",
    )

    if st.button(
        "💬 Submit answer for scoring and feedback",
        key=f"grade_btn_{idx}",
        use_container_width=True,
    ):
        fb = grade_one(idx, question, answer)
        st.session_state.last_feedback = fb
        st.session_state.graded.append(fb)

    fb = st.session_state.last_feedback
    if fb and fb.get("index") == idx + 1:
        st.markdown(
            f"<div style='border:2px solid {difficulty_color};padding:14px;border-radius:8px;'>"
            f"{fb['persona_note']} {fb['comment']}"
            f"</div>",
            unsafe_allow_html=True,
        )

        color_map = {
            "very bad": "#ef4444",
            "bad": "#f97316",
            "intermediate": "#eab308",
            "good": "#22c55e",
            "very good": "#16a34a",
        }
        v_color = color_map[fb["verdict"]]
        st.markdown(
            f"<div style='border:2px dashed {v_color};padding:12px;border-radius:8px;margin-top:8px'>"
            f"<b style='color:{v_color}'>Verdict: {fb['verdict'].upper()}</b> · "
            f"+{fb['points']:.2f} pts · Weighted: {fb['weighted']:.2f}/5<br>"
            f"<b>Scores:</b> Clarity: {fb['scores']['Clarity']}/5 · Depth: {fb['scores']['Depth']}/5 · "
            f"Structure: {fb['scores']['Structure']}/5 · Overall: {fb['scores']['Overall']}/5<br>"
            f"<b>Tip:</b> {fb['tip']}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(1.0, max(0.0, fb["points"])))

        if st.button("➡️ Next Question", key=f"next_{idx}", use_container_width=True):
            st.session_state.idx += 1
            st.session_state.last_feedback = None
            if st.session_state.idx >= 10:
                st.session_state.finished = True

# Final summary
if st.session_state.finished:
    st.divider()
    st.subheader("🏁 Final Evaluation")

    items = st.session_state.graded
    total_points = round(sum(i["points"] for i in items), 2)
    avg_weighted = (
        round(sum(i["weighted"] for i in items) / len(items), 2) if items else 0.0
    )

    if total_points <= 3:
        note = "❌ Not Ready For Interview"
        banner_color = "#ef4444"
    elif total_points <= 6:
        note = "🟠 Almost Ready for the Interview"
        banner_color = "#f59e0b"
    else:
        note = "🟢 Ready for the Interview"
        banner_color = "#22c55e"

    st.markdown(
        f"<div style='padding:14px;background:{banner_color};color:#111;border-radius:10px;font-weight:700'>"
        f"Total Points: {total_points:.2f}/10 • Avg Weighted: {avg_weighted:.2f}/5 • {note}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Feedback Table")
    cols = st.columns([3, 4, 1.2, 2.4])
    with cols[0]:
        st.markdown("**Question**")
    with cols[1]:
        st.markdown("**Answer**")
    with cols[2]:
        st.markdown("**Overall (Weighted)**")
    with cols[3]:
        st.markdown("**Tip**")

    for i in items:
        q = i["question"]
        a = i["answer"]
        with st.container():
            c = st.columns([3, 4, 1.2, 2.4])
            with c[0]:
                st.write(q)
            with c[1]:
                st.write(a)
            with c[2]:
                st.write(f"{i['weighted']:.2f}/5")
            with c[3]:
                st.write(i["tip"])
        st.markdown("<hr style='margin:6px 0;opacity:0.2'>", unsafe_allow_html=True)

    if st.button("🔄 Take a New Quiz", use_container_width=True):
        st.session_state.q10 = []
        st.session_state.idx = 0
        st.session_state.graded = []
        st.session_state.last_feedback = None
        st.session_state.finished = False
        st.session_state.started = False
        st.session_state.run_id += 1

# Footer
st.caption("© 2025 PrepMate — Built with Streamlit & OpenAI API")
