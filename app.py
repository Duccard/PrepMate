import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ No OpenAI API key found. Please add it to your .env file.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- App layout ----
st.set_page_config(page_title="Interview Practice", page_icon="🎤", layout="wide")
st.title("🎤 Interview Practice App")
st.caption("Practice answering interview questions and get instant feedback.")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    role = st.selectbox("Select role/domain", ["General", "Data Science", "Backend", "Frontend", "Product", "HR"])
    difficulty = st.radio("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7)
    st.divider()
    st.caption("🔒 Basic guard against unsafe prompts")

# Main inputs
topic = st.text_area("What do you want to practice?", placeholder="e.g. SQL, system design, behavioral questions...")
job_desc = st.text_area("Optional: paste job description here")
resume = st.text_area("Optional: paste resume summary or bullet points")

if st.button("🧠 Generate Questions"):
    with st.spinner("Generating interview questions..."):
        prompt = f"""
        You are an experienced interviewer.
        Role: {role}
        Difficulty: {difficulty}
        Topic: {topic or 'General interview'}
        Job Description: {job_desc or 'N/A'}
        Resume Summary: {resume or 'N/A'}

        Generate 5 relevant interview questions, a mix of technical and behavioral.
        """
        response = client.responses.create(
            model="gpt-4o-mini",
            temperature=temperature,
            input=prompt
        )
        st.subheader("📋 Suggested Questions:")
        st.write(response.output_text)

st.divider()
st.caption("🚀 MVP complete — next: add feedback & scoring features.")
