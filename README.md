# PrepMate, The Interview Preparation App

A simple **Streamlit** web app that helps users prepare for job interviews using **OpenAI’s GPT models**.

## 🌐 Live Demo
https://prepmate-vehotkn5mymmd3kbyrnppd.streamlit.app/

## 🚀 Features
- Generates tailored interview questions based on:
  - Role / domain (e.g. Data Science, Backend, HR)
  - Difficulty level (Easy, Medium, Hard)
  - Custom topics, job descriptions, and resume details
- Adjustable model temperature for creative or focused outputs
- Secure API key handling via `.env`
- Clean Streamlit interface for quick interaction

## 🛠️ Setup

1. **Clone this repo**

   ```bash
   git clone https://github.com/TuringCollegeSubmissions/vmikul-AE.1.4.git
   cd "Interview App"

2. Create and activate a virtual environment

python -m venv .venv
source .venv/bin/activate     # mac/linux
# .\.venv\Scripts\activate    # windows

3. Install dependencies

pip install streamlit openai python-dotenv

4. Add your OpenAI API key

Create a file named .env in the project folder and add:
OPENAI_API_KEY=sk-yourkeyhere

5. Run the app:
streamlit run app.py

📋 How it works

Choose your role, difficulty, and what you want to practice.

(Optional) Paste a job description or your resume bullet points.

Click Generate Questions — the app will create relevant interview questions using OpenAI.

🧩 Tech Stack

Python 3.11+

Streamlit

OpenAI API

dotenv
