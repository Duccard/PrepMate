# PrepMate – Interview Preparation App

PrepMate is a Streamlit web application that helps you practice job interviews with AI-generated questions and structured feedback. It uses OpenAI GPT models to play the role of an interviewer, grade your answers, and provide tips for improvement.

---

## Live Demo

You can try the app here:

https://prepmate-vehotkn5mymmd3kbyrnppd.streamlit.app/

---

## Features

### Question Generation

- Generates exactly 10 interview questions per session.
- Questions are tailored based on:
  - Topic or role (for example: data analysis, backend, leadership, product management).
  - Difficulty level: Easy, Medium, Hard.
  - Optional job description text.
  - Optional resume highlights.
  - Optional uploaded files (TXT, MD, simple PDF context labels).

### Interviewer Personas

<<<<<<< HEAD
python -m venv .venv
source .venv/bin/activate    
.\.venv\Scripts\activate    
=======
Choose an interviewer style that shapes the tone of feedback:
>>>>>>> a254778 (Updated comments and READ ME)

- Neutral  
- Friendly coach  
- Strict bar-raiser  
- Motivational mentor  
- Calm psychologist  
- Playful mock interviewer  
- Corporate recruiter  
- Sarcastic interviewer  

The persona influences the wording of comments and tips, but not the numeric scoring logic.

### Per-Question Grading

For each answer, PrepMate:

- Scores four dimensions on a 1–5 scale:
  - Clarity  
  - Depth  
  - Structure  
  - Overall  
- Computes a weighted score:

<<<<<<< HEAD
## 📋 How it works
=======
\[
\text{weighted} = \frac{\text{Clarity} + \text{Depth} + \text{Structure} + 2 \times \text{Overall}}{5}
\]
>>>>>>> a254778 (Updated comments and READ ME)

- Determines:
  - A verdict band: very bad, bad, intermediate, good, very good.  
  - A normalized points value in the range 0.0–1.0.  
- Produces:
  - Persona-flavored main comment.  
  - A short, actionable improvement tip (usually under 16 words).

The app also handles “auto-zero” answers (for example, empty or “I don’t know”) by giving a zero score with a generic tip.

### Final Evaluation

<<<<<<< HEAD
## 🧩 Tech Stack
=======
After all 10 questions are answered, you receive:
>>>>>>> a254778 (Updated comments and READ ME)

- Total points out of 10 (sum of all question points).  
- Average weighted score out of 5.  
- A readiness label:
  - Not Ready For Interview  
  - Almost Ready For The Interview  
  - Ready For The Interview  
- A feedback table for all questions showing:
  - Question text  
  - Your answer  
  - Weighted score  
  - Tip  

Then you can restart and generate a new quiz.

### Models and Settings

The sidebar includes:

- Difficulty (Easy, Medium, Hard)  
- Interviewer persona  
- OpenAI model selection (`gpt-4o-mini`, `gpt-4.1-mini`, `gpt-4.1`)  
- Max tokens per API call  
- Mock mode (local grading simulator without API usage)

### Safety and Misuse Guard

A simple misuse guard blocks prompts containing unsafe keywords (for example, phishing, malware, bypass security). If detected, question generation is stopped with an error message.

---

# Setup & Documentation Summary

| Section | Details |
|--------|---------|
| **1. Clone the repository** | ```bash\ngit clone https://github.com/TuringCollegeSubmissions/vmikul-AE.1.4.git\ncd vmikul-AE.1.4\n``` |
| **2. Create a virtual environment** | ```bash\npython -m venv .venv\n``` |
| **Activate the environment (macOS/Linux)** | ```bash\nsource .venv/bin/activate\n``` |
| **Activate the environment (Windows)** | ```bash\n.\.venv\Scripts\activate\n``` |
| **3. Install dependencies** | ```bash\npip install streamlit openai python-dotenv\n``` |
| **Alternative: Using requirements.txt** | `requirements.txt`: ```text\nstreamlit\nopenai\npython-dotenv\n``` Install: ```bash\npip install -r requirements.txt\n``` |
| **4. Add your OpenAI API key** | `.env` file: ```ini\nOPENAI_API_KEY=sk-your-openai-api-key\n``` |
| **5. Run the app** | ```bash\nstreamlit run app.py\n``` The app opens at **http://localhost:8501** |
| **How It Works – Overview Flow** | - Select topic, difficulty, persona, and model<br>- Optionally add job description, resume, or files<br>- Click *Generate Questions* to get 10 tailored questions<br>- Answer questions one-by-one<br>- Each answer gets graded automatically<br>- Final evaluation and feedback table displayed |
| **Session State Variables** | - **q10**: 10 generated questions<br>- **idx**: current question index<br>- **graded**: grading results<br>- **last_feedback**: latest feedback<br>- **started**: interview session started<br>- **finished**: all questions completed<br>- **run_id**: reset counter |
| **Project Structure** | ```bash\nvmikul-AE.1.4/\n│── app.py            # Main Streamlit app\n│── .env              # API key (ignored by Git)\n│── requirements.txt  # Optional dependency file\n└── README.md         # Documentation\n``` |
| **Customization Options** | - Modify question-generation prompts<br>- Adjust grading logic & weighting<br>- Change or add personas<br>- Edit layout and UI components<br>- All located inside **app.py** |
| **License** | This project is licensed under the **MIT License** (or replace with your preferred license). |

