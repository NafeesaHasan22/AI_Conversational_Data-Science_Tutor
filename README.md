# AI Conversational Data Science Tutor

A sleek, intelligent, and fully interactive AI tutor built using **Streamlit**, **LangChain**, and **Gemini Pro**. This tool empowers students and professionals to deepen their understanding of Data Science through conversation, code execution, and intelligent feedback.

---

## Key Features

- **💬 Natural Language Q&A** — Ask any Data Science question and get instant, accurate responses
- **📊 Data Upload** — Upload CSV and PDF files for real-time data exploration
- **🧠 Memory-Enabled Conversations** — Context-aware chat remembers your questions
- **🧪 Python REPL Integration** — Run Python code dynamically within the chat
- **📚 Quiz Generator** — Topic-specific multiple-choice quizzes on demand
- **📈 EDA & Summaries** — Generate insights and session summaries automatically
- **🌙 Dark Mode Support** — Smooth toggle with fully themed UI
- **🎨 Stylish UI** — Avatars, animations, and responsive design

---

## Tech Stack

| Technology        | Purpose                         |
|-------------------|----------------------------------|
| Streamlit         | Interactive UI                  |
| LangChain         | LLM orchestration               |
| Gemini 1.5 Pro    | Conversational AI               |
| Python REPL Tool  | Code execution & analysis       |
| Pandas            | Data handling                   |
| PyPDF2            | PDF parsing                     |

---

## Installation

```bash
# Clone the repository
https://github.com/your-username/ai-ds-tutor.git
cd ai-ds-tutor

# Optional: Set up a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## API Key Configuration

To use Gemini Pro:
1. Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Save the key to a text file (e.g., `google_api.txt`)
3. Update the file path in your script:
```python
with open("path/to/google_api.txt") as f:
    api_key = f.read().strip()
```

---

## Launch the App

```bash
streamlit run app.py
```
_Replace `app.py` with your actual filename if different._

---

## UI Preview

![chat example](./screenshots/chat_example.png)
![dark mode](./screenshots/dark_mode.png)

---

## Contributing

Contributions, ideas, and improvements are warmly welcomed! Feel free to fork this project and create a pull request.

---

> _“Learning data science should be interactive, intuitive, and accessible — this app is my step toward that vision.”_

Made with 💙 using GenAI, LangChain, and Streamlit.

