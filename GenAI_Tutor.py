import streamlit as st

# Set page config Streamlit command
st.set_page_config(page_title="AI Conversational Data Science Tutor", layout="wide")

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.agents.agent_types import AgentType
from langchain.llms import GooglePalm
import pandas as pd
import traceback
import os
import random
import PyPDF2

# Set page config
from streamlit_extras.switch_page_button import switch_page

# Dark mode toggle + theme switch
def set_theme(dark_mode):
    if dark_mode:
        st.markdown("""
        <style>
        html, body, .stApp {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        .css-1d391kg, .css-1v3fvcr, .css-1v0mbdj, .stText, .stMarkdown, .stDataFrame {
            color: #fafafa !important;
        }
        .sidebar .sidebar-content, .css-1d391kg, .stSidebarContent {
            background-color: #1c1f26 !important;
        }
        .stChatInput input {
            background-color: #20232a !important;
            color: #fafafa !important;
        }
        .stButton button, .stDownloadButton button {
            background-color: #292b34 !important;
            color: #fafafa !important;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
        html, body, .stApp {
            background-color: #f5f7fa !important;
            color: #000000 !important;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6 !important;
        }
        .stChatInput input {
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)

# Dark mode toggle (must come after page config)
dark_mode = st.sidebar.toggle("🌙 Dark Mode")
set_theme(dark_mode)

# Custom header with emoji and style
st.markdown("""
<div style='text-align: center; padding: 1rem;'>
    <h1 style='font-size: 2.5rem;'>🤖 AI Conversational <span style='color:#ff4b4b;'>Data Science</span> Tutor</h1>
    <p style='font-size: 1.1rem; color: grey;'>Your intelligent assistant for all things data science — from fundamentals to code execution.</p>
</div>
""", unsafe_allow_html=True)


# Initialize session state early
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Sidebar options
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2721/2721260.png", width=100)

st.sidebar.header("🧭 Navigation")
mode = st.sidebar.radio("Select Mode:", ["Beginner", "Advanced"])

# Quick Topics
topics = [
    "", "Data Cleaning", "Feature Engineering", "Linear Regression", "Logistic Regression",
    "K-Means Clustering", "PCA", "Random Forest", "SVM", "Time Series Forecasting",
    "Neural Networks", "CNN", "LSTM", "XGBoost", "Data Visualization", "Model Evaluation",
    "SHAP", "SQL for Data Science", "Big Data Tools", "Pandas", "Matplotlib", "Explainable AI"
]
topic = st.sidebar.selectbox("Choose a topic:", topics)
if topic:
    st.session_state.pre_fill = f"Explain {topic} with examples"
else:
    st.session_state.pre_fill = ""

# Contextual Tip Based on Conversation
last_user_message = ""
for msg in reversed(st.session_state.chat_history):
    if isinstance(msg, HumanMessage):
        last_user_message = msg.content.lower()
        break

contextual_tip = "Use cross-validation to ensure your model generalizes well."
if "regression" in last_user_message:
    contextual_tip = "Try using R-squared and RMSE to evaluate regression models."
elif "classification" in last_user_message:
    contextual_tip = "Confusion matrix, precision, and recall are key metrics for classification."
elif "clustering" in last_user_message:
    contextual_tip = "Use silhouette score to evaluate clustering quality."
elif "feature" in last_user_message:
    contextual_tip = "Feature engineering can boost model performance significantly."
elif "eda" in last_user_message or "exploratory" in last_user_message:
    contextual_tip = "Start your EDA by checking missing values and summary statistics."
elif "time series" in last_user_message:
    contextual_tip = "Always check for seasonality and stationarity in time series data."

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip Based on Your Conversation:**")
st.sidebar.info(contextual_tip)

# Quiz Me Feature
st.sidebar.markdown("---")
st.sidebar.markdown("🧠 **Topic-Based Quiz Generator**")
selected_quiz_topic = st.sidebar.selectbox("Select a topic for quiz:", [""] + topics[1:])
if selected_quiz_topic:
    if st.sidebar.button("Generate Quiz"):
        st.session_state.pre_fill = f"Give me a multiple-choice quiz on {selected_quiz_topic}."

# File uploader for CSV and PDF
uploaded_file = st.sidebar.file_uploader("📂 Upload a CSV or PDF file", type=["csv", "pdf"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())
        st.session_state.df = df
        st.session_state.pre_fill = "What insights can I gain from this dataset?"
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        pdf_text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        st.subheader("📄 PDF Content Preview")
        st.text_area("PDF Text Extracted:", pdf_text[:3000], height=300)
        st.session_state.pdf_text = pdf_text
        st.session_state.pre_fill = "Summarize the key points from the uploaded PDF."

# Configure Google AI API Key
with open(r"C:\Users\nafeesa hasan\Downloads\KEYS\google_api.txt") as f:
    api_key = f.read().strip()

genai.configure(api_key=api_key)

# Memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Model
llm = ChatGoogleGenerativeAI(model= "gemini-1.5-pro", api_key=api_key)

# Python REPL Tool
tools = [
    Tool(
        name="Python REPL",
        func=PythonREPLTool().run,
        description="Use this for Python code execution and DataFrame analysis."
    )
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=st.session_state.memory,
    verbose=False
)

# System prompt based on mode
system_prompt = (
    "You are a helpful and expert Data Science tutor. Answer only Data Science-related questions. "
    "If a question is not related to Data Science, politely decline. "
    + ("Explain like I'm new to the topic." if mode == "Beginner" else "Use advanced technical language.")
)

# Add system message once
if not any(isinstance(msg, SystemMessage) for msg in st.session_state.chat_history):
    st.session_state.chat_history.append(SystemMessage(content=system_prompt))

# Input box
st.markdown("""
<style>
    .stChatInput input {
        border-radius: 10px;
        padding: 0.75rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)
user_input = st.chat_input("Ask your Data Science question here...")

# Display chat history
user_avatar = "https://cdn-icons-png.flaticon.com/512/1946/1946429.png"
ai_avatar = "https://cdn-icons-png.flaticon.com/512/4712/4712033.png"

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar=user_avatar):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar=ai_avatar):
            st.markdown(message.content)

# Process user input
# If a PDF was uploaded, allow summarization or Q&A based on it
if user_input and uploaded_file and uploaded_file.name.endswith(".pdf"):
    user_input = f"""Based on this PDF content:
{st.session_state.get('pdf_text', '')}

User question: {user_input}"""
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    try:
        response = agent.run(user_input)
    except Exception as e:
        error_trace = traceback.format_exc()
        response = f"⚠️ Unexpected error: {str(e)}\n\n```\n{error_trace}\n```"

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=response))

# Clear Chat
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Session Tools")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.memory.clear()
    st.experimental_rerun()

# Summarize Conversation
if st.sidebar.button("🧠 Summarize Conversation"):
    history_text = "\n".join([msg.content for msg in st.session_state.chat_history if isinstance(msg, (HumanMessage, AIMessage))])
    summary_prompt = f"Summarize this Data Science tutoring session:\n{history_text}"
    summary = llm.predict(summary_prompt)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Session Summary")
    st.sidebar.info(summary)

# Download Transcript
if st.sidebar.button("📥 Download Transcript"):
    transcript = "\n\n".join([
        f"User: {msg.content}" if isinstance(msg, HumanMessage) else f"Tutor: {msg.content}"
        for msg in st.session_state.chat_history
    ])
    st.download_button("Download as TXT", data=transcript, file_name="chat_transcript.txt")
