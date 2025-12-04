# Final-Year

# unified_ai_toolkit.py
import streamlit as st
import openai
import os
import datetime
import sqlite3
from zipfile import ZipFile
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI as LangOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# DB setup
conn = sqlite3.connect("interactions.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS interactions (timestamp TEXT, tool TEXT, input TEXT, output TEXT)''')

# Save interaction to DB
def db_log(tool, input_text, output):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO interactions VALUES (?, ?, ?, ?)", (now, tool, input_text, output))
    conn.commit()

# Page config
st.set_page_config(page_title="Unified AI Toolkit", layout="wide")
st.title("🧠 Final Year Project – Unified Generative AI Interface")

app = st.sidebar.selectbox("Select a Tool", (
    "Fin RAG", "Chat RAG", "Ceres Evaluation", "Eclipse Generator",
    "Shield Ranking", "Yutori Workflow", "Query Complexity", "📦 Download ZIP"
))

@st.cache_resource(show_spinner=False)
def call_gpt4(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return res["choices"][0]["message"]["content"]

@st.cache_resource(show_spinner=False)
def load_vector_store():
    return FAISS.load_local("faiss_index", OpenAIEmbeddings())

if app == "Fin RAG":
    st.header("📊 Fin RAG – Financial Claim Verifier")
    query = st.text_input("Enter financial claim")
    rubric = st.selectbox("Choose Rubric", ["Relevance", "Fact Support", "Reasoning"])
    if query:
        db = load_vector_store()
        llm = LangOpenAI(temperature=0.2)
        chain = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        response = chain.run(query)
        st.success(f"Rubric ({rubric}):\n{response}")
        db_log("Fin RAG", f"{query} [{rubric}]", response)

elif app == "Chat RAG":
    st.header("💬 Chat RAG – Contextual QA")
    user_query = st.text_input("Ask something about the document")
    rubric = st.selectbox("Conversation Evaluation Rubric", ["Contextual Fit", "Continuity", "Relevance"])
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    if user_query:
        db = load_vector_store()
        retriever = db.as_retriever()
        llm = LangOpenAI()
        memory = ConversationBufferMemory()
        chat_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
        result = chat_chain.run({"question": user_query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((user_query, result))
        st.info(f"Rubric ({rubric}):\n{result}")
        db_log("Chat RAG", user_query, result)

elif app == "Ceres Evaluation":
    st.header("📚 Ceres – Lesson Plan Evaluation")
    subject = st.selectbox("Select Subject", [
        "History", "Geography", "Civics", "Political Science", "Psychology", "Sociology", "Philosophy", "Economics",
        "Literature", "English", "Hindi", "Sanskrit", "Art", "Music", "Dance", "Drama", "Physical Education",
        "Environmental Studies", "Social Studies", "Communication Skills"
    ])
    grade = st.selectbox("Select Grade Level", ["Primary (1–5)", "Middle (6–8)", "Secondary (9–10)", "Senior Secondary (11–12)", "Undergraduate"])
    lesson = st.text_area("Paste lesson content")
    rubric = st.selectbox("Evaluation Rubric", ["Clarity", "Tone", "Completeness", "Curriculum Alignment", "Student Engagement"])
    if lesson:
        prompt = f"Evaluate this lesson based on {rubric} rubric:\n{lesson}"
        result = call_gpt4(prompt)
        st.code(result)
        db_log("Ceres Evaluation", lesson, result)

elif app == "Eclipse Generator":
    st.header("📝 Eclipse – Document Generator")
    outline = st.text_area("Outline")
    rubric = st.selectbox("Writing Rubric", ["Structure", "Coherence", "Instructional Flow"])
    if outline:
        prompt = f"Expand this into a training manual using {rubric} rubric:\n{outline}"
        result = call_gpt4(prompt)
        st.text_area("Generated Manual", result, height=300)
        db_log("Eclipse Generator", outline, result)

elif app == "Shield Ranking":
    st.header("⚖️ Shield – Compare Outputs")
    prompt = st.text_area("Main Prompt")
    a = st.text_area("Response A")
    b = st.text_area("Response B")
    rubric = st.selectbox("Comparison Criteria", ["Groundedness", "Tone", "Completeness", "Justification"])
    if st.button("Evaluate"):
        compare_prompt = f"Prompt: {prompt}\nResponse A: {a}\nResponse B: {b}\nScore both on {rubric} and justify."
        result = call_gpt4(compare_prompt)
        st.code(result)
        db_log("Shield Ranking", prompt, result)

elif app == "Yutori Workflow":
    st.header("🌿 Yutori – Wellness Workflows")
    topic = st.text_input("Topic")
    rubric = st.selectbox("Tone Rubric", ["Mindful", "Minimal", "Reflective"])
    if topic:
        prompt = f"Write a weekly workflow for {topic} using a {rubric} tone."
        result = call_gpt4(prompt)
        st.text_area("Workflow", result, height=300)
        db_log("Yutori Workflow", topic, result)

elif app == "Query Complexity":
    st.header("🧠 Prompt Difficulty Scoring")
    query = st.text_input("User Query")
    rubric = st.selectbox("Scoring Focus", ["Reasoning Depth", "Knowledge Breadth", "Inference Chain"])
    if query:
        prompt = f"Rate the complexity (1-5) of this query based on {rubric}:\n{query}"
        result = call_gpt4(prompt)
        st.write(result)
        db_log("Query Complexity", query, result)

elif app == "📦 Download ZIP":
    st.header("📦 Download Source Code")
    with ZipFile("AI_Toolkit_Full.zip", 'w') as zipf:
        zipf.write("unified_ai_toolkit.py")
        zipf.write("interactions.db")
    with open("AI_Toolkit_Full.zip", "rb") as f:
        st.download_button("Download Full Project ZIP", f, file_name="AI_Toolkit_Full.zip")


from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

loader = TextLoader("sample_10k.txt")
docs = loader.load()

embedding = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embedding)
db.save_local("faiss_index")

print("✅ FAISS index created.")
