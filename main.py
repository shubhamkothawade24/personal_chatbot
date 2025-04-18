import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os

# Load bio
with open("data/about_me.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Split into sentences (answers)
sentences = [s.strip() for s in raw_text.split('.') if s.strip()]

# Load transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed bio sentences
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Streamlit UI
st.set_page_config(page_title="Personal Chatbot", layout="centered")
st.title("🤖Shubham's Personal Chatbot")
st.write("Ask questions based on Shubham's bio!")

user_question = st.text_input("💬 Ask something:")

if user_question:
    # Embed user query
    query_embedding = model.encode(user_question, convert_to_tensor=True)

    # Find most similar sentence from the bio
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]
    best_score = float(scores.max())
    best_index = int(scores.argmax())

    if best_score < 0.3:
        st.write("🤖 Sorry, I couldn't find anything related to that.")
    else:
        st.write(f"**🤖 Answer:** {sentences[best_index]}")
