import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load bio from file
with open("data/about_me.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Basic sentence splitting
sentences = [s.strip() for s in raw_text.split('.') if s.strip()]

# TF-IDF processing
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

# Get chatbot response
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, sentence_vectors)
    max_score_index = similarity_scores.argmax()
    max_score = similarity_scores[0, max_score_index]
    
    if max_score < 0.2:
        return "🤖 Sorry, I couldn't find anything related to that."
    return sentences[max_score_index]

# Streamlit UI
st.set_page_config(page_title="Shubham Chatbot", layout="centered")
st.title("🤖 Personal Chatbot")
st.write("Ask me questions based on Shubham's profile!")

user_input = st.text_input("🔎 Your question")
if user_input:
    answer = get_response(user_input)
    st.markdown(f"**🤖 Answer:** {answer}")
