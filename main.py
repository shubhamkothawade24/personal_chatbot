import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load bio from file ---
with open("data/about_me.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# --- Preprocess into sentences ---
sentences = [s.strip() for s in raw_text.split('.') if s.strip()]

# --- TF-IDF processing ---
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(sentences)

# --- Get chatbot response ---
def get_response(user_input):
    user_vec = vectorizer.transform([user_input])
    similarity_scores = cosine_similarity(user_vec, sentence_vectors)
    max_score_index = similarity_scores.argmax()
    max_score = similarity_scores[0, max_score_index]
    
    if max_score < 0.2:
        return "🤖 Sorry, I couldn't find anything related to that."
    return sentences[max_score_index]

# --- Initialize Streamlit ---
st.set_page_config(page_title="Shubham Chatbot", layout="centered")
st.title("🤖 Personal Chatbot")
st.write("Ask me anything based on Shubham's bio!")

# --- Session state for conversation ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Input box ---
user_input = st.text_input("🔎 Ask your question", key="input_box")

# --- Process input ---
if user_input:
    bot_response = get_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_response))
    st.session_state.input_box = ""  # clear input

# --- Display chat history ---
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")
