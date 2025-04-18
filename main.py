import streamlit as st
import nltk
import os
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---------- 🛠 Download NLTK resources to local folder ----------
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# ---------- 📄 Load your data ----------
with open("data/about_me.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

sentences = sent_tokenize(raw_text)

# ---------- 🧠 NLP Preprocessing ----------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation])

processed_sentences = [preprocess(sentence) for sentence in sentences]

# ---------- 🔍 TF-IDF + Cosine Similarity ----------
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(processed_sentences)

def get_response(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(input_vector, sentence_vectors)
    max_sim_index = similarity.argmax()
    max_sim_score = similarity[0][max_sim_index]

    if max_sim_score < 0.2:
        return "I'm sorry, I couldn't find anything related to that."
    return sentences[max_sim_index]

# ---------- 💬 Streamlit UI ----------
st.title("Shubham's Personal Chatbot 🤖")
st.write("Ask anything based on what's written about Shubham!")

user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.markdown(f"**Chatbot:** {response}")
