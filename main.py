import os
import streamlit as st
import nltk
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create nltk_data folder if it doesn't exist
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download necessary NLTK data safely
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# Load your bio from file
with open("data/about_me.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize sentences
sentences = sent_tokenize(raw_text)

# Preprocess for TF-IDF
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join(
        stemmer.stem(word) for word in tokens
        if word not in stop_words and word not in string.punctuation
    )

processed_sentences = [preprocess(sent) for sent in sentences]

# TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_sentences)

# Query response function
def get_response(query):
    query_proc = preprocess(query)
    query_vec = vectorizer.transform([query_proc])
    similarity = cosine_similarity(query_vec, X)
    max_idx = similarity.argmax()
    score = similarity[0][max_idx]

    if score < 0.2:
        return "🤖 Sorry, I couldn't find anything relevant in your bio."
    return sentences[max_idx]

# Streamlit interface
st.title("👤 Personal Chatbot")
st.write("Ask me questions about Shubham!")

query = st.text_input("💬 Your question:")
if query:
    response = get_response(query)
    st.write(f"**🤖 Bot:** {response}")
