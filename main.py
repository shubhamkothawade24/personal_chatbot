import streamlit as st
import nltk
import os
import string

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ----- ⬇️ Setup: Ensure nltk_data is downloaded into the app directory -----
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download punkt (for sentence tokenization)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

# Download stopwords
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", download_dir=nltk_data_path)

# ----- 📄 Load your personal data -----
with open("data/about_me.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Tokenize sentences from your data
sentences = sent_tokenize(raw_text)

# ----- 🔧 NLP Preprocessing -----
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation])

# Preprocess all sentences
processed_sentences = [preprocess(sentence) for sentence in sentences]

# ----- 🔍 Vectorize and Match -----
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(processed_sentences)

def get_response(user_input):
    processed_input = preprocess(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarity = cosine_similarity(input_vector, sentence_vectors)
    best_match_index = similarity.argmax()
    best_score = similarity[0][best_match_index]

    if best_score < 0.2:
        return "Sorry, I couldn't find anything relevant in my knowledge."
    return sentences[best_match_index]

# ----- 💬 Streamlit Interface -----
st.title("👤 Personal Chatbot")
st.write("Ask me anything based on the information provided!")

user_input = st.text_input("Your question:")

if user_input:
    response = get_response(user_input)
    st.markdown(f"**Chatbot:** {response}")
