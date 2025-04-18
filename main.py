import streamlit as st
import nltk
import os
import string

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🛠 Download and point to local nltk_data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download punkt and stopwords if not already
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

# 📄 Load your custom text
with open("data/about_me.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

sentences = sent_tokenize(raw_text)

# 🧠 NLP Preprocessing
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words and w not in string.punctuation])

processed_sentences = [preprocess(sent) for sent in sentences]

vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(processed_sentences)

# 💬 Chatbot Response Logic
def get_response(user_input):
    input_vec = vectorizer.transform([preprocess(user_input)])
    similarity = cosine_similarity(input_vec, sentence_vectors)
    max_sim_index = similarity.argmax()
    max_score = similarity[0][max_sim_index]
    if max_score < 0.2:
        return "Sorry, I couldn't find relevant info."
    return sentences[max_sim_index]

# 🌐 Streamlit UI
st.title("👋 Ask About Shubham")
user_input = st.text_input("Ask something:")

if user_input:
    st.markdown("**Answer:** " + get_response(user_input))
