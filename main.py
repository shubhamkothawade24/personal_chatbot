import streamlit as st
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Load the custom data
with open("data/about_me.txt", "r") as file:
    raw_text = file.read()

sentences = sent_tokenize(raw_text)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return " ".join([stemmer.stem(word) for word in tokens if word not in stop_words and word not in string.punctuation])

processed_sentences = [preprocess(sent) for sent in sentences]

# Vectorize
vectorizer = TfidfVectorizer()
sentence_vectors = vectorizer.fit_transform(processed_sentences)

# Response function
def get_response(user_input):
    user_input_processed = preprocess(user_input)
    user_vector = vectorizer.transform([user_input_processed])
    similarity = cosine_similarity(user_vector, sentence_vectors)
    max_sim_index = similarity.argmax()
    max_sim_value = similarity[0][max_sim_index]

    if max_sim_value < 0.2:
        return "I'm sorry, I couldn't find that information."
    else:
        return sentences[max_sim_index]

# Streamlit UI
st.title("Shubham's Personal Chatbot 🤖")
st.write("Ask anything related to Shubham!")

user_input = st.text_input("You:", "")

if user_input:
    response = get_response(user_input)
    st.markdown(f"**Chatbot:** {response}")
