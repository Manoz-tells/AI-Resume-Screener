import streamlit as st
import PyPDF2
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# PDF Text Extraction Function
def extract_text_from_pdf(uploaded_file):

    text = ""

    reader = PyPDF2.PdfReader(uploaded_file)

    for page in reader.pages:
        text += page.extract_text()

    return text


# Text Preprocessing Function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))

    filtered_words = [
        word for word in words
        if word not in stop_words
    ]

    # Join cleaned words
    cleaned_text = ' '.join(filtered_words)

    return cleaned_text


# Streamlit UI
st.title("AI Resume Screening System")

st.write("Upload your resume and compare it with a job description.")


# Upload Resume
uploaded_file = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


# Job Description Input
job_description = st.text_area(
    "Paste Job Description"
)


# Analyze Button
if st.button("Analyze Resume"):

    if uploaded_file is not None and job_description != "":

        # Extract PDF Text
        resume_text = extract_text_from_pdf(uploaded_file)

        # Preprocess Text
        cleaned_resume = preprocess_text(resume_text)
        cleaned_job = preprocess_text(job_description)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()

        documents = [cleaned_resume, cleaned_job]

        tfidf_matrix = vectorizer.fit_transform(documents)

        # Cosine Similarity
        similarity_score = cosine_similarity(
            tfidf_matrix[0:1],
            tfidf_matrix[1:2]
        )

        # Match Percentage
        match_percentage = similarity_score[0][0] * 100

        # Display Score
        st.success(
            f"Resume Match Score: {match_percentage:.2f}%"
        )

    else:
        st.warning(
            "Please upload resume and enter job description."
        )