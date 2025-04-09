import os
import re
import fitz  # PyMuPDF
import docx
import pytesseract
import streamlit as st
from PIL import Image
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.generativeai import configure, GenerativeModel

# Configure environment
os.environ["MONGO_URI"] = "disabled"
os.environ["MONGODB_URI"] = "disabled"

# Set up tesseract path for OCR
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Load Google AI API Key
GOOGLE_AI_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_AI_API_KEY:
    st.error("❌ Google AI API Key is missing from Streamlit Secrets!")
    st.stop()

configure(api_key=GOOGLE_AI_API_KEY)
model = GenerativeModel("gemini-1.5-pro-002")

# Predefined job descriptions (shortened for brevity)
predefined_job_descriptions = {
    "General ATS Score": "",
    "Data Scientist": "Strong skills in Python, statistics, machine learning, and data visualization tools.",
    "Software Engineer": "Proficient in software development, algorithms, data structures, and multiple programming languages.",
    # Add more roles as needed
}

# Text extraction from various file types
def extract_text(file):
    if file.type == "application/pdf":
        return "".join(page.get_text("text") for page in fitz.open(stream=file.read(), filetype="pdf"))
    elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
        return pytesseract.image_to_string(Image.open(file))
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "\n".join([para.text for para in docx.Document(file).paragraphs])
    return ""

# Simple resume detector
RESUME_KEYWORDS = ["experience", "education", "skills", "projects", "certifications", "responsibilities", "summary"]

def is_resume(text):
    match_count = sum(1 for keyword in RESUME_KEYWORDS if keyword in text.lower())
    return match_count >= 2

# Text cleaner
def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).strip()

# Keyword match score
def match_keywords(resume_text, job_desc):
    if not job_desc:
        return 50
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    return round(cosine_similarity(vectors)[0, 1] * 100, 2)

# Formatting score
def check_formatting(text):
    score = 100
    if len(re.findall(r'\b(education|experience|skills|projects|certifications)\b', text)) < 3:
        score -= 20
    if len(re.findall(r'[\u2022\-|\*]', text)) < 3:
        score -= 10
    if len(text.split()) < 200:
        score -= 20
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 10:
        score -= 10
    return max(score, 0)

# ATS score calculator
def calculate_ats_score(resume_text, job_desc):
    keyword_score = match_keywords(resume_text, job_desc) * 0.5
    formatting_score = check_formatting(resume_text) * 0.3
    length_score = min(len(resume_text.split()) / 1000 * 20, 20)
    return round(keyword_score + formatting_score + length_score, 2)

# AI-based improvement suggestions
def get_resume_improvements(resume_text, job_desc):
    prompt = f"""
    Analyze the following resume text and provide improvement suggestions. Consider structure, keywords, readability, and ATS compatibility.
    Resume:
    {resume_text}
    Job Description:
    {job_desc if job_desc else 'General ATS Guidelines'}
    """
    return model.generate_content(prompt).text.strip()

# Streamlit UI
st.title("\U0001F4C4 Resume ATS Score Checker")
user_type = st.radio("Are you a Recruiter or a Candidate?", ("Candidate", "Recruiter"))
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if user_type == "Recruiter":
    job_description = st.text_area("\U0001F4CC Paste the Job Description Here")
elif user_type == "Candidate":
    job_profile = st.selectbox("Choose Your Job Role", list(predefined_job_descriptions.keys()))

if uploaded_file:
    extracted_text = extract_text(uploaded_file)
    cleaned_text = clean_text(extracted_text)

    if not is_resume(cleaned_text):
        st.error("🚫 The uploaded file does not appear to be a resume. Please upload a valid resume file.")
    else:
        job_desc = job_description if user_type == "Recruiter" else predefined_job_descriptions[job_profile]
        ats_score = calculate_ats_score(cleaned_text, job_desc)

        st.subheader(f"\U0001F4CA ATS Score: {ats_score}%")

        improvements = get_resume_improvements(cleaned_text, job_desc)
        st.subheader("\U0001F4CC Resume Improvement Suggestions")
        st.write(improvements)
