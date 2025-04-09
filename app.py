import os
import streamlit as st
import fitz  # PyMuPDF
import docx
import re
import spacy
import pytesseract
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google.generativeai import configure, GenerativeModel

# Set Tesseract path (update if needed)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Google Gemini
GOOGLE_AI_API_KEY = st.secrets.get("GOOGLE_API_KEY")
if not GOOGLE_AI_API_KEY:
    st.error("❌ Google AI API Key is missing from Streamlit Secrets!")
    st.stop()
configure(api_key=GOOGLE_AI_API_KEY)
model = GenerativeModel("gemini-1.5-pro-002")

# --- Predefined Job Descriptions (as is, not shown here for brevity) ---

# --- Enhanced text extraction and image classification ---

def is_probably_resume_text(text):
    keywords = ['education', 'experience', 'skills', 'projects', 'certifications', 'summary']
    count = sum(1 for kw in keywords if kw in text.lower())
    return count >= 2

def extract_text(file):
    if file.type == "application/pdf":
        return "".join(page.get_text("text") + "\n" for page in fitz.open(stream=file.read(), filetype="pdf"))
    elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        if len(text.split()) < 50 or not is_probably_resume_text(text):
            return None  # Probably not a resume
        return text
    else:
        return "\n".join([para.text for para in docx.Document(file).paragraphs])

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

def match_keywords(resume_text, job_desc):
    if not job_desc:
        return 50
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    return round(cosine_similarity(vectors)[0, 1] * 100, 2)

def check_formatting(text):
    score = 100
    if len(re.findall(r'\b(education|experience|skills|projects|certifications)\b', text)) < 3:
        score -= 20
    if len(re.findall(r'\b[•|-]\b', text)) < 3:
        score -= 10
    if len(text.split()) < 200:
        score -= 20
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 10:
        score -= 10
    return max(score, 0)

def calculate_ats_score(resume_text, job_desc):
    keyword_score = match_keywords(resume_text, job_desc) * 0.5
    formatting_score = check_formatting(resume_text) * 0.3
    length_score = min(len(resume_text.split()) / 1000 * 20, 20)
    return round(keyword_score + formatting_score + length_score, 2)

def get_resume_improvements(resume_text, job_desc):
    prompt = f"""
    Analyze the following resume text and provide improvement suggestions. Consider structure, keywords, readability, and ATS compatibility.
    Resume:
    {resume_text}
    Job Description:
    {job_desc if job_desc else 'General ATS Guidelines'}
    """
    return model.generate_content(prompt).text.strip()

# --- Streamlit UI ---
st.title("📄 Resume ATS Score Checker")
user_type = st.radio("Are you a Recruiter or a Candidate?", ("Candidate", "Recruiter"))
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if user_type == "Recruiter":
    job_description = st.text_area("📌 Paste the Job Description Here")
elif user_type == "Candidate":
    job_profile = st.selectbox("Choose Your Job Role", list(predefined_job_descriptions.keys()))

if uploaded_file:
    raw_text = extract_text(uploaded_file)
    if raw_text is None or len(raw_text.strip()) == 0:
        st.error("❌ This doesn't look like a valid resume. Please upload a proper resume in PDF, DOCX, or Image format.")
    else:
        resume_text = clean_text(raw_text)
        job_desc = job_description if user_type == "Recruiter" else predefined_job_descriptions[job_profile]
        ats_score = calculate_ats_score(resume_text, job_desc)
        st.subheader(f"📊 ATS Score: {ats_score}%")
        improvements = get_resume_improvements(resume_text, job_desc)
        st.subheader("📌 Resume Improvement Suggestions")
        st.write(improvements)
