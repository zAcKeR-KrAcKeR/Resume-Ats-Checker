import os
os.environ["MONGO_URI"] = "disabled"
os.environ["MONGODB_URI"] = "disabled"

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
from dotenv import load_dotenv

import subprocess
import importlib.util

##   model_name = "en_core_web_sm"
  #  if importlib.util.find_spec(model_name) is None:
 #       subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)

#ensure_spacy_model()


# Ensure the model is available
#try:
   # nlp = spacy.load("en_core_web_sm")
#except OSError:
   # subprocess.run(["python", "-m", "spacy", "download", "en-core-web-sm"], check=True)
   # nlp = spacy.load("en_core_web_sm")


# Ensure SpaCy model is installed before loading
#model_name = "en_core_web_sm"

#if not spacy.util.is_package(model_name):
    #print(f"Downloading SpaCy model: {model_name}...")
    #os.system(f"python -m spacy download {model_name}")

#nlp = spacy.load(model_name)


# Set Tesseract path manually for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Verify if Tesseract is found
print("Tesseract Version:", pytesseract.get_tesseract_version()) 

# Enable this for windows Manually set Tesseract path
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

print(pytesseract.get_tesseract_version())  # Verify in Python

# Load environment variables
load_dotenv()

# Load NLP Model
#nlp = spacy.load("en_core_web_sm")

# Google AI API Key
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
if not GOOGLE_AI_API_KEY:
    raise ValueError("Google AI API Key is missing! Add it to your .env file.")
configure(api_key=GOOGLE_AI_API_KEY)
model = GenerativeModel("gemini-1.5-pro-002")

# Predefined Job Descriptions
predefined_job_descriptions = {
    "General ATS Score": "",
    "Database Engineer": "Experience with SQL, NoSQL, database optimization, and indexing techniques.",
    "AI Engineer": "Expertise in deep learning, reinforcement learning, and frameworks like TensorFlow, PyTorch.",
    "Data Scientist": "Strong skills in Python, statistics, machine learning, and data visualization tools.",
    "GIS Analyst": "Proficiency in spatial data analysis, ArcGIS, QGIS, remote sensing, and geospatial databases.",
    "GIS Developer": "Experience in developing geospatial applications using Python, JavaScript, and GIS APIs.",
    "Remote Sensing Specialist": "Expertise in image processing, satellite data analysis, and GIS tools like Google Earth Engine.",
    "Cartographer": "Strong skills in map design, GIS tools, and visualization of spatial data.",
    "Geospatial Data Scientist": "Experience in ML for geospatial data, predictive modeling, and spatial statistics.",
}

# Extract text from different formats
def extract_text(file):
    if file.type == "application/pdf":
        return "".join(page.get_text("text") + "\n" for page in fitz.open(stream=file.read(), filetype="pdf"))
    elif file.type in ["image/png", "image/jpeg", "image/jpg"]:
        return pytesseract.image_to_string(Image.open(file))
    else:
        return "\n".join([para.text for para in docx.Document(file).paragraphs])

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text.strip()

# Calculate keyword match score
def match_keywords(resume_text, job_desc):
    if not job_desc:
        return 50  # Default score for general resumes
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, job_desc])
    return round(cosine_similarity(vectors)[0, 1] * 100, 2)

# Improved formatting check
def check_formatting(text):
    score = 100
    if len(re.findall(r'\b(education|experience|skills|projects|certifications)\b', text)) < 3:
        score -= 20  # Missing key sections
    if len(re.findall(r'\b[•|-]\b', text)) < 3:
        score -= 10  # No bullet points
    if len(text.split()) < 200:
        score -= 20  # Too short
    if len(re.findall(r'\b[A-Z]{3,}\b', text)) > 10:
        score -= 10  # Too many uppercase words
    return max(score, 0)

# Enhanced ATS score calculation
def calculate_ats_score(resume_text, job_desc):
    keyword_score = match_keywords(resume_text, job_desc) * 0.5
    formatting_score = check_formatting(resume_text) * 0.3
    length_score = min(len(resume_text.split()) / 1000 * 20, 20)
    return round(keyword_score + formatting_score + length_score, 2)

# Generate resume improvement suggestions
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
st.title("📄 Resume ATS Score Checker")
user_type = st.radio("Are you a Recruiter or a Candidate?", ("Candidate", "Recruiter"))
uploaded_file = st.file_uploader("Upload Resume (PDF, DOCX, or Image)", type=["pdf", "docx", "png", "jpg", "jpeg"])

if user_type == "Recruiter":
    job_description = st.text_area("📌 Paste the Job Description Here")
elif user_type == "Candidate":
    job_profile = st.selectbox("Choose Your Job Role", list(predefined_job_descriptions.keys()))

if uploaded_file:
    resume_text = clean_text(extract_text(uploaded_file))
    job_desc = job_description if user_type == "Recruiter" else predefined_job_descriptions[job_profile]
    ats_score = calculate_ats_score(resume_text, job_desc)
    st.subheader(f"📊 ATS Score: {ats_score}%")
    
    # Generate resume improvement suggestions
    improvements = get_resume_improvements(resume_text, job_desc)
    st.subheader("📌 Resume Improvement Suggestions")
    st.write(improvements)
