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
#load_dotenv()

# Load NLP Model
#nlp = spacy.load("en_core_web_sm")

# Google AI API Key
#import streamlit as st
#from google.generativeai import configure, GenerativeModel

# Get Gemini API key securely from Streamlit Secrets
GOOGLE_AI_API_KEY = st.secrets.get("GOOGLE_API_KEY")

# Show error and stop app if key isn't set
if not GOOGLE_AI_API_KEY:
    st.error("❌ Google AI API Key is missing from Streamlit Secrets!")
    st.stop()

# Configure the Gemini model
configure(api_key=GOOGLE_AI_API_KEY)
model = GenerativeModel("gemini-1.5-pro-002")



# Predefined Job Descriptions
predefined_job_descriptions = {
    "General ATS Score": "",  # New general option
    "Database Engineer": "Experience with SQL, NoSQL, database optimization, and indexing techniques.",
    "AI Engineer": "Expertise in deep learning, reinforcement learning, and frameworks like TensorFlow, PyTorch.",
    "Data Scientist": "Strong skills in Python, statistics, machine learning, and data visualization tools.",
    "ML Engineer": "Experience with machine learning pipelines, model deployment, and cloud-based ML services.",
    "Data Analyst": "Proficiency in SQL, Excel, Tableau, and Python for data analysis and visualization.",
    "Business Analyst": "Strong analytical thinking, domain expertise, and experience with BI tools.",
    "Software Engineer": "Proficient in software development, algorithms, data structures, and multiple programming languages.",
    "Cloud Engineer": "Experience with AWS, Azure, GCP, cloud security, and scalable architectures.",

    # Cybersecurity & Data Protection
    "Cybersecurity Analyst": "Expertise in penetration testing, network security, SIEM tools, and threat analysis.",
    "SOC Analyst": "Monitoring and analyzing security incidents, incident response, SIEM tools, and threat intelligence.",
    "Data Protection Officer": "Ensures compliance with data privacy laws (e.g., GDPR), conducts privacy impact assessments, and manages data protection strategies.",
    "Information Security Specialist": "Risk assessments, security policies, vulnerability management, and compliance with security standards like ISO 27001.",
    "Cloud Security Engineer": "Design and implement cloud security measures, manage identity & access controls, and ensure compliance with security best practices.",
    "Threat Intelligence Analyst": "Research and analyze cyber threats, develop threat reports, and enhance defensive security measures.",
    "Penetration Tester": "Conduct security assessments, identify vulnerabilities, and simulate cyberattacks to test security defenses.",
    
    # Compliance & Risk Management
    "Compliance Officer": "Ensures adherence to legal regulations, internal policies, and industry standards such as GDPR, HIPAA, and SOX.",
    "Risk Analyst": "Analyzes financial, operational, and cybersecurity risks, develops risk mitigation strategies, and ensures regulatory compliance.",
    "Regulatory Compliance Specialist": "Monitors and ensures compliance with industry regulations, including financial, healthcare, and IT security standards.",
    "Audit & Compliance Manager": "Conducts internal audits, identifies compliance gaps, and implements corrective actions for regulatory adherence.",
    "Ethics & Compliance Officer": "Develops corporate compliance programs, enforces ethical policies, and trains employees on regulatory requirements.",
    "Financial Compliance Analyst": "Ensures compliance with financial regulations, anti-money laundering (AML) laws, and tax laws.",
    "IT Compliance Analyst": "Assesses IT systems for compliance with security standards such as PCI-DSS, ISO 27001, and NIST frameworks.",

    # Other Technical & Engineering Roles
    "Product Manager": "Strong understanding of market research, product lifecycle, and Agile methodologies.",
    "DevOps Engineer": "Experience with CI/CD, Kubernetes, Docker, and cloud infrastructure automation.",
    "Full Stack Developer": "Proficiency in front-end and back-end development, databases, and frameworks like React, Node.js.",
    "Embedded Systems Engineer": "Experience with firmware development, microcontrollers, and real-time operating systems.",
    "UI/UX Designer": "Strong grasp of design principles, wireframing, prototyping, and tools like Figma, Adobe XD.",
    "Mechanical Engineer": "Expertise in CAD, thermodynamics, structural analysis, and material science.",
    "Electrical Engineer": "Knowledge of circuit design, power systems, embedded electronics, and signal processing.",
    "Civil Engineer": "Experience in structural engineering, construction management, and CAD software.",
    "Marketing Specialist": "Skills in digital marketing, SEO, content creation, and analytics tools.",
    "Finance Analyst": "Expertise in financial modeling, risk analysis, investment strategies, and Excel.",
    "HR Manager": "Strong understanding of talent acquisition, employee engagement, and HR policies.",

    # GIS & Geoinformatics Roles
    "GIS Analyst": "Proficiency in spatial data analysis, ArcGIS, QGIS, remote sensing, and geospatial databases.",
    "GIS Developer": "Experience in developing geospatial applications using Python, JavaScript, and GIS APIs like Leaflet and Mapbox.",
    "Remote Sensing Specialist": "Expertise in image processing, satellite data analysis, and software like ENVI, Google Earth Engine.",
    "Cartographer": "Strong skills in map design, GIS tools, and visualization of spatial data.",
    "Geospatial Data Scientist": "Experience in machine learning for geospatial data, predictive modeling, and spatial statistics.",
    "Urban Planner": "Expertise in land-use planning, geospatial analysis for smart cities, and urban development policies.",
    "Surveyor": "Experience in land surveying, GPS, total stations, and geodetic computations.",
    "Geographic Information Systems Manager": "Managing GIS projects, enterprise GIS solutions, and database administration.",
    "Hydrologist": "Use of GIS for watershed modeling, flood risk assessment, and hydrological data analysis.",
    "Environmental Scientist": "Applying GIS in environmental impact assessments, biodiversity mapping, and conservation planning.",
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
