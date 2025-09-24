# app.py
import os
import json
import re
import streamlit as st
import pandas as pd
import pdfplumber
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# ---------- Load NLP ----------
nlp = spacy.load("en_core_web_sm")

# ---------- Configure Gemini ----------
# genai.configure(api_key=os.getenv("AIzaSyCQ8h1NWy67h_c3af2kdovQ7JR-2osM2VA", ""))
# MODEL = genai.GenerativeModel("gemini-pro")

# ---------- Load Data ----------
internships = pd.read_csv("data/internships.csv")

st.set_page_config(page_title="AI Internship Recommender", layout="centered")
st.title("AI-Based Internship Recommendation System")
st.write("Prototype for Smart India Hackathon 2025")

# ---------- Helpers ----------
# def extract_text_from_pdf(file_bytes: bytes) -> str:
#     text = ""
#     with pdfplumber.open(BytesIO(file_bytes)) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text

# def safe_parse_json(text: str):
#     try:
#         return json.loads(text)
#     except:
#         start = text.find("{")
#         end = text.rfind("}")
#         if start != -1 and end != -1:
#             try:
#                 return json.loads(text[start:end+1])
#             except:
#                 pass
#     return None

# def call_llm_extract_resume(resume_text: str) -> dict:
#     prompt = f"""
#     Extract the following from resume text and return JSON only:
#     - Name
#     - Age
#     - Education
#     - Skills (as a list)
#     - Location

#     Resume Text:
#     {resume_text}
#     """
#     try:
#         response = MODEL.generate_content(prompt)
#         parsed = safe_parse_json(response.text)
#         if parsed:
#             return parsed
#     except:
#         pass
#     return {"Name": None, "Age": None, "Education": None, "Skills": [], "Location": None}


# ---------- CV Parsing Functions ----------
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    elif file.name.endswith(".docx"):
        doc = Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")
    return text

def parse_cv(file):
    text = extract_text(file).strip()

    # Regex
    email = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s-]{8,}\d", text)

    # NLP
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    education_keywords = ["B.Tech", "BE", "M.Tech", "B.Sc", "M.Sc", "BCA", "MCA", "Diploma"]
    education_found = [word for word in education_keywords if word.lower() in text.lower()]

    # Skills from dataset
    all_skills = set()
    for skills in internships["skills_required"].dropna():
        for skill in skills.split(";"):
            all_skills.add(skill.strip())
    skills_found = [skill for skill in all_skills if skill.lower() in text.lower()]

    tech_words = ["Flask", "Python", "Django", "TensorFlow", "Docker", "Git", "Postman"]
    locations = [ent.text for ent in doc.ents if ent.label_ == "GPE" and ent.text not in tech_words]

    return {
        "name": names[0] if names else None,
        "email": email[0] if email else None,
        "phone": phone[0] if phone else None,
        "education": education_found,
        "location": locations[0] if locations else None,
        "skills": skills_found
    }

def preprocess_skills(skills_str_or_list):
    if isinstance(skills_str_or_list, list):
        skills = skills_str_or_list
    else:
        skills = re.split(r"[;,|\n]", str(skills_str_or_list))
    return [s.strip().lower() for s in skills if s.strip()]

def calculate_score(user_skills, internship_skills, user_location, internship_location):
    cand_skills = preprocess_skills(user_skills)
    int_skills = preprocess_skills(internship_skills)

    if not cand_skills or not int_skills:
        skill_similarity = 0
    else:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([" ".join(cand_skills), " ".join(int_skills)])
        skill_similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    cand_loc = str(user_location).lower().strip()
    int_loc = str(internship_location).lower().strip()
    location_similarity = 1 if cand_loc and cand_loc in int_loc else 0

    return round((0.8 * skill_similarity + 0.2 * location_similarity) * 100, 2)

def recommend_internships(skills, location_pref, top_n=3):
    recommendations = []
    for _, row in internships.iterrows():
        score = calculate_score(skills, row["skills_required"], location_pref, row["location"])
        recommendations.append({
            "Company": row["company"],
            "Title": row["title"],
            "Location": row["location"],
            "Duration": row["duration"],
            "Stipend": row["stipend"],
            "Match Score": score
        })
    return sorted(recommendations, key=lambda x: x["Match Score"], reverse=True)[:top_n]

# ---------- Streamlit UI ----------
mode = st.radio("Choose Input Method:", ["📄 Upload Resume", "✍️ Manual Entry"])

# Initialize variables to avoid NameError
name = ""
education = ""
skills = ""
location_pref = ""

resume_data = {}

if mode == "📄 Upload Resume":
    uploaded_file = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file:
        st.success("✅ File uploaded successfully")
        parsed_info = parse_cv(uploaded_file)
        st.subheader("Extracted Information")
        st.json(parsed_info)
        resume_data = parsed_info

        # Assign parsed values to variables
        name = parsed_info.get("name", "")
        education = ", ".join(parsed_info.get("education", []))
        skills = ", ".join(parsed_info.get("skills", []))
        location_pref = parsed_info.get("location", "")

else:  # Manual entry
    st.subheader("Enter / Confirm Your Details")
    name = st.text_input("Name", value=resume_data.get("name", "") if resume_data else "")
    education = st.text_input("Education", value=", ".join(resume_data.get("education", [])) if resume_data else "")
    skills = st.text_area("Skills (comma-separated)", value=", ".join(resume_data.get("skills", [])) if resume_data else "")
    location_pref = st.text_input("Preferred Location", value=resume_data.get("location", "") if resume_data else "")

# ---------- Recommend ----------
if st.button("Find My Internships"):
    if not skills.strip():
        st.warning("⚠️ Please enter your skills.")
    else:
        results = recommend_internships(skills, location_pref)
        st.write(f"### 🔍 Top Internship Recommendations for {name if name else 'You'}:")
        for r in results:
            st.markdown(f"""
            ---
            **🏢 Company:** {r['Company']}  
            **💼 Title:** {r['Title']}  
            **📍 Location:** {r['Location']}  
            **⏳ Duration:** {r['Duration']}  
            **💰 Stipend:** {r['Stipend']}  
            **✅ Match Score:** {r['Match Score']}%
            """)
