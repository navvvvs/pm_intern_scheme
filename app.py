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
import google.generativeai as genai

# ---------- Configure Gemini ----------
genai.configure(api_key=os.getenv("AIzaSyCQ8h1NWy67h_c3af2kdovQ7JR-2osM2VA", ""))
MODEL = genai.GenerativeModel("gemini-pro")

# ---------- Load Data ----------
internships = pd.read_csv("data/internships.csv")

st.set_page_config(page_title="AI Internship Recommender", layout="centered")
st.title("AI-Based Internship Recommendation System")
st.write("Prototype for Smart India Hackathon 2025")

# ---------- Helpers ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end+1])
            except:
                pass
    return None

def call_llm_extract_resume(resume_text: str) -> dict:
    prompt = f"""
    Extract the following from resume text and return JSON only:
    - Name
    - Age
    - Education
    - Skills (as a list)
    - Location

    Resume Text:
    {resume_text}
    """
    try:
        response = MODEL.generate_content(prompt)
        parsed = safe_parse_json(response.text)
        if parsed:
            return parsed
    except:
        pass
    return {"Name": None, "Age": None, "Education": None, "Skills": [], "Location": None}

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

# ---------- UI Options ----------
mode = st.radio("Choose Input Method:", ["📄 Upload Resume", "✍️ Manual Entry"])

resume_data = {}

if mode == "📄 Upload Resume":
    uploaded = st.file_uploader("Upload your Resume (PDF)", type=["pdf"])
    if uploaded:
        text = extract_text_from_pdf(uploaded.read())
        st.success("✅ Resume uploaded successfully")
        if text.strip():
            with st.spinner("Extracting details with AI..."):
                resume_data = call_llm_extract_resume(text)
            st.json(resume_data)
        else:
            st.warning("Could not extract text from PDF.")

# Manual input (fallback or user choice)
st.subheader("Enter / Confirm Your Details")

name = st.text_input("Name", value=resume_data.get("Name", "") if resume_data else "")
education = st.text_input("Education", value=resume_data.get("Education", "") if resume_data else "")
skills = st.text_area("Skills (comma-separated)", value=", ".join(resume_data.get("Skills", [])) if resume_data else "")
location_pref = st.text_input("Preferred Location", value=resume_data.get("Location", "") if resume_data else "")

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
