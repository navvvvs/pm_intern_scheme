# app.py
import os
import re
import streamlit as st
import pandas as pd
import pdfplumber
from io import BytesIO
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

# ---------- Load NLP ----------
nlp = spacy.load("en_core_web_sm")

# ---------- Load Data ----------
internships = pd.read_csv("data/internships.csv")

# ---------- Streamlit Config ----------
st.set_page_config(
    page_title="AI Internship Recommender",
    page_icon="💼",
    layout="wide"
)

# ---------- Custom Styling ----------
st.markdown("""
    <style>
    .stApp {
        background-color: #FFF1E0; /* light peach */
    }

    .main {
        background-color: #fdfdfd;
        padding: 2rem;
    }
    .title {
        text-align: center;
        font-size: 3.8rem;
        font-weight: 900;
        color: #5a1e1e;
        margin-bottom: 0.3rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.8rem;
        font-weight: 600;
        color: #7a2d2d;
        margin-bottom: 2rem;
    }
    
    .stRadio > label {
        font-weight: 600;
        color: #5a1e1e;
    }
    .upload-btn > button {
        background-color: #7a2d2d;
        color: white;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1.05rem;
    }
    .upload-btn > button:hover {
        background-color: #5a1e1e;
        transform: scale(1.02);
    }
    .recommend-card {
        background: white;
        padding: 1rem 1.5rem;
        border-radius: 14px;
        box-shadow: 0 3px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 6px solid #7a2d2d;
    }
    footer {
        text-align: center;
        padding: 1rem;
        color: #6b7280;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Helper Functions ----------
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
    email = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    phone = re.findall(r"\+?\d[\d\s-]{8,}\d", text)
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    education_keywords = ["B.Tech", "BE", "M.Tech", "B.Sc", "M.Sc", "BCA", "MCA", "Diploma"]
    education_found = [word for word in education_keywords if word.lower() in text.lower()]

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

# ---------- UI ----------
st.markdown('<div class="title">AI BASED INTERNSHIP RECOMMENDATION SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">For PM Internship Scheme</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)

    st.write("Upload your resume or fill in details manually to find the best internship matches for you.")
    st.write("#### Choose Input Method:")

    mode = st.radio("", ["Upload Resume", "Manual Entry"], horizontal=True)

    resume_data = {}
    name = education = skills = location_pref = ""

    if mode == "Upload Resume":
        uploaded_file = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"], label_visibility="collapsed")
        if uploaded_file:
            st.success("File uploaded successfully")
            parsed_info = parse_cv(uploaded_file)
            st.subheader("➤ Extracted Information")
            st.json(parsed_info)
            resume_data = parsed_info
            name = parsed_info.get("name", "")
            education = ", ".join(parsed_info.get("education", []))
            skills = ", ".join(parsed_info.get("skills", []))
            location_pref = parsed_info.get("location", "")
    else:
        st.subheader("Enter / Confirm Your Details")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name", value=resume_data.get("name", ""))
            education = st.text_input("Education", value=", ".join(resume_data.get("education", [])) if resume_data else "")
        with col2:
            skills = st.text_area("Skills (comma-separated)", value=", ".join(resume_data.get("skills", [])) if resume_data else "")
            location_pref = st.text_input("Preferred Location", value=resume_data.get("location", ""))

    # Recommend Button
    st.markdown('<div class="upload-btn">', unsafe_allow_html=True)
    find_button = st.button("Find My Internships")
    st.markdown('</div>', unsafe_allow_html=True)

    # Recommendations
    if find_button:
        if not skills.strip():
            st.warning("Please enter your skills.")
        else:
            results = recommend_internships(skills, location_pref)
            st.markdown(f"### Top Internship Recommendations for **{name if name else 'You'}**")
            for r in results:
                st.markdown(f"""
                <div class="recommend-card">
                    <b>➤ Company:</b> {r['Company']} <br>
                    <b>➤ Title:</b> {r['Title']} <br>
                    <b>➤ Location:</b> {r['Location']} <br>
                    <b>➤ Duration:</b> {r['Duration']} <br>
                    <b>➤ Stipend:</b> {r['Stipend']} <br>
                    <b>✅ Match Score:</b> {r['Match Score']}%
                </div>
                """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close section box

# ---------- Footer ----------
st.markdown("<footer> Built for Smart India Hackathon 2025 | AI Internship Recommender</footer>", unsafe_allow_html=True)
