import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Load Data ----------
internships = pd.read_csv("data/internships.csv")

st.title("AI-Based Internship Recommendation System")
st.write("Prototype for Smart India Hackathon 2025")

# ---------- User Input ----------
st.subheader("Enter Your Details")

name = st.text_input("Name")
education = st.text_input("Education (e.g., B.Tech CSE, MBA, etc.)")
skills = st.text_area("Skills (comma-separated, e.g., Python, Machine Learning, Pandas)")
preferred_domain = st.text_input("Preferred Domain (e.g., Data Science, Web Development)")
location_pref = st.text_input("Preferred Location (optional)")


# ---------- Helpers ----------
def preprocess_skills(skills_str):
    """Split skills on comma/semicolon, lowercase, and strip spaces."""
    skills = re.split(r"[;,]", str(skills_str))
    return [s.strip().lower() for s in skills if s.strip()]


def calculate_score(user_skills, internship_skills, user_location, internship_location):
    """Compute weighted score based on skills + location"""

    # --- Skills similarity ---
    cand_skills = preprocess_skills(user_skills)
    int_skills = preprocess_skills(internship_skills)

    skill_texts = [" ".join(cand_skills), " ".join(int_skills)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(skill_texts)
    skill_similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # --- Location similarity ---
    cand_loc = str(user_location).lower().strip()
    int_loc = str(internship_location).lower().strip()
    location_similarity = 1 if cand_loc and cand_loc in int_loc else 0

    # --- Final weighted score ---
    final_score = (0.8 * skill_similarity) + (0.2 * location_similarity)
    return round(final_score * 100, 2)  # percentage


def recommend_internships(user_skills, user_location, top_n=3):
    """Recommend internships for given user"""
    recommendations = []

    for _, row in internships.iterrows():
        score = calculate_score(
            user_skills,
            row["skills_required"],
            user_location,
            row["location"]
        )
        recommendations.append({
            "Company": row["company"],
            "Title": row["title"],
            "Location": row["location"],
            "Duration": row["duration"],
            "Stipend": row["stipend"],
            "Match Score": score
        })

    # Sort by match score
    recommendations = sorted(recommendations, key=lambda x: x["Match Score"], reverse=True)

    return recommendations[:top_n]


# ---------- Button Action ----------
if st.button("Find My Internships"):
    if skills.strip() == "":
        st.warning("⚠️ Please enter at least your skills to get recommendations.")
    else:
        results = recommend_internships(skills, location_pref)

        st.write(f"### 🔍 Top Internship Recommendations for {name if name else 'You'}:")

        # Show results as cards
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
