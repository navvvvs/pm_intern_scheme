# recommender.py
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Utility ----------
def preprocess_skills(skills_str_or_list):
    """Convert skills string/list into normalized list of lowercase words."""
    if isinstance(skills_str_or_list, list):
        skills = skills_str_or_list
    else:
        skills = re.split(r"[;,|\n]", str(skills_str_or_list))
    return [s.strip().lower() for s in skills if s.strip()]

def calculate_score(user_skills, internship_skills, user_location, internship_location):
    """Calculate weighted score (80% skills + 20% location)."""
    cand_skills = preprocess_skills(user_skills)
    int_skills = preprocess_skills(internship_skills)

    # Skills similarity
    if not cand_skills or not int_skills:
        skill_similarity = 0
    else:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([" ".join(cand_skills), " ".join(int_skills)])
        skill_similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # Location similarity (exact match / substring match)
    cand_loc = str(user_location).lower().strip()
    int_loc = str(internship_location).lower().strip()
    location_similarity = 1 if cand_loc and cand_loc in int_loc else 0

    # Weighted final score
    return round((0.8 * skill_similarity + 0.2 * location_similarity) * 100, 2)

def recommend_internships(skills, location_pref, internships: pd.DataFrame, top_n=3):
    """Return top N internship recommendations."""
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
