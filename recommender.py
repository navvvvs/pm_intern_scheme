import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Helpers ----------
def preprocess_skills(skills_str):
    """Split skills on comma or semicolon, lowercase and strip"""
    skills = re.split(r'[;,]', str(skills_str))
    return [s.strip().lower() for s in skills if s.strip()]

def calculate_score(candidate, internship):
    """Compute weighted score based on skills + location"""
    # --- Skills similarity ---
    cand_skills = preprocess_skills(candidate['Skills'])
    int_skills = preprocess_skills(internship['skills_required'])

    skill_texts = [" ".join(cand_skills), " ".join(int_skills)]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(skill_texts)
    skill_similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # --- Location similarity (substring check) ---
    cand_loc = str(candidate['Location']).lower()
    int_loc = str(internship['location']).lower()
    location_similarity = 1 if cand_loc in int_loc else 0

    # --- Final weighted score ---
    final_score = (0.8 * skill_similarity) + (0.2 * location_similarity)
    return round(final_score * 100, 2)  # return as percentage

def recommend_internships(candidate_id, candidates_df, internships_df, top_n=5):
    """Recommend internships for a given candidate"""
    candidate = candidates_df[candidates_df['CandidateID'] == candidate_id].iloc[0]

    recommendations = []
    for _, internship in internships_df.iterrows():
        score = calculate_score(candidate, internship)
        recommendations.append({
            "InternshipID": internship['internship_id'],
            "Company": internship['company'],
            "Title": internship['title'],
            "Location": internship['location'],
            "Duration": internship['duration'],
            "Stipend": internship['stipend'],
            "Match Score": f"{score}%"
        })

    # Sort by match score
    recommendations = sorted(recommendations, key=lambda x: float(x["Match Score"][:-1]), reverse=True)

    return recommendations[:top_n]


# ---------- Run Test ----------
if __name__ == "__main__":
    # Load CSVs
    candidates_df = pd.read_csv("candidates.csv")
    internships_df = pd.read_csv("internships.csv")

    # Example: Recommend for Aarav (CAND001)
    recs = recommend_internships("CAND001", candidates_df, internships_df, top_n=5)

    print(f"\nTop recommendations for {candidates_df.loc[0, 'Name']} ({candidates_df.loc[0, 'Location']}):\n")
    for r in recs:
        print(f"🏢 {r['Company']} | 💼 {r['Title']} | 📍 {r['Location']} | "
              f"⏳ {r['Duration']} | 💰 {r['Stipend']} | ✅ Match Score: {r['Match Score']}")

