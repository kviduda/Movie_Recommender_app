
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
import requests
import time

# Load data
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# Load model
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# TMDB poster fetch function
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    try:
        data = requests.get(url).json()
        poster_path = data.get("poster_path", "")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

# Generate recommendations
user_ids = ratings['userId'].unique()
movie_ids = movies['movieId'].tolist()
recommendations = []

print("Generating recommendations", end="", flush=True)

for count, user_id in enumerate(user_ids):
    rated = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    candidates = [mid for mid in movie_ids if mid not in rated]
    preds = [(mid, model.predict(user_id, mid).est) for mid in candidates]
    top_n = sorted(preds, key=lambda x: x[1], reverse=True)[:5]

    for movie_id, _ in top_n:
        title = movies[movies['movieId'] == movie_id]['title'].values[0]
        poster = fetch_poster(movie_id)
        recommendations.append({"userId": user_id, "title": title, "poster_url": poster})
    
    if count % 5 == 0:
        print(".", end="", flush=True)

# Save to CSV
df = pd.DataFrame(recommendations)
df.to_csv("user_recommendations.csv", index=False)
print("\n✅ user_recommendations.csv created.")
