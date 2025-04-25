
import streamlit as st
import pandas as pd
import pickle
import requests

# TMDB API key (replace with your own if needed)
API_KEY = "8265bd1679663a7ea12ac168da84d2e8"

# Fetch poster image from TMDB
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path', '')
    if poster_path:
        return f"https://image.tmdb.org/t/p/w500/{poster_path}"
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"

# Recommend top-N movies for a given user
def recommend_for_user(user_id, model, movies_df, ratings_df, n=5):
    movie_ids = movies_df['movieId'].tolist()
    already_rated = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    candidates = [mid for mid in movie_ids if mid not in already_rated]

    predictions = [(mid, model.predict(user_id, mid).est) for mid in candidates]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    recommended_titles = []
    recommended_posters = []

    for movie_id, _ in top_n:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
        poster = fetch_poster(movie_id)
        recommended_titles.append(title)
        recommended_posters.append(poster)

    return recommended_titles, recommended_posters

# Load data and model
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

# Streamlit UI
st.title("🎬 Personalized Movie Recommendations")
st.markdown("Get movie recommendations based on your user profile.")

user_ids = sorted(ratings['userId'].unique())
selected_user = st.selectbox("Select your user ID:", user_ids)

if st.button("Recommend Movies"):
    titles, posters = recommend_for_user(selected_user, model, movies, ratings)

    st.subheader("Recommended Movies for You:")
    cols = st.columns(5)
    for i, col in enumerate(cols):
        if i < len(posters):
            with col:
                st.image(posters[i], use_container_width=True)
                st.caption(titles[i])
