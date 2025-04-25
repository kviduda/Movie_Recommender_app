import streamlit as st
import pandas as pd

# Load precomputed recommendations
df = pd.read_csv("user_recommendations.csv")

# UI
st.title("🎬 Personalized Movie Recommendations")
st.markdown("Select your user ID to get movie recommendations based on your profile.")

# Dropdown to select userId
user_ids = sorted(df["userId"].unique())
selected_user = st.selectbox("Choose your user ID:", user_ids)

# Filter recommendations for the selected user
user_recs = df[df["userId"] == selected_user]

# Show recommendations
st.subheader("Recommended Movies:")
cols = st.columns(5)
for i, (index, row) in enumerate(user_recs.iterrows()):
    with cols[i % 5]:
        st.image(row["poster_url"], use_container_width=True)
        st.caption(row["title"])
