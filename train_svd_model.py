
import pandas as pd
import pickle
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split

# Load ratings data
ratings = pd.read_csv("ratings.csv")

# Define rating scale and load data into surprise format
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Initialize and train the SVD model
model = SVD(n_factors=100, lr_all=0.005, reg_all=0.02, n_epochs=20)
model.fit(trainset)

# Save the trained model
with open("svd_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ SVD model trained and saved as svd_model.pkl")
