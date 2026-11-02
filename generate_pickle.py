import pandas as pd
import numpy as np
import ast
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer


# Step 1: Load datasets
movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")


# Step 2: Merge datasets
movies = movies.merge(credits, on="title")


# Step 3: Select required columns
movies = movies[
    ["genres", "id", "title", "keywords",
     "overview", "crew", "cast"]
]


# Step 4: Convert genres and keywords
def convert(obj):
    result = []

    for item in ast.literal_eval(obj):
        result.append(item["name"])

    return result


movies["genres"] = movies["genres"].apply(convert)
movies["keywords"] = movies["keywords"].apply(convert)


# Step 5: Get top 3 actors
def get_cast(obj):
    result = []

    for item in ast.literal_eval(obj)[:3]:
        result.append(item["name"])

    return result


movies["cast"] = movies["cast"].apply(get_cast)


# Step 6: Get director
def get_director(obj):
    result = []

    for item in ast.literal_eval(obj):
        if item["job"] == "Director":
            result.append(item["name"])
            break

    return result


movies["crew"] = movies["crew"].apply(get_director)


# Step 7: Handle missing overview
movies["overview"] = movies["overview"].fillna("")


# Step 8: Convert overview into list
movies["overview"] = movies["overview"].apply(
    lambda x: x.split()
)


# Step 9: Remove spaces
for column in ["genres", "keywords", "cast", "crew"]:
    movies[column] = movies[column].apply(
        lambda x: [i.replace(" ", "") for i in x]
    )


# Step 10: Create tags
movies["tags"] = (
    movies["genres"]
    + movies["keywords"]
    + movies["overview"]
    + movies["crew"]
    + movies["cast"]
)


# Step 11: Create new dataframe
new_df = movies[["id", "title", "tags"]].copy()


# Step 12: Convert tags into text
new_df["tags"] = new_df["tags"].apply(
    lambda x: " ".join(x)
)

new_df["tags"] = new_df["tags"].str.lower()


# Step 13: Apply stemming
ps = PorterStemmer()

def stem(text):
    return " ".join(ps.stem(word) for word in text.split())


new_df["tags"] = new_df["tags"].apply(stem)


# Step 14: Convert movie IDs into lists
# Your app expects id[0]
new_df["id"] = new_df["id"].astype(str).apply(
    lambda x: [x]
)


# Step 15: Convert text into vectors
cv = CountVectorizer(
    max_features=5000,
    stop_words="english"
)

vectors = cv.fit_transform(new_df["tags"]).toarray()


# Step 16: Calculate cosine similarity
similarity = cosine_similarity(vectors)


# Step 17: Reduce file size
similarity = similarity.astype(np.float32)


# Step 18: Generate movie.pkl
with open("movie.pkl", "wb") as file:
    pickle.dump(new_df, file)


# Step 19: Generate similarity.pkl
with open("similarity.pkl", "wb") as file:
    pickle.dump(similarity, file)


print("Both pickle files generated successfully!")

print("Movies:", len(new_df))
print("Similarity shape:", similarity.shape)