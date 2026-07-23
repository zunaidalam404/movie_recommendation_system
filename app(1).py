import streamlit as st
import pickle
import pandas as pd
import requests
from nltk import data


def fetch_poster(movie_id):
    response = requests.get(f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US")
    data = response.json()

    if "poster_path" in data and data["poster_path"]:
        return "https://image.tmdb.org/t/p/w500/" + data["poster_path"]
    else:
        return "https://via.placeholder.com/500x750?text=No+Image"

# Load Data
movies = pickle.load(open("movie.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    recommended_movies_posters=[]
    for i in movie_list:
        movie_id = movies.iloc[i[0]]["id"][0]
        recommended_movies.append(movies.iloc[i[0]]["title"])
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommended_movies, recommended_movies_posters
print(movies.head())

# Dropdown
selected_movie = st.selectbox(
    "Select a Movie",
    movies["title"].values
)

# Button
if st.button("Recommend"):
    recommendations, posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.image(posters[0])
        st.write(recommendations[0])

    with col2:
        st.image(posters[1])
        st.write(recommendations[1])

    with col3:
        st.image(posters[2])
        st.write(recommendations[2])

    with col4:
        st.image(posters[3])
        st.write(recommendations[3])

    with col5:
        st.image(posters[4])
        st.write(recommendations[4])


