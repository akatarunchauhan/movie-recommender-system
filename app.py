import streamlit as st
import pandas as pd
import pickle
import requests

# Load data
movies = pd.DataFrame(pickle.load(open("movies_list.pkl", "rb")))
similarity = pickle.load(open("similarity.pkl", "rb"))



@st.cache_data
def fetch_posters(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=9ffc3d5dc1fd7040afe47352ffa2d405&language=en-US"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"

    except (requests.exceptions.RequestException, Exception) as e:
        # This catches timeouts, connection errors, etc.
        st.warning(f"Could not fetch poster for ID {movie_id}. Using placeholder.")
        return "https://via.placeholder.com/500x750?text=Connection+Error"


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in movies_list[1:6]:
        # Get movie_id and title
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_names.append(movies.iloc[i[0]].title)
        # Fetch poster
        recommended_movie_posters.append(fetch_posters(movie_id))

    return recommended_movie_names, recommended_movie_posters


st.title("Movies Recommender System")
selected_movie_name = st.selectbox("Enter a movie name...", movies['title'].values)

if st.button("Show Recommendations"):
    with st.spinner('Fetching recommendations...'):
        names, posters = recommend(selected_movie_name)

        cols = st.columns(5)
        for i in range(5):
            with cols[i]:
                st.text(names[i])
                st.image(posters[i])