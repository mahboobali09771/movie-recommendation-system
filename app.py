import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("movies.csv")

# Prepare data
movies['genres'] = movies['genres'].str.replace('|',' ')

cv = CountVectorizer()
matrix = cv.fit_transform(movies['genres'])

similarity = cosine_similarity(matrix)

similarity_df = pd.DataFrame(similarity,
                             index=movies['title'],
                             columns=movies['title'])

# Recommendation function
def recommend(movie_name):

    if movie_name not in similarity_df.columns:
        return ["Movie not found"]

    similar_movies = similarity_df[movie_name].sort_values(ascending=False)[1:6]

    return similar_movies.index.tolist()


# Streamlit UI
st.title("🎬 Movie Recommendation System")

movie = st.text_input("Enter Movie Name")

if st.button("Recommend"):

    result = recommend(movie)

    for m in result:
        st.write(m)