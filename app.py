import pickle
import streamlit as st
from whatdoyouwannasee import loaded_knn, vectorizer, newDF


def recommend_movies():

    query = vectorizer.transform([' '.join(all_keywords)])
    distances, indices = loaded_knn.kneighbors(query)
    top_movies = newDF.iloc[indices[0]][['title', 'vote_average']]
    top_movies = top_movies.sort_values(by='vote_average', ascending=False)

    return top_movies['title']


st.markdown("<h1 style='text-align: center; color: #818380;'>Movie Recommender System</h1>",
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #818380;'>Find a movie from a dataset of 5,000 movies!</h4>",
            unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #818380;'>Web App created by İzem und Damla</h4>",
            unsafe_allow_html=True)


all_keywords = []

user_keywords = st.text_input('Please enter at least 5 keywords:')
if user_keywords:
    all_keywords = user_keywords.split()


if user_keywords:
    st.write('Selected words:', all_keywords)

if st.button('Show Recommendation'):
    if len(all_keywords) > 4:
        st.write("Recommended Movies based on your interests are :")
        top_movies = recommend_movies()
        container = st.container()
        with container:
            st.text(top_movies.iloc[0])
            st.text(top_movies.iloc[1])
            st.text(top_movies.iloc[2])
            st.text(top_movies.iloc[3])
            st.text(top_movies.iloc[4])
