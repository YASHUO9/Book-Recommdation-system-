import pickle 
import numpy as np
import streamlit as st
import sys
from src.Machine_Recommendation.utils.utils import recommend_book
st.title("Book recommdation system using Machine Learning")
model = pickle.load(open('artifacts/model.pkl','rb'))
book_name = pickle.load(open('artifacts/book_name.pkl','rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl','rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl','rb'))






selected_book = st.selectbox( 
                             'Type or Select a book',book_name)
if st.button('Show Recommendation '):
    recommended_books , poster_url = recommend_book(selected_book,book_pivot)
    col1,col2,col3,col4,col5 = st.columns(5)
    
    with col1:
        st.text(recommended_books[1])
        st.image(poster_url[1])
        
    with col2:  
        st.text(recommended_books[2])
        st.image(poster_url[2])
        
    with col3:  
        st.text(recommended_books[3])
        st.image(poster_url[3])
        
    with col4:  
        st.text(recommended_books[4])
        st.image(poster_url[4])
        
    with col5:  
        st.text(recommended_books[5])
        st.image(poster_url[5])
    
    