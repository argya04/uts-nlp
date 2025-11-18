# inisialisasi streamlit
import streamlit as st
import streamlit.components.v1 as components

# begin of navigation menu
home_page = st.Page("home.py", title="Home", icon=":material/home:", default = True)
sentiment_page = st.Page("sentiment.py", title="Analisis Sentimen", icon=":material/add_circle:")
topicmodeling_page = st.Page("topic-modeling.py", title="Topic Modeling", icon=":material/delete:")

pg = st.navigation([home_page, sentiment_page, topicmodeling_page])
pg.run()
