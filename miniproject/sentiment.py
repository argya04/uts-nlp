# inisialisasi streamlit
import streamlit as st
import streamlit.components.v1 as components
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer # untuk melakukan ekstraksi fitur tiap kata 


# begin of title
st.set_page_config(page_title='Sentimen', layout='wide')
st.markdown("# Sentiment Prediction")
st.write("Nama : Argya Falan Rifqi, Faizal Fauzi")
st.write("NIM  : 2212500686, 2212500900")
st.write("---")
# end of title

# Load model lda klasifikasi
model_lda = pickle.load(open('lda_klasifikasi.sav', 'rb'))
tfidf_vector = TfidfVectorizer()
load_vectorizer = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tfidf.sav", "rb"))))

# Begin of sentiment prediction
clean_text = st.text_input("Enter your text here")
sentiment_detection = ''

if st.button('🔍 Predict Sentiment'):
    with st.spinner(" Working on it..."):
        sentimen_prediction = model_lda.predict(load_vectorizer.fit_transform([clean_text]))
        if(sentimen_prediction=='positive'):
            sentiment_detection = "Sentimen Positif"
        elif(sentimen_prediction=='neutral'):
            sentiment_detection = "Sentimen Netral"
        else:
            sentiment_detection = "Sentimen Negatif"
        st.success(sentiment_detection)
# End of sentiment prediction
