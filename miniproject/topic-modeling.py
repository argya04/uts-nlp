# inisialisasi streamlit
import streamlit as st
import streamlit.components.v1 as components
import traceback
import pickle

# untuk memanipulasi data pada DataFrame
import pandas as pd

# untuk operasi aritmatika
import numpy as np

# untuk visualisasi data
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud

#untuk cleaning data
import string
import re

# untuk stopword dan stemming
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# untuk translate data ke bahasa tertentu
from deep_translator import GoogleTranslator
import time

# untuk memberikan sentimen pada data berbahasa inggris
from textblob import TextBlob

from sklearn.model_selection import train_test_split        # untuk membagi data menjadi data latih (training) dan data uji (testing)
from sklearn.feature_extraction.text import TfidfVectorizer # untuk melakukan ekstraksi fitur tiap kata 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pyLDAvis
from pyLDAvis import prepare

# begin of title
st.set_page_config(page_title='Topic Modeling', layout='wide')
st.markdown("# Topic Modeling")
st.write("Nama : Argya Falan Rifqi, Faizal Fauzi")
st.write("NIM  : 2212500686, 2212500900")
st.write("---")
# end of title

# Begin of text preprocessing code=
def preprocessing(text):
    # Begin of Casefolding function
    text = text.lower()
    # End of Casefolding function

    # Begin of Remove Symbol function
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # Hapus URL
    text = re.sub(r"@\w+", ' ', text)                    # Hapus mention
    text = re.sub(r"#\w+", ' ', text)                    # Hapus hashtag
    text = re.sub(r"\d+", ' ', text)                     # Hapus angka
    text = re.sub(r"[^\w\s]", ' ', text)                 # Hapus tanda baca
    text = re.sub(r"\s+", ' ', text).strip()             # Hapus spasi berlebih
    text = re.sub(r"[^\x00-\x7F]+", ' ', text)           # Hapus emoji & non-ASCII
    # End of Remove Symbol function

    # Begin of Normalize slangword
    data = pd.read_excel('kamus_slangword.xlsx')
    slang_dict = dict(zip(data['tidak_baku'], data['baku']))
    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    text = ' '.join(words)
    # End of Normalize slangword

    # Begin of Tokenize function
    tokens = re.findall(r'\b\w+\b', text)
    # End of Tokenize function

    # Begin of Stopword function
    factory = StopWordRemoverFactory()
    stop_words = set(factory.get_stop_words())
    tokens = [word for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    # End of Stopword function

    # Begin of Stemming function
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)
    return text
    # End of Stemming function
    
# End of text preprocessing code

# # ==========================
# # 📤 Upload file
# # ==========================
uploaded_file = st.file_uploader("Upload dataset anda", type=["csv", "xlsx"])

# Initialize session untuk setiap proses
if "df" not in st.session_state:
    st.session_state.df = None

if "text_column" not in st.session_state:
    st.session_state.text_column = None
    
if "preprocessed" not in st.session_state:
    st.session_state.preprocessed = False

if "labeled" not in st.session_state:
    st.session_state.labeled = False
    
if "lda_session" not in st.session_state:
        st.session_state.lda_session = False

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview Dataset")
    st.dataframe(df)

    # Pilih kolom teks
    # if not st.session_state.preprocessed:
    #     text_column = st.selectbox("Pilih kolom teks:", df.columns)
    # else:
    #     text_column = st.session_state.text_column

    # Jumlah topik untuk topik modeling
    # n_topics = st.slider("Jumlah topik (n_components)", 5, 10, 15, 20, 25, 50)

    # Tombol preprocessing
    if not st.session_state.preprocessed:
        text_column = st.selectbox("Pilih kolom teks:", df.columns, key="textcol")
        if st.button("🔍 Lakukan Preprocessing"):
            with st.spinner("Sedang melakukan preprocessing..."):
                df["tweet_clean"] = df[text_column].astype(str).apply(preprocessing)

            st.session_state.df = df
            st.session_state.text_column = text_column
            st.session_state.preprocessed = True
            st.success("Preprocessing selesai!")

    if st.session_state.preprocessed:
        df = st.session_state.df
        text_column = st.session_state.text_column
        st.subheader("📌 Hasil Preprocessing")
        st.dataframe(df[[text_column, "tweet_clean"]])
    # End of Preprocessing code

        # Begin of Machine Learning Process (LDA)
        st.subheader("🧠 Implementasi Machine Learning")
        if not st.session_state.lda_session:
            if st.button("🧩 Run LDA Topic Modeling"):
                with st.spinner("Work on progress..."):
                    df = st.session_state.df
                    x = df['tweet_clean']

                    # Begin of CountVectorizer
                    vectorizer = CountVectorizer(min_df=5)
                    X_count = vectorizer.fit_transform(x)
                    # End of CountVectorizer
                    
                    # Begin of LDA Topic Modeling
                    lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
                    lda_tf.fit(X_count)
                    # End of LDA Topic Modeling

                    # Simpan ke session_state
                    st.session_state.lda_session = True
                    st.session_state.X_counts = X_count
                    st.session_state.vectorizer = vectorizer
                    st.session_state.lda_tf = lda_tf
                    st.session_state.df_result = df

                st.success("Proses LDA selesai!")

        # Begin of implementation visualization
        if st.session_state.lda_session:

            df = st.session_state.df_result
            X = st.session_state.X_counts
            vectorizer = st.session_state.vectorizer
            lda_tf = st.session_state.lda_tf

            # Begin of LDA topic visualization with Wordcloud
            st.subheader("WordCloud Topik LDA")

            terms = vectorizer.get_feature_names_out()
            components = lda_tf.components_

            for idx, topic in enumerate(components):
                st.write(f"### Topik {idx+1}")

                top_words = {terms[i]: topic[i] for i in topic.argsort()[:-30:-1]}
                wc_topic = WordCloud(width=600, height=400, background_color="white").generate_from_frequencies(top_words)

                fig_tw, ax_tw = plt.subplots(figsize=(6, 4))
                ax_tw.imshow(wc_topic)
                ax_tw.axis("off")
                st.pyplot(fig_tw)
            # End of LDA topic visualization with Wordcloud

            # Begin of LDA topic visualization with pyLDAvis
            st.subheader("📊 Visualisasi LDA dengan pyLDAvis")
            # Patch untuk sklearn (karena modul sudah dihapus)
            def sklearn_lda_to_pyldavis(lda_model, X, vectorizer):
                vocab = vectorizer.get_feature_names_out()

                # topic-term distributions
                topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, None]

                # doc-topic distributions
                doc_topic_dists = lda_model.transform(X)

                # document lengths
                doc_lengths = X.sum(axis=1).A1

                # term frequencies
                term_frequency = np.asarray(X.sum(axis=0)).ravel()

                return prepare(
                    topic_term_dists=topic_term_dists,
                    doc_topic_dists=doc_topic_dists,
                    doc_lengths=doc_lengths,
                    vocab=vocab,
                    term_frequency=term_frequency
                )

            # fungsi visualisasi
            panel = sklearn_lda_to_pyldavis(lda_tf, X_count, vectorizer)
            html_vis = pyLDAvis.prepared_data_to_html(panel)
            st.components.v1.html(html_vis, height=800, scrolling=True)

