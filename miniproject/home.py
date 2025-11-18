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
st.set_page_config(page_title='Home', layout='wide')
st.markdown("# Mini Project UTS NLP")
st.write("Nama : Argya Falan Rifqi, Faizal Fauzi")
st.write("NIM  : 2212500686, 2212500900")
st.write("---")
# end of title

# # =================================
# # Begin of text preprocessing code
# # =================================
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
# # =================================
# # End of text preprocessing code
# # =================================

# # =================================
# # Begin of dataset labeling code
# # =================================
def translate_inggris(kalimat):
     try:
          time.sleep(0.1)  # jeda untuk mencegah rate limit
          terjemahan = GoogleTranslator(source='id', target='en').translate(kalimat)
          return terjemahan.lower()
     except Exception as e:
        # Tampilkan kalimat yang gagal dan lanjutkan
        print(f"[Gagal menerjemahkan kalimat]: {kalimat}\n[error]: {e}")
        return kalimat
    
def pelabelan_sentimen(kalimat_inggris):
    blob = TextBlob(kalimat_inggris)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        sentimen = "positive"
    elif polarity == 0:
        sentimen = "neutral"
    else: 
        sentimen = "negative"
    return pd.Series([polarity, sentimen])

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

        # Begin of Labeling code
        if not st.session_state.labeled:
            if st.button("🏷️ Jalankan Labeling"):
                with st.spinner("Sedang melakukan labeling..."):
                    df["tweet_translated"] = df["tweet_clean"].apply(translate_inggris)
                    df[["polarity", "sentimen"]] = df["tweet_translated"].apply(pelabelan_sentimen)

                st.session_state.df = df
                st.session_state.labeled = True
                st.success("Proses labeling selesai!")       
    
    if st.session_state.labeled:
        df = st.session_state.df
        text_column = st.session_state.text_column
        st.subheader("📌 Hasil Labeling")
        st.dataframe(df[[text_column, "tweet_clean", "tweet_translated", "polarity", "sentimen"]])
    # End of Labeling code

        # Begin of Labeling Visualization code
        st.subheader("📊 Visualisasi Hasil Labeling")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Bar Chart")
            bar_width = st.slider("Lebar Bar Chart", 2, 15, 10) # angka paling belakang default size nya
            bar_height = st.slider("Tinggi Bar Chart", 2, 15, 2) # angka paling belakang default size nya

        with col2:
            st.markdown("### Pie Chart")
            pie_width = st.slider("Lebar Pie Chart", 2, 15, 10) # angka paling belakang default size nya
            pie_height = st.slider("Tinggi Pie Chart", 2, 15, 3)# angka paling belakang default size nya
            
        hitung_sentimen = df['sentimen'].value_counts()

        # BAR CHART
        fig1, ax1 = plt.subplots(figsize=(bar_width, bar_height))
        ax1.bar(hitung_sentimen.index, hitung_sentimen.values)
        ax1.set_title("Distribusi Sentimen")
        ax1.set_xlabel("Kategori Sentimen")
        ax1.set_ylabel("Jumlah")
        for i, v in enumerate(hitung_sentimen.values):
            ax1.text(i, v + 0.2, str(v), ha='center')
        st.pyplot(fig1)

        # PIE CHART
        fig2, ax2 = plt.subplots(figsize=(pie_width, pie_height))
        ax2.pie(
            hitung_sentimen,
            labels=hitung_sentimen.index,
            autopct='%1.1f%%',
            startangle=140
        )
        ax2.set_title("Persentase Sentimen")
        st.pyplot(fig2)
        # End of Visualize labeling result
        
        # Begin of Machine Learning Process (LDA)
        st.subheader("🧠 Implementasi Machine Learning")
        if not st.session_state.lda_session:
            if st.button("🧩 Run LDA Algorithm"):
                with st.spinner("Work on progress..."):
                    df = st.session_state.df
                    x = df['tweet_clean']
                    y = df['sentimen']

                    # Begin of CountVectorizer
                    vectorizer = CountVectorizer(min_df=5)
                    X_count = vectorizer.fit_transform(x)
                    # End of CountVectorizer
                    
                    # Begin of TfidfVectorizer
                    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1))
                    tfidf_vectorizer.fit_transform(x)
                    words = tfidf_vectorizer.get_feature_names_out()
                    pickle.dump(tfidf_vectorizer.vocabulary_,open("feature_tfidf.sav", "wb"))
                    x1 = tfidf_vectorizer.fit_transform(x).toarray()
                    data_tfidf = pd.DataFrame(x1,columns=words)
                    
                    x_train = np.array(data_tfidf)
                    y_train = np.array(y)
                    chi2_features = SelectKBest(chi2, k=2100)
                    x_kbest_features = chi2_features.fit_transform(x_train, y_train)
                    Data = pd.DataFrame({
                        "nilai": chi2_features.scores_,
                        "fitur": words
                    })
                    mask = chi2_features.get_support() # mask adalah nilai tertinggi dari hasil chi square
                    # menampilkan fitur yang terpilih berdasarkan nilai mask atau nilai tertinggi dari hasil chi square
                    new_feature = []
                    for bool, f in zip(mask, words):
                        if bool:
                            new_feature.append(f)
                        selected_feature = new_feature

                    new_selected_feature = {}
                    for (k,v) in tfidf_vectorizer.vocabulary_.items():
                        if k in selected_feature:
                            new_selected_feature[k]=v
                    pickle.dump(new_selected_feature, open("new_selected_feature_tfidf.sav", "wb"))
                    data_selected_feature = pd.DataFrame(x_kbest_features, columns = selected_feature)
                    # End of TfidfVectorizer
                    
                    # Begin of split data to 80% training set, 20% testing set.
                    selected_x = x_kbest_features
                    x = selected_x
                    y = df['sentimen']
                    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                    # End of split data.
                    
                    # Begin of LDA Classification
                    lda_classify = LinearDiscriminantAnalysis()
                    lda_classify.fit(X_train, y_train)
                    y_pred = lda_classify.predict(X_test.toarray())
                    # End of LDA Classification

                    # Begin of LDA Topic Modeling
                    lda_tf = LatentDirichletAllocation(n_components=20, random_state=0)
                    lda_tf.fit(X_count)
                    # End of LDA Topic Modeling

                    # Simpan ke session_state
                    st.session_state.lda_session = True
                    st.session_state.X_counts = X_count
                    st.session_state.vectorizer = vectorizer
                    st.session_state.lda_tf = lda_tf
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    st.session_state.df_result = df
                    st.session_state.X_train = X_train

                st.success("Proses LDA selesai!")

        # Begin of implementation visualization
        if st.session_state.lda_session:

            df = st.session_state.df_result
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            X = st.session_state.X_counts
            vectorizer = st.session_state.vectorizer
            lda_tf = st.session_state.lda_tf

            st.subheader("Evaluasi Model LDA")

            # Begin of Classification Report
            st.text(classification_report(y_test, y_pred))
            # End of Classification Report

            # Begin of Confusion Matrix
            st.subheader("Confusion Matrix")
            # End of Confusion Matrix

            labels = ['positive', 'neutral', 'negative']
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap="Reds", xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix - LDA")
            st.pyplot(fig_cm)

            # Begin of Sentiment Visualization with Wordcloud
            st.subheader("☁️ WordCloud Berdasarkan Sentimen")

            sentiments = ['positive', 'neutral', 'negative']

            for s in sentiments:
                text = " ".join(df[df['sentimen'] == s]['tweet_clean'])
                wc = WordCloud(width=600, height=400, background_color="white").generate(text)

                st.write(f"### WordCloud – {s.capitalize()}")
                fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
                ax_wc.imshow(wc)
                ax_wc.axis("off")
                st.pyplot(fig_wc)
            # End of Sentiment Visualization with Wordcloud

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
