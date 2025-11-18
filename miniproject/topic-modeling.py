import streamlit as st
import streamlit.components.v1 as components

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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pyLDAvis
import pyLDAvis.sklearn

# begin of title
st.set_page_config(page_title='Home', layout='wide')
st.markdown("# Topic Modeling")
st.write("Nama : Argya Falan Rifqi, Faizal Fauzi")
st.write("NIM  : 2212500686, 2212500900")
# end of title
