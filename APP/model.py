# For this project, we will be using the following libraries
import pandas as pd

# For data visualization, we will be using the following libraries
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# algebra
import numpy as np

# text processing

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# ML libraries

## sklearn

# preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer,
)

# models
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin

# xgboost
from xgboost import XGBClassifier

# multiclass
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

import warnings
import streamlit as st

warnings.filterwarnings("ignore")
import string
import process as processing
from StringPreprocessing import text_processing

data = pd.read_csv(
    r"/home/ahmed/Ai/Data science and Ml projects/Emotion-Detection/Data set/concat_data.csv"
)


data = processing.process_X_data(data)


def MNB_one_vs_all(X, y):
    text_processing_odj = text_processing(
        lower=True,
        remove_special_characters=True,
        remove_punctuation=False,
        remove_stop_words=False,
        stem_the_words=False,
    )
    vectorizer = CountVectorizer(max_features=3000)
    MNB = MultinomialNB()
    MNB_pipeline = Pipeline(
        [
            ("text_processing", text_processing_odj),
            ("vectorizer", vectorizer),
            ("MNB", MNB),
        ]
    )
    oneVall = OneVsRestClassifier(MNB_pipeline)
    oneVall.fit(X, y)
    return oneVall


def RFc(X, y):
    text_processing_odj = text_processing(
        lower=True,
        remove_special_characters=True,
        remove_punctuation=True,
        remove_stop_words=True,
        stem_the_words=False,
    )
    RFC = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    tdidf = CountVectorizer(max_features=3000)
    RFC_pipeline = Pipeline(
        [
            ("text_processing", text_processing_odj),
            ("vectorizer", tdidf),
            ("RFC", RFC),
        ]
    )
    RFC_pipeline.fit(X, y)
    return RFC_pipeline


def XGBC(X, y):
    text_processing_odj = text_processing(
        lower=True,
        remove_special_characters=True,
        remove_punctuation=True,
        remove_stop_words=True,
        stem_the_words=False,
    )
    xgb = XGBClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    )
    tdidf = TfidfVectorizer(max_features=3000)
    xgb_pipeline = Pipeline(
        [
            ("text_processing", text_processing_odj),
            ("vectorizer", tdidf),
            ("xgb", xgb),
        ]
    )
    xgb_pipeline.fit(X, y)
    return xgb_pipeline


def SVMC(X, y):
    text_processing_odj = text_processing(
        lower=True,
        remove_special_characters=True,
        remove_punctuation=True,
        remove_stop_words=True,
        stem_the_words=False,
    )
    count = CountVectorizer(max_features=3000)
    svm = SVC(kernel="linear", gamma=1, C=0.5, random_state=42)
    svm_pipeline = Pipeline(
        [
            ("text_processing", text_processing_odj),
            ("vectorizer", count),
            ("svm", svm),
        ]
    )
    svm_pipeline.fit(X=X, y=y)
    return svm_pipeline


def Vot(X, y):
    MNB = MNB_one_vs_all(X, y)
    RFC = RFc(X, y)
    XGB = XGBC(X, y)
    SVM = SVMC(X, y)
    vot = VotingClassifier(
        estimators=[("MNB", MNB), ("RFC", RFC), ("XGB", XGB), ("SVM", SVM)]
    )
    vot.fit(X, y)
    return vot


model = Vot(data["text"], data["emotion"])
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    f.close()