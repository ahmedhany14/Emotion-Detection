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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, Normalizer

# models
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
# xgboost
from xgboost import XGBClassifier

# multiclass
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
import streamlit as st
# remove duplicates
def remove_duplicates(df):
    df = df.drop_duplicates()
    return df
def process_X_data(X):

    # 1. remoce duplicates
    X = remove_duplicates(X)
    
    # 2. remove love and surprise from the data
    X = X[X.emotion != "love"]
    X = X[X.emotion != "surprise"]

    # 3. label encode the emotions
    le = LabelEncoder()
    X["emotion"] = le.fit_transform(X["emotion"])
    return  X
    