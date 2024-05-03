import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from StringPreprocessing import text_processing
from sklearn.pipeline import Pipeline
import pickle

st.header("Emotion Detection")

model = pickle.load(open("model.pkl", "rb"))
text_processing_odj = text_processing(
    lower=True,
    remove_special_characters=True,
    remove_punctuation=True,
    remove_stop_words=True,
    stem_the_words=False,
)
process_pip = Pipeline(
    [
        ("text_processing", text_processing_odj),
    ]
)
text = st.text_input("Enter your text here")

if text is not None:
    st.write("### You entered")
    st.write("##### ", text)

    data = {"text": [text]}

    df = pd.DataFrame(data)
    process_pip.fit(df["text"])
    df["text"] = process_pip.transform(df["text"])
    model.predict(df["text"])
    st.write("### Emotion Detected :")

    def update(state):
        if state == 0:
            return "Angry"
        elif state == 1:
            return "Fear"
        elif state == 2:
            return "Joy"
        elif state == 3:
            return "sadness"
        else:
            return "Neutral"

    state = update(model.predict(df["text"]))
    st.write("##### ", state)
