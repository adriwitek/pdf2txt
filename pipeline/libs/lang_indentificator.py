import pandas as pd
import os
import fasttext
import sys


FASTTEXT_MODEL_PATH = '/app/pdf2txt/pipeline/models/language_identification/lid.176.bin'



# Load the FastText language detection model
model_language = fasttext.load_model(FASTTEXT_MODEL_PATH)


# Function to get the language of a given text using fasttext
def get_language(text):
    global model_language
    text_without_newlines = text.replace("\n", " ")
    language = model_language.predict(text_without_newlines, k=1)[0][0][-2:]
    return language

