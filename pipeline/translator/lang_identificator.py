import pandas as pd
import os
import fasttext
import sys


FASTTEXT_MODEL_PATH = '/gpfs/projects/bsc88/NextProcurement/Pipeline/pipeline/pdf2txt/translator_2/fast_text_model_binary/lid.176.bin'



# Load the FastText language detection model
model_language = fasttext.load_model(FASTTEXT_MODEL_PATH)


# Function to get the language of a given text using fasttext
def get_language(text):
    global model_language
    text_without_newlines = text.replace("\n", " ")
    language = model_language.predict(text_without_newlines, k=1)[0][0][-2:]
    return language

