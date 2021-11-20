from typing import Dict

import pandas as pd
import numpy as np
from navec import Navec
import spacy

nlp = spacy.load("ru_core_news_md")
navec = Navec.load("navec_hudlit_v1_12B_500K_300d_100q.tar")
print("load text models")


def load_data() -> Dict[str, str]:
    df = pd.read_excel("eps.xlsx", sheet_name="codes")

    cat_to_desc = dict()

    for category, category_description in zip(df["category"], df["category_description"]):
        cat_to_desc[str(category)] = str(category_description)
    
    return cat_to_desc 


def text_to_vectors(text: str) -> np.ndarray:
    global nlp, navec

    doc = nlp(text)
    vectors = [navec.get(token.lemma_) for token in doc]
    vectors = [v for v in vectors if v is not None]

    if len(vectors) == 0:
        vectors = [np.zeros_like(navec["дом"])]

    return np.array(vectors)


def text_to_vector(text: str) -> np.ndarray:
    return text_to_vectors(text).max(axis=0)
