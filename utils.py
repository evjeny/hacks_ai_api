from typing import Dict, List
import io

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


def class_probas_to_multiprobas(probas) -> np.ndarray:
    result = np.zeros(len(probas))
    for i, proba in enumerate(probas):
        result[i] = proba[0, 1]
    return result


def onehot(size: int, index: int) -> np.ndarray:
    result = np.zeros(size)
    result[index] = 1
    return result


def inverse_multihot(encoder, multihot: np.ndarray) -> List[str]:
    size = len(multihot)
    result = []
    for i in range(len(multihot)):
        if multihot[i] == 1:
            result.append(
                encoder.inverse_transform(onehot(size, i).reshape(1, -1))[0, 0]
            )
    return result


def read_table(file: bytes, sheet_name: str):
    return pd.read_excel(io.BytesIO(file), sheet_name=sheet_name)
