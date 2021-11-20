from typing import List, Tuple

import joblib

import utils


class Model:
    def __init__(self) -> None:
        self.encoder = joblib.load("encoder.joblib")
        self.forest = joblib.load("random_forest.joblib")
    
    def classify(self, text: str) -> Tuple[str, float]:
        vector = utils.text_to_vector(text).reshape(1, -1)
        pred_proba = self.forest.predict_proba(vector)[0]
        pred_class = self.forest.predict(vector)
        pred_label = self.encoder.inverse_transform(pred_class)[0]

        return pred_label, pred_proba.max()
