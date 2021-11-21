from typing import List, Tuple

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
import joblib

import utils


class Model:
    def classify(self, text: str, thresh: float = 0.1) -> List[Tuple[str, float]]:
        pass


class RFModel(Model):
    def __init__(self) -> None:
        self.onehot: OneHotEncoder = joblib.load("onehot.joblib")
        self.forest: RandomForestClassifier = joblib.load("rf_multiclass.joblib")
    
    def classify(self, text: str, thresh: float = 0.1) -> List[Tuple[str, float]]:
        vector = utils.text_to_vector(text).reshape(1, -1)

        class_probas = self.forest.predict_proba(vector)
        multiprobas = utils.class_probas_to_multiprobas(class_probas)
        multihot = multiprobas.copy()
        multihot[multihot >= thresh] = 1
        multihot[multihot < thresh] = 0

        pred_labels = utils.inverse_multihot(self.onehot, multihot)
        pred_probas = [multiprobas[i] for i in range(len(multihot)) if multihot[i] == 1]
        
        return [(label, proba) for label, proba in zip(pred_labels, pred_probas)]


class KNNModel(Model):
    def __init__(self) -> None:
        self.onehot: OneHotEncoder = joblib.load("onehot.joblib")
        self.knc: KNeighborsClassifier = joblib.load("knc.joblib")
    
    def classify(self, text: str, thresh: float = 0.1) -> List[Tuple[str, float]]:
        vector = utils.text_to_vector(text).reshape(1, -1)

        class_probas = self.knc.predict_proba(vector)
        multiprobas = utils.class_probas_to_multiprobas(class_probas)
        multihot = multiprobas.copy()
        multihot[multihot >= thresh] = 1
        multihot[multihot < thresh] = 0
        
        pred_labels = utils.inverse_multihot(self.onehot, multihot)
        pred_probas = [multiprobas[i] for i in range(len(multihot)) if multihot[i] == 1]
        
        return [(label, proba) for label, proba in zip(pred_labels, pred_probas)]
