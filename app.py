import io

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model import KNNModel, RFModel, MLPModel
from utils import load_data, read_table

cat_to_desc = load_data()
cat_list = list(cat_to_desc.keys())

knn = KNNModel()
rf = RFModel()
mlp = MLPModel()

model_mapper = {
    "knn": knn,
    "rf": rf,
    "mlp": mlp
}

for name, model in model_mapper.items():
    print(name, ":", model.classify("вилка"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "Hello world"


@app.get("/classify/{text}")
def classify(text: str, model_type: str = "knn"):
    global model_mapper, knn, cat_to_desc

    if model_type in model_mapper:
        model_type = "knn"

    categories = model_mapper[model_type].classify(text)

    return {
        "categories": [
            {
                "category": category,
                "category_description": cat_to_desc.get(category, "Нет описания"),
                "probability": proba
            }
            for category, proba in categories
        ],
        "model_type": model_type
    }
