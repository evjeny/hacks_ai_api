from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from model import Model
from utils import load_data

cat_to_desc = load_data()
cat_list = list(cat_to_desc.keys())

model = Model()
print(model.classify("ложка столовая"))

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
def classify(text: str):
    global model, cat_to_desc

    category, proba = model.classify(text)
    category_description = cat_to_desc.get(category, "Нет описания")

    return {
        "category": category,
        "category_description": category_description,
        "probability": proba
    }
