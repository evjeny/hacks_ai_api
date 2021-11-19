from fastapi import FastAPI

from model import Model
from utils import load_data, get_parent_category

cat_to_desc = load_data()
cat_list = list(cat_to_desc.keys())

model = Model(cat_list)

app = FastAPI()


@app.get("/")
def read_root():
    return "Hello world"


@app.get("/classify/{text}")
def classify(text: str):
    global model, cat_to_desc

    category, proba = model.classify(text)
    category_description = cat_to_desc.get(category, "Нет описания")

    parent_category = get_parent_category(category)
    parent_category_description = cat_to_desc.get(parent_category, "Нет описания")

    return {
        "category": category,
        "category_description": category_description,
        "probability": proba,
        "parent_category": parent_category,
        "parent_category_description": parent_category_description
    }
