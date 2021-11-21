import io

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model import KNNModel, RFModel
from utils import load_data, read_table

cat_to_desc = load_data()
cat_list = list(cat_to_desc.keys())

model = KNNModel()
print(model.classify("вилка"))

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

    categories = model.classify(text)

    return {
        "categories": [
            {
                "category": category,
                "category_description": cat_to_desc.get(category, "Нет описания"),
                "probability": proba
            }
            for category, proba in categories
        ]
    }


@app.post("/handle_table")
def handle_table(
    table: UploadFile = File(...), thresh: float = Form(...), sheet_name: str = Form("main"),
    name_column: str = Form("name"), category_column: str = Form("category")
):
    df = read_table(table.file.read(), sheet_name)
    names = df[name_column]
    categories = df[category_column]
    
    for name, category in zip(names, categories):
        pass

    return StreamingResponse(io.BytesIO(table.file.read()), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
