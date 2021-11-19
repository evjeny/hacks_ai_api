from typing import Dict

import pandas as pd


def load_data() -> Dict[str, str]:
    df = pd.read_excel("eps.xlsx", sheet_name="codes")

    cat_to_desc = dict()

    for category, category_description in zip(df["category"], df["category_description"]):
        cat_to_desc[str(category)] = str(category_description)
    
    return cat_to_desc 


def get_parent_category(category: str) -> str:
    if "." in category:
        return category.split(".")[0]

    return category

