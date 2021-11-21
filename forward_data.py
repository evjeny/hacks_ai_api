import argparse
from typing import List

import pandas as pd
import numpy as np
import tqdm

from model import KNNModel


def get_names(excel_path: str, sheet_name: str, col_name: str) -> List[str]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    names = np.array(df[col_name]).tolist()
    return names


def get_categories(names: List[str]) -> List[List[str]]:
    knn = KNNModel()

    result = []
    for name in tqdm.tqdm(names, desc="forwarding data"):
        result.append(knn.classify(name))
    
    return result


def save_csv(data: List[List[str]], filename: str):
    with open(filename, "w+") as f:
        f.write("cat1,p1,cat2,p2,cat3,p3\n")
        for categories in data:
            categories = sorted(categories, key=lambda t: t[1], reverse=True)
            f.write(",".join(f"{lab},{p}" for lab,p in categories) + "\n")


def main():
    parser = argparse.ArgumentParser("Data forwarder for teamEVA")
    parser.add_argument("--excel", type=str, help="path to excel file")
    parser.add_argument("--sheet_name", type=str, help="name of sheet with data")
    parser.add_argument("--col_name", type=str, help="name of column")
    parser.add_argument("--output_file", type=str, help="path to output csv file")
    args = parser.parse_args()

    names = get_names(args.excel, args.sheet_name, args.col_name)
    print(f"Read {len(names)} entries")
    save_csv(get_categories(names), args.output_file)


if __name__ == "__main__":
    main()
