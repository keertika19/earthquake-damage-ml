"""
Live-demo helper: load a trained model and predict damage for random buildings
from the HELD-OUT validation split (same split as train.py, so the model has
never seen these buildings during training).

    python demo.py --model rf --n 5
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = {1: "Low damage", 2: "Medium damage", 3: "Near-complete destruction"}
SEED = 42  # must match train.py

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="rf", choices=["knn", "nn", "rf"])
ap.add_argument("--data-dir", default="data", type=Path)
ap.add_argument("--models-dir", default="models", type=Path)
ap.add_argument("--n", type=int, default=5)
args = ap.parse_args()

model = joblib.load(args.models_dir / f"{args.model}.joblib")

X = pd.read_csv(args.data_dir / "train_values.csv", index_col="building_id")
y = pd.read_csv(args.data_dir / "train_labels.csv", index_col="building_id")["damage_grade"].loc[X.index]
_, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

sample = X_val.sample(args.n)
pred = model.predict(sample)

cols = ["count_floors_pre_eq", "age", "foundation_type", "roof_type", "ground_floor_type"]
out = sample[cols].copy()
out["predicted"] = [LABELS[p] for p in pred]
out["actual"] = [LABELS[v] for v in y_val.loc[sample.index]]
out["correct"] = out["predicted"] == out["actual"]
print(out.to_string())
print(f"\nCorrect: {out['correct'].sum()}/{args.n}")
