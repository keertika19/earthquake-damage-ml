"""
Nepal earthquake building-damage prediction (damage_grade 1/2/3)
Models: K-Nearest Neighbors, Neural Network (MLP), Random Forest

Usage examples
    python train.py --data-dir data                    # run all three models
    python train.py --data-dir data --model rf         # one model only
    python train.py --data-dir data --sample 30000     # quick smoke test
    python train.py --data-dir data --predict-test     # also write submissions
    python train.py --data-dir data --model rf --rf-class-weight balanced --rf-max-depth 20
"""
import argparse
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
TARGET = "damage_grade"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_data(data_dir: Path, sample: int = 0):
    X = pd.read_csv(data_dir / "train_values.csv", index_col="building_id")
    y = pd.read_csv(data_dir / "train_labels.csv", index_col="building_id")[TARGET]
    y = y.loc[X.index]
    if sample:
        X = X.sample(sample, random_state=SEED)
        y = y.loc[X.index]
    return X, y


def make_preprocessor(X: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """One-hot encode categorical columns; optionally standardise numeric ones."""
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([("cat", ohe, cat_cols), ("num", num, num_cols)])


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_knn(X):
    # KNN is distance-based -> scale features, then keep the 20 best (as in the paper)
    return Pipeline(
        [
            ("prep", make_preprocessor(X, scale=True)),
            ("select", SelectKBest(f_classif, k=20)),
            ("clf", KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ]
    )


def build_nn(X):
    # Two-layer network, L2 regularisation (alpha), mini-batches, softmax output
    return Pipeline(
        [
            ("prep", make_preprocessor(X, scale=True)),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-4,
                    batch_size=1000,
                    learning_rate_init=1e-3,
                    max_iter=200,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=SEED,
                ),
            ),
        ]
    )


def build_rf(X, class_weight=None, max_depth=None):
    # Trees need no scaling. Small grid search like the paper, with 3-fold CV.
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(X, scale=False)),
            (
                "clf",
                RandomForestClassifier(
                    random_state=SEED,
                    n_jobs=-1,
                    class_weight=class_weight,
                    max_depth=max_depth,
                ),
            ),
        ]
    )
    grid = {
        "clf__n_estimators": [100, 200],
        "clf__min_samples_leaf": [1, 5],
    }
    return GridSearchCV(pipe, grid, cv=3, scoring="f1_micro", n_jobs=1, verbose=1)


BUILDERS = {"knn": build_knn, "nn": build_nn, "rf": build_rf}
NAMES = {"knn": "K-Nearest Neighbors", "nn": "Neural Network (MLP)", "rf": "Random Forest"}


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------
def evaluate(key, model, X_tr, y_tr, X_val, y_val, out: Path):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    t0 = time.time()
    pred_val = model.predict(X_val)
    pred_time = time.time() - t0
    pred_tr = model.predict(X_tr)

    row = {
        "model": NAMES[key],
        "train_micro_f1": f1_score(y_tr, pred_tr, average="micro"),
        "val_micro_f1": f1_score(y_val, pred_val, average="micro"),
        "val_macro_f1": f1_score(y_val, pred_val, average="macro"),
        "fit_seconds": round(fit_time, 1),
        "predict_seconds": round(pred_time, 1),
    }
    if key == "rf":
        row["best_params"] = str(model.best_params_)

    print(f"\n===== {NAMES[key]} =====")
    print(classification_report(y_val, pred_val, digits=4))

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_val, pred_val, normalize="true", ax=ax, cmap="Blues", values_format=".2f"
    )
    ax.set_title(NAMES[key])
    fig.tight_layout()
    fig.savefig(out / f"confusion_{key}.png", dpi=150)
    plt.close(fig)

    if key == "rf":
        plot_rf_importance(model.best_estimator_, out)
    return row


def plot_rf_importance(pipe, out: Path, top=15):
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    idx = np.argsort(imp)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i].split("__")[-1] for i in idx], imp[idx])
    ax.set_title("Random Forest - top feature importances")
    fig.tight_layout()
    fig.savefig(out / "rf_feature_importance.png", dpi=150)
    plt.close(fig)


def plot_nn_loss(model, out: Path):
    clf = model.named_steps["clf"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(clf.loss_curve_, label="train loss")
    if getattr(clf, "validation_scores_", None):
        ax2 = ax.twinx()
        ax2.plot(clf.validation_scores_, color="orange", label="val accuracy")
        ax2.set_ylabel("validation accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Neural network training curve")
    fig.tight_layout()
    fig.savefig(out / "nn_training_curve.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--out-dir", default="results", type=Path)
    ap.add_argument("--model", default="all", choices=["all", *BUILDERS])
    ap.add_argument("--models-dir", default="models", type=Path, help="where to save trained models")
    ap.add_argument("--sample", type=int, default=0, help="use a random subset (fast test)")
    ap.add_argument("--rf-class-weight", default=None, choices=[None, "balanced"], help="RF class weighting")
    ap.add_argument("--rf-max-depth", type=int, default=None, help="limit RF tree depth (reduces overfitting)")
    ap.add_argument("--predict-test", action="store_true", help="write DrivenData-style submission CSVs")
    args = ap.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)
    args.models_dir.mkdir(exist_ok=True, parents=True)

    X, y = load_data(args.data_dir, args.sample)
    print(f"Loaded {len(X):,} buildings, {X.shape[1]} features")
    print("Class distribution:\n", y.value_counts(normalize=True).sort_index().round(3))

    # The competition test set has no labels, so we hold out our own validation set.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    keys = list(BUILDERS) if args.model == "all" else [args.model]
    rows = []
    for key in keys:
        if key == "rf":
            model = build_rf(X_tr, args.rf_class_weight, args.rf_max_depth)
        else:
            model = BUILDERS[key](X_tr)
        rows.append(evaluate(key, model, X_tr, y_tr, X_val, y_val, args.out_dir))
        if key == "nn":
            plot_nn_loss(model, args.out_dir)
        joblib.dump(model, args.models_dir / f"{key}.joblib", compress=3)

        if args.predict_test:
            test = pd.read_csv(args.data_dir / "test_values.csv", index_col="building_id")
            sub = pd.DataFrame({TARGET: model.predict(test)}, index=test.index)
            sub.to_csv(args.out_dir / f"submission_{key}.csv")

    results = pd.DataFrame(rows)
    results.to_csv(args.out_dir / "metrics.csv", index=False)
    print("\n", results.drop(columns=["best_params"], errors="ignore").to_string(index=False))


if __name__ == "__main__":
    main()
