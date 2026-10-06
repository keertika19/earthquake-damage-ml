# Earthquake Building Damage Prediction (Nepal 2015)

UE24CS352A – Machine Learning mini-project.
Predicts the damage level of a building (1 = low, 2 = medium, 3 = near-complete destruction)
from structural and legal features using three models: **K-Nearest Neighbors, Neural Network (MLP), Random Forest**.

**Team:** <Name 1 (SRN)>, <Name 2 (SRN)>

## Dataset
"Richter's Predictor: Modeling Earthquake Damage" (DrivenData), from the 2015 Gorkha earthquake open data portal.
Download `train_values.csv`, `train_labels.csv`, `test_values.csv` and place them in a `data/` folder.
(The datasets are not committed to the repo.)

## Setup
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python train.py --data-dir data                 # trains + evaluates KNN, NN, RF
python train.py --data-dir data --model rf      # a single model (knn | nn | rf)
python train.py --data-dir data --sample 30000  # fast smoke test
python train.py --data-dir data --predict-test  # also writes submission CSVs
python demo.py --model rf --n 5                 # live demo on held-out validation buildings
python train.py --data-dir data --model rf --rf-class-weight balanced --rf-max-depth 20   # RF variant
```
Optional: `--models-dir <path>` saves/loads models elsewhere (e.g. `/content/models` on Colab).

### Running on Google Colab
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/earthquake-damage-ml
!python train.py --data-dir data
!python demo.py --model rf --n 5
```
Outputs: `results/metrics.csv`, confusion matrices, RF feature importances, NN training curve; trained models in `models/`.

## Method summary
| Model | Preprocessing | Key settings |
|---|---|---|
| KNN | one-hot + standardise, SelectKBest (k=20) | k = 7 |
| Neural network | one-hot + standardise | 1 hidden layer (64), L2 reg, mini-batch 1000, early stopping |
| Random Forest | one-hot | GridSearchCV over n_estimators, min_samples_leaf (3-fold CV) |

Evaluation: stratified 80/20 train/validation split (seed 42), metric = micro-averaged F1 (plus macro F1 and confusion matrices).

## Repo structure
```
train.py          training + evaluation of all models
demo.py           live-demo script
requirements.txt
results/          metrics and plots (generated)
models/           saved models (generated, git-ignored)
data/             CSVs (git-ignored)
```
