"""
train.py — Run ONCE to generate model.pkl
==========================================
Usage:
    python train.py

Place this file in the same folder as app.py and Amazon_Reviews.csv
Output: model.pkl (saved in the same directory)
"""

import re
import os
import pickle
import numpy as np
import pandas as pd
from io import StringIO

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# ─────────────────────────────────────────
# CONFIG — change paths if needed
# ─────────────────────────────────────────
CSV_PATH    = os.path.join(os.path.dirname(__file__), "Amazon_Reviews.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# ─────────────────────────────────────────
# STEP 1 — LOAD CSV
# ─────────────────────────────────────────
print("[1/5] Loading dataset from:", CSV_PATH)

try:
    with open(CSV_PATH, "rb") as f:
        raw = f.read().replace(b"\x00", b"")
    text_data = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(text_data), on_bad_lines="skip", engine="python")
    print(f"      Loaded — shape: {df.shape}")
except Exception as e:
    raise RuntimeError(f"Failed to load CSV: {e}")

if df is None or df.empty:
    raise RuntimeError("CSV loaded but is empty. Check the file.")

print("      Columns:", df.columns.tolist())

# ─────────────────────────────────────────
# STEP 2 — BUILD SENTIMENT LABELS
# ─────────────────────────────────────────
print("[2/5] Detecting text and rating columns, building labels …")

def _norm(s):
    return re.sub(r"\s+", " ", str(s).lower().strip())

col_norm = {_norm(c): c for c in df.columns}

text_col = None
for candidate in ["review text", "reviewtext", "review_text", "review body", "review", "text", "comment", "content"]:
    if candidate in col_norm:
        text_col = col_norm[candidate]
        break

rating_col = None
for candidate in ["rating", "overall", "score", "stars", "star rating"]:
    if candidate in col_norm:
        rating_col = col_norm[candidate]
        break

if text_col is None:
    raise ValueError(f"Cannot find text column. Available: {df.columns.tolist()}")
if rating_col is None:
    raise ValueError(f"Cannot find rating column. Available: {df.columns.tolist()}")

print(f"      Text column   : '{text_col}'")
print(f"      Rating column : '{rating_col}'")

raw_ratings = df[rating_col].astype(str).str.strip()
print(f"      Rating sample : {raw_ratings.value_counts().head(5).to_dict()}")

# Pattern: "Rated X out of Y stars"
rated = raw_ratings.str.extract(r"Rated\s+(\d+(?:\.\d+)?)\s+out\s+of\s+(\d+(?:\.\d+)?)", expand=True)
num   = pd.to_numeric(rated[0], errors="coerce")
denom = pd.to_numeric(rated[1], errors="coerce")

if num.notna().mean() >= 0.5:
    ratio = num / denom
    df["sentiment"] = (ratio > 0.6).astype(int)
else:
    numeric = pd.to_numeric(raw_ratings, errors="coerce")
    if numeric.notna().mean() >= 0.7:
        rmax = numeric.max()
        if rmax <= 5:
            df["sentiment"] = (numeric >= 4).astype(int)
        elif rmax <= 10:
            df["sentiment"] = (numeric >= 7).astype(int)
        else:
            df["sentiment"] = (numeric >= 60).astype(int)
    else:
        raise ValueError(
            f"Could not parse ratings.\nSample: {raw_ratings.value_counts().head(10).to_dict()}"
        )

df["review_text"] = df[text_col].astype(str).str.strip()
df = (
    df[["review_text", "sentiment"]]
    .dropna()
    .loc[lambda d: d["review_text"].str.len() >= 5]
    .loc[lambda d: d["sentiment"].isin([0, 1])]
    .drop_duplicates(subset=["review_text"])
    .reset_index(drop=True)
)
df["sentiment"] = df["sentiment"].astype(int)

n_pos = int(df["sentiment"].sum())
n_neg = int((df["sentiment"] == 0).sum())
print(f"      Clean rows    : {len(df):,}")
print(f"      Positive      : {n_pos:,}  |  Negative: {n_neg:,}")

if df["sentiment"].nunique() < 2:
    raise ValueError("Only one class present. Check rating column and threshold.")

# ─────────────────────────────────────────
# STEP 3 — PREPROCESS TEXT
# ─────────────────────────────────────────
print("[3/5] Preprocessing text …")

_STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "what","which","who","this","that","these","those","am","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","a","an","the",
    "and","but","or","nor","for","yet","so","at","by","of","in","on","to",
    "up","as","if","than","then","very","just","not","no"
}

def _lemmatize(token):
    for suffix in ("ing", "tion", "ness", "ly", "ed", "er", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[:-len(suffix)]
    return token

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [_lemmatize(t) for t in text.split() if len(t) >= 2 and t not in _STOP_WORDS]
    return " ".join(tokens)

df["clean"] = df["review_text"].apply(preprocess)
print(f"      Example raw   : {df['review_text'].iloc[0][:80]}")
print(f"      Example clean : {df['clean'].iloc[0][:80]}")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─────────────────────────────────────────
# STEP 4 — TRAIN + EVALUATE
# ─────────────────────────────────────────
print("[4/5] Training and evaluating models …")
print("-" * 55)

label_names = ["Negative", "Positive"]

def make_tfidf():
    return TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode"
    )

def evaluate_model(name, model, X_test, y_test):
    """Print full classification metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)

    print(f"\n  [{name}]")
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  F1-Score  : {f1 * 100:.2f}%")
    print("\n  Confusion Matrix:")
    print(f"               Predicted Neg  Predicted Pos")
    print(f"  Actual Neg       {cm[0][0]:<10}     {cm[0][1]}")
    print(f"  Actual Pos       {cm[1][0]:<10}     {cm[1][1]}")
    print("\n  Classification Report:")
    for line in report.split("\n"):
        print("  " + line)

    return {
        "accuracy":  round(acc * 100, 1),
        "precision": round(prec * 100, 1),
        "recall":    round(rec * 100, 1),
        "f1":        round(f1 * 100, 1),
        "confusion_matrix": cm.tolist(),
    }

# Logistic Regression
lr = Pipeline([("tfidf", make_tfidf()), ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42))])
lr.fit(X_train, y_train)
lr_metrics = evaluate_model("Logistic Regression", lr, X_test, y_test)

# Linear SVM
svm = Pipeline([("tfidf", make_tfidf()), ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42))])
svm.fit(X_train, y_train)
svm_metrics = evaluate_model("Linear SVM", svm, X_test, y_test)

# Naive Bayes
nb = Pipeline([("tfidf", make_tfidf()), ("clf", MultinomialNB(alpha=0.1))])
nb.fit(X_train, y_train)
nb_metrics = evaluate_model("Naive Bayes", nb, X_test, y_test)

# ─────────────────────────────────────────
# STEP 5 — SAVE
# ─────────────────────────────────────────
print("\n[5/5] Saving model.pkl …")

best_acc = max(lr_metrics["accuracy"], svm_metrics["accuracy"], nb_metrics["accuracy"])

model_data = {
    "lr":  lr,
    "svm": svm,
    "nb":  nb,
    "metrics": {
        "Logistic Regression": lr_metrics,
        "Linear SVM":          svm_metrics,
        "Naive Bayes":         nb_metrics,
    },
    "stats": {
        "total":    int(len(df)),
        "positive": n_pos,
        "negative": n_neg,
        "best_acc": best_acc,
    },
}

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(model_data, f)

print()
print("=" * 55)
print(f"  ✓ model.pkl saved → {OUTPUT_PATH}")
print(f"  Best accuracy   : {best_acc}%")
print(f"  LR  F1          : {lr_metrics['f1']}%")
print(f"  SVM F1          : {svm_metrics['f1']}%")
print(f"  NB  F1          : {nb_metrics['f1']}%")
print("  Now run:  python app.py")
print("=" * 55)
