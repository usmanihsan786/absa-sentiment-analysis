"""
Aspect-Based Sentiment Analysis API — Flask Backend
=====================================================
Endpoints:
  POST /predict          → analyze a single review (overall + aspect-level)
  POST /analyze-csv      → upload CSV, analyze multiple reviews
  GET  /dashboard-data   → aspect-wise statistics from history
  GET  /models           → list of trained models + metrics
  GET  /stats            → dataset statistics
  GET  /history          → recent analysis history
  GET  /health           → health check

Run:
    python app.py
"""

import re
import os
import csv
import json
import pickle
import sqlite3
import io
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template 

# ─────────────────────────────────────────
# INIT
# ─────────────────────────────────────────
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/predict",       methods=["OPTIONS"])
@app.route("/analyze-csv",   methods=["OPTIONS"])
@app.route("/dashboard-data",methods=["OPTIONS"])
@app.route("/models",        methods=["OPTIONS"])
@app.route("/stats",         methods=["OPTIONS"])
def options(*args, **kwargs):
    return "", 204

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
print("[startup] Loading models from", MODEL_PATH)
with open(MODEL_PATH, "rb") as f:
    MODEL_DATA = pickle.load(f)

LR_MODEL  = MODEL_DATA["lr"]
SVM_MODEL = MODEL_DATA["svm"]
NB_MODEL  = MODEL_DATA["nb"]
METRICS   = MODEL_DATA["metrics"]
STATS     = MODEL_DATA["stats"]
print(f"[startup] Models loaded. Dataset stats: {STATS}")

# ─────────────────────────────────────────
# ASPECT CONFIGURATION
# Easily add or remove aspects here.
# Each aspect has a list of trigger keywords.
# ─────────────────────────────────────────
ASPECT_KEYWORDS = {
    "quality": [
        "quality", "build", "material", "solid", "sturdy", "durable", "durability",
        "well made", "craftsmanship", "construction", "premium", "cheap", "flimsy",
        "robust", "fragile", "breaks", "broke", "lasting", "long-lasting"
    ],
    "price": [
        "price", "cost", "value", "expensive", "cheap", "affordable", "overpriced",
        "worth", "money", "budget", "pricey", "reasonable", "bargain", "deal",
        "costly", "economical", "fee", "charge"
    ],
    "delivery": [
        "delivery", "shipping", "shipped", "courier", "arrived", "arrival",
        "dispatch", "dispatched", "transit", "package", "packaging", "delivered",
        "tracking", "delayed", "delay", "late", "fast delivery", "slow delivery",
        "on time", "quick delivery", "damaged in transit"
    ],
    "packaging": [
        "packaging", "package", "box", "wrapped", "wrapping", "packing", "packed",
        "unboxing", "box condition", "damaged box", "well packed", "poorly packed"
    ],
    "customer_service": [
        "customer service", "support", "help", "staff", "agent", "representative",
        "response", "helpful", "rude", "polite", "service", "contact", "resolved",
        "complaint", "refund", "return", "exchange", "communication"
    ],
    "performance": [
        "performance", "speed", "fast", "slow", "efficient", "powerful", "lag",
        "smooth", "responsive", "processing", "runs", "works", "capability",
        "benchmark", "performs", "effective", "functional"
    ],
    "battery": [
        "battery", "battery life", "charge", "charging", "charged", "power",
        "drain", "drains", "lasts", "hours", "standby", "runtime",
        "long battery", "short battery", "quick charge", "fast charge"
    ],
    "camera": [
        "camera", "photo", "photos", "picture", "pictures", "image", "images",
        "video", "recording", "lens", "megapixel", "zoom", "flash", "selfie",
        "night mode", "portrait", "photography"
    ],
    "design": [
        "design", "look", "looks", "appearance", "style", "stylish", "beautiful",
        "ugly", "aesthetic", "color", "colour", "sleek", "elegant", "attractive",
        "modern", "slim", "compact"
    ],
    "size": [
        "size", "small", "large", "big", "tiny", "compact", "portable",
        "lightweight", "heavy", "weight", "dimensions", "fits", "pocket",
        "bulky", "thin", "thick"
    ],
    "usability": [
        "easy", "easy to use", "user friendly", "intuitive", "complicated",
        "difficult", "simple", "straightforward", "interface", "setup",
        "instructions", "manual", "confusing", "clear", "user-friendly",
        "convenient", "handy", "practical"
    ],
}

# Sentiment lexicon — words that lean positive or negative in context
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "fantastic", "wonderful",
    "awesome", "perfect", "best", "love", "loved", "like", "liked",
    "outstanding", "superb", "brilliant", "impressive", "exceptional",
    "recommend", "recommended", "happy", "satisfied", "pleased",
    "fast", "quick", "on time", "reliable", "solid", "sturdy",
    "beautiful", "elegant", "clean", "clear", "easy", "convenient",
    "helpful", "responsive", "smooth", "efficient", "powerful",
    "affordable", "worth", "value", "bargain", "reasonable", "accurate",
    "durable", "long", "lasting", "premium", "high quality", "well made"
}

NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
    "useless", "disappointing", "disappointed", "waste", "broken",
    "defective", "faulty", "slow", "late", "delayed", "rude", "unhelpful",
    "expensive", "overpriced", "cheap", "flimsy", "fragile", "ugly",
    "difficult", "complicated", "confusing", "hard", "frustrating",
    "annoying", "damaged", "missing", "wrong", "incorrect", "false",
    "misleading", "scam", "fraud", "fake", "counterfeit", "never",
    "not working", "doesn't work", "stopped working", "broke", "breaks"
}

NEGATION_WORDS = {"not", "never", "no", "n't", "neither", "nor", "without", "barely", "hardly"}

# ─────────────────────────────────────────
# TEXT PREPROCESSING
# ─────────────────────────────────────────
_STOP_WORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "what","which","who","this","that","these","those","am","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","a","an","the",
    "and","but","or","nor","for","yet","so","at","by","of","in","on","to",
    "up","as","if","than","then","very","just"
}

def _simple_lemmatize(token: str) -> str:
    for suffix in ("ing", "tion", "ness", "ly", "ed", "er", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token

def preprocess(text: str) -> str:
    """Clean text for ML model input."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        _simple_lemmatize(t)
        for t in text.split()
        if len(t) >= 2 and t not in _STOP_WORDS
    ]
    return " ".join(tokens)

# ─────────────────────────────────────────
# ASPECT EXTRACTION
# ─────────────────────────────────────────
def extract_aspects(text: str) -> dict:
    """
    Detect which aspects are mentioned in the review.
    Returns {aspect_name: [list of matched keyword strings]}
    """
    text_lower = text.lower()
    found = {}
    for aspect, keywords in ASPECT_KEYWORDS.items():
        matched = [kw for kw in keywords if kw in text_lower]
        if matched:
            # De-duplicate (remove substring matches if longer match exists)
            deduped = []
            for kw in matched:
                if not any(kw != other and kw in other for other in matched):
                    deduped.append(kw)
            found[aspect] = list(set(deduped))
    return found

# ─────────────────────────────────────────
# CONTEXT WINDOW SENTIMENT SCORING
# ─────────────────────────────────────────
def score_context_window(text: str, keyword: str, window: int = 8) -> float:
    """
    Score sentiment in a ±window word context around a keyword.
    Returns a float: positive > 0, negative < 0.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    positions = [i for i, w in enumerate(words) if keyword.replace(" ", "") in w.replace(" ", "")]
    if not positions:
        # Try multi-word keyword
        text_lower = text.lower()
        if keyword not in text_lower:
            return 0.0
        positions = [0]  # fallback center

    score = 0.0
    for pos in positions:
        start = max(0, pos - window)
        end   = min(len(words), pos + window + 1)
        context = words[start:end]

        # Check for negation
        negated = False
        for i, w in enumerate(context):
            if w in NEGATION_WORDS:
                negated = True

        for w in context:
            if w in POSITIVE_WORDS:
                score += -1.0 if negated else +1.0
            if w in NEGATIVE_WORDS:
                score += +1.0 if negated else -1.0

    return score / max(len(positions), 1)

def classify_aspect_sentiment(score: float, threshold: float = 0.3) -> tuple[str, float]:
    """
    Convert a raw score to a sentiment label + rough confidence.
    Returns (label, confidence)
    """
    if score > threshold:
        label = "Positive"
        confidence = min(0.95, 0.55 + abs(score) * 0.1)
    elif score < -threshold:
        label = "Negative"
        confidence = min(0.95, 0.55 + abs(score) * 0.1)
    else:
        label = "Neutral"
        confidence = 0.55
    return label, round(confidence, 2)

# ─────────────────────────────────────────
# OVERALL SENTIMENT WITH MODEL
# ─────────────────────────────────────────
def predict_with_proba(model, clean_text: str):
    """Run the ML model. Returns (label, confidence, pos_score, neg_score)."""
    if hasattr(model.named_steps["clf"], "predict_proba"):
        proba   = model.predict_proba([clean_text])[0]
        classes = list(model.named_steps["clf"].classes_)
        pos_idx = classes.index(1)
        neg_idx = classes.index(0)
        pos_score = float(proba[pos_idx])
        neg_score = float(proba[neg_idx])
        label     = "Positive" if pos_score >= neg_score else "Negative"
        confidence = float(max(pos_score, neg_score))
    else:
        # SVM — decision function
        decision  = model.decision_function([clean_text])[0]
        sig       = 1 / (1 + np.exp(-abs(decision)))
        label_int = model.predict([clean_text])[0]
        label     = "Positive" if label_int == 1 else "Negative"
        confidence = float(sig)
        pos_score  = float(sig) if label == "Positive" else float(1 - sig)
        neg_score  = float(1 - pos_score)

    return label, round(confidence, 4), round(pos_score, 4), round(neg_score, 4)

def determine_overall_from_aspects(aspect_results: list) -> str:
    """
    If multiple aspects detected, compute an overall label:
    all positive → Positive, all negative → Negative,
    mix → Mixed, all neutral → Neutral
    """
    if not aspect_results:
        return None  # Fall back to ML model output
    labels = [a["sentiment"] for a in aspect_results]
    pos = labels.count("Positive")
    neg = labels.count("Negative")
    neu = labels.count("Neutral")
    total = len(labels)

    if pos == total:
        return "Positive"
    if neg == total:
        return "Negative"
    if neu == total:
        return "Neutral"
    if pos > 0 and neg > 0:
        return "Mixed"
    if pos > neg:
        return "Positive"
    return "Negative"

# ─────────────────────────────────────────
# HIGHLIGHT ASPECTS IN TEXT
# ─────────────────────────────────────────
def highlight_aspects(text: str, aspect_results: list) -> list:
    """
    Return a list of {word, aspect, sentiment} tokens for frontend highlighting.
    """
    words = text.split()
    text_lower = text.lower()
    highlights = []

    # Build lookup: keyword → (aspect, sentiment)
    keyword_map = {}
    for ar in aspect_results:
        for kw in ar.get("matched_keywords", []):
            keyword_map[kw.lower()] = (ar["aspect"], ar["sentiment"])

    for word in words:
        clean_word = re.sub(r"[^a-z]", "", word.lower())
        matched_aspect = None
        matched_sentiment = None
        for kw, (asp, sent) in keyword_map.items():
            if clean_word == kw or kw in clean_word:
                matched_aspect    = asp
                matched_sentiment = sent
                break
        highlights.append({
            "word":      word,
            "aspect":    matched_aspect,
            "sentiment": matched_sentiment,
        })

    return highlights

# ─────────────────────────────────────────
# CORE ANALYSIS FUNCTION
# ─────────────────────────────────────────
def analyze_review(text: str, model_name: str = "lr") -> dict:
    """Full ABSA pipeline for one review."""
    # 1. Select model
    model_map = {
        "lr": LR_MODEL, "logistic": LR_MODEL,
        "svm": SVM_MODEL,
        "nb": NB_MODEL, "naive bayes": NB_MODEL,
    }
    model = model_map.get(model_name.lower(), LR_MODEL)
    display_map = {"lr": "Logistic Regression", "svm": "Linear SVM", "nb": "Naive Bayes"}
    display_name = display_map.get(model_name.lower(), "Logistic Regression")

    # 2. Overall sentiment via ML model
    clean = preprocess(text) or text.lower()
    ml_label, ml_conf, pos_score, neg_score = predict_with_proba(model, clean)

    # 3. Aspect extraction
    found_aspects = extract_aspects(text)

    # 4. Aspect-level sentiment scoring
    aspect_results = []
    for aspect, keywords in found_aspects.items():
        scores = [score_context_window(text, kw) for kw in keywords]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        sentiment, confidence = classify_aspect_sentiment(avg_score)
        aspect_results.append({
            "aspect":            aspect,
            "sentiment":         sentiment,
            "confidence":        confidence,
            "matched_keywords":  keywords,
        })

    # Sort: negative first, then positive, then neutral
    order = {"Negative": 0, "Positive": 1, "Neutral": 2}
    aspect_results.sort(key=lambda x: order.get(x["sentiment"], 3))

    # 5. Final overall sentiment
    aspect_based_overall = determine_overall_from_aspects(aspect_results)
    overall_sentiment    = aspect_based_overall if aspect_based_overall else ml_label

    # 6. Highlights
    highlights = highlight_aspects(text, aspect_results)

    return {
        "review":            text,
        "overall_sentiment": overall_sentiment,
        "ml_sentiment":      ml_label,
        "ml_confidence":     ml_conf,
        "pos_score":         pos_score,
        "neg_score":         neg_score,
        "model_used":        display_name,
        "aspects":           aspect_results,
        "highlights":        highlights,
        "word_count":        len(text.split()),
        "aspects_found":     len(aspect_results),
    }

# ─────────────────────────────────────────
# SQLITE HISTORY DATABASE
# ─────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text   TEXT,
            overall_sent  TEXT,
            aspects_json  TEXT,
            analyzed_at   TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_to_history(review: str, overall: str, aspects: list):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO history (review_text, overall_sent, aspects_json, analyzed_at) VALUES (?,?,?,?)",
        (review[:1000], overall, json.dumps(aspects), datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(limit: int = 50) -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, review_text, overall_sent, aspects_json, analyzed_at "
        "FROM history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [
        {
            "id":           r[0],
            "review":       r[1],
            "overall":      r[2],
            "aspects":      json.loads(r[3]),
            "analyzed_at":  r[4],
        }
        for r in rows
    ]

def get_all_history() -> list:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, review_text, overall_sent, aspects_json, analyzed_at "
        "FROM history ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [
        {
            "id":           r[0],
            "review":       r[1],
            "overall":      r[2],
            "aspects":      json.loads(r[3]),
            "analyzed_at":  r[4],
        }
        for r in rows
    ]

init_db()

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": True, "aspects_configured": len(ASPECT_KEYWORDS)})


@app.route("/stats")
def stats():
    return jsonify(STATS)


@app.route("/models")
def models():
    result = []
    for name, m in METRICS.items():
        result.append({"name": name, "accuracy": m["accuracy"], "f1": m["f1"]})
    best_acc = max(r["accuracy"] for r in result)
    for r in result:
        r["best"] = r["accuracy"] == best_acc
    return jsonify(result)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Expected JSON body."}), 400

    # Accept both 'review' and legacy 'text' keys
    text = str(data.get("review") or data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Provide a 'review' field."}), 400
    if len(text) > 5000:
        return jsonify({"error": "Text exceeds 5000 characters."}), 400

    model_name = str(data.get("model", "lr")).lower()
    result     = analyze_review(text, model_name)

    save_to_history(text, result["overall_sentiment"], result["aspects"])
    return jsonify(result)


@app.route("/analyze-csv", methods=["POST"])
def analyze_csv():
    """
    Upload a CSV with a 'review' or 'text' column.
    Returns list of analysis results.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use form-data key 'file'."}), 400

    file       = request.files["file"]
    model_name = request.form.get("model", "lr")

    content  = file.read().decode("utf-8", errors="replace")
    reader   = csv.DictReader(io.StringIO(content))
    headers  = reader.fieldnames or []

    # Detect text column
    text_col = None
    for c in headers:
        if c.lower().strip() in ("review", "text", "review text", "reviewtext", "comment", "content"):
            text_col = c
            break
    if text_col is None and headers:
        text_col = headers[0]  # fallback to first column

    results = []
    for i, row in enumerate(reader):
        if i >= 500:  # cap at 500 rows per request
            break
        text = str(row.get(text_col, "")).strip()
        if not text or len(text) < 5:
            continue
        res = analyze_review(text[:2000], model_name)
        save_to_history(text, res["overall_sentiment"], res["aspects"])
        results.append({
            "row":               i + 1,
            "review":            text[:300],
            "overall_sentiment": res["overall_sentiment"],
            "aspects_found":     res["aspects_found"],
            "aspects":           res["aspects"],
        })

    # Aggregate stats
    sentiments = [r["overall_sentiment"] for r in results]
    agg = {
        "total":    len(results),
        "positive": sentiments.count("Positive"),
        "negative": sentiments.count("Negative"),
        "neutral":  sentiments.count("Neutral"),
        "mixed":    sentiments.count("Mixed"),
    }
    return jsonify({"summary": agg, "results": results})


@app.route("/dashboard-data")
def dashboard_data():
    """Aggregate stats from history for charts."""
    history = get_all_history()
    if not history:
        return jsonify({"total": 0, "message": "No reviews analyzed yet."})

    overall_counts = {"Positive": 0, "Negative": 0, "Neutral": 0, "Mixed": 0}
    aspect_counts  = {a: {"Positive": 0, "Negative": 0, "Neutral": 0} for a in ASPECT_KEYWORDS}

    for entry in history:
        s = entry["overall"]
        if s in overall_counts:
            overall_counts[s] += 1
        for asp in entry["aspects"]:
            name = asp.get("aspect", "")
            sent = asp.get("sentiment", "")
            if name in aspect_counts and sent in aspect_counts[name]:
                aspect_counts[name][sent] += 1

    # Filter aspects that were actually mentioned
    active_aspects = {k: v for k, v in aspect_counts.items() if sum(v.values()) > 0}

    # Most positive / negative / discussed
    def dominant(aspect_data):
        totals     = {a: sum(v.values()) for a, v in active_aspects.items()}
        pos_ratios = {a: v["Positive"] / max(sum(v.values()), 1) for a, v in active_aspects.items()}
        neg_ratios = {a: v["Negative"] / max(sum(v.values()), 1) for a, v in active_aspects.items()}
        return {
            "most_discussed": max(totals, key=totals.get) if totals else None,
            "most_positive":  max(pos_ratios, key=pos_ratios.get) if pos_ratios else None,
            "most_negative":  max(neg_ratios, key=neg_ratios.get) if neg_ratios else None,
        }

    return jsonify({
        "total":           len(history),
        "overall_counts":  overall_counts,
        "aspect_counts":   active_aspects,
        "insights":        dominant(active_aspects),
        "available_aspects": list(ASPECT_KEYWORDS.keys()),
    })


@app.route("/history")
def history():
    limit = int(request.args.get("limit", 50))
    return jsonify(get_history(limit))


@app.route("/export-csv")
def export_csv():
    """Export full history as CSV."""
    history = get_all_history()
    output  = io.StringIO()
    writer  = csv.writer(output)
    writer.writerow(["id", "review", "overall_sentiment", "aspects", "analyzed_at"])
    for h in history:
        writer.writerow([h["id"], h["review"], h["overall"],
                         json.dumps(h["aspects"]), h["analyzed_at"]])
    output.seek(0)
    from flask import Response
    return Response(
        output.read(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=absa_history.csv"}
    )


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
