import pandas as pd
import re
import spacy
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Global objects (lazy loading)
nlp = None
model = None
vectorizer = None


def clean_text(text):
    return re.sub(r'[^a-zA-Z ]', '', text.lower())


def load_model():
    global nlp, model, vectorizer

    if model is not None:
        return  # already loaded

    print("🔹 Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    # -------- ENTITY RULER (CUSTOM ENTITIES) --------
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns([
        {"label": "FACILITY", "pattern": "street light"},
        {"label": "FACILITY", "pattern": "garbage"},
        {"label": "FACILITY", "pattern": "road"},
        {"label": "FACILITY", "pattern": "water leak"},
        {"label": "FACILITY", "pattern": "traffic signal"},
        {"label": "FACILITY", "pattern": "drainage"},
        {"label": "ADDRESS", "pattern": "park avenue"},
        {"label": "ADDRESS", "pattern": "main street"},
        {"label": "ORG", "pattern": "Department of Transportation"},
        {"label": "ORG", "pattern": "Department of Sanitation"},
        {"label": "ORG", "pattern": "Water Department"},
        {"label": "ORG", "pattern": "Municipal Corporation"},
    ])

    print("🔹 Loading NYC 311 dataset (filtered)...")
    df = pd.read_csv("dataset/nyc_311.csv", nrows=5000, low_memory=False)

    df = df[[
        "Descriptor", "Complaint Type",
        "Created Date", "Agency Name",
        "Incident Address", "Borough"
    ]].dropna()

    # -------- REMOVE NOISE COMPLAINTS --------
    allowed_services = [
        "Street Light Condition",
        "Sanitation Condition",
        "Water System",
        "Sewer",
        "Street Condition",
        "Traffic Signal Condition"
    ]

    df = df[df["Complaint Type"].isin(allowed_services)]

    # -------- BUILD NATURAL TEXT --------
    df["text"] = (
        df["Descriptor"] + " at " +
        df["Incident Address"] + " in " +
        df["Borough"] + " reported on " +
        df["Created Date"] + " to " +
        df["Agency Name"]
    )

    df["clean"] = df["text"].apply(clean_text)

    print("🔹 Training ML model...")
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(df["clean"])
    y = df["Complaint Type"]

    model = MultinomialNB()
    model.fit(X, y)

    print("✅ Model loaded successfully")


def analyze_complaint(text):
    load_model()

    # -------- SERVICE PREDICTION --------
    vec = vectorizer.transform([clean_text(text)])
    category = model.predict(vec)[0]

    # -------- ENTITY EXTRACTION (CLEAN FORMAT) --------
    doc = nlp(text)

    entities = {
        "LOCATION": [],
        "FACILITY": [],
        "ADDRESS": [],
        "DATE": [],
        "ORGANIZATION": []
    }

    address_parts = []

    for ent in doc.ents:
        if ent.label_ == "GPE":
            entities["LOCATION"].append(ent.text)

        elif ent.label_ == "ORG":
            entities["ORGANIZATION"].append(ent.text)

        elif ent.label_ == "DATE":
            entities["DATE"].append(ent.text)

        elif ent.label_ in ["FACILITY", "FAC"]:
            entities["FACILITY"].append(ent.text)

        elif ent.label_ == "CARDINAL":
            address_parts.append(ent.text)

    if address_parts:
        entities["ADDRESS"].append(" ".join(address_parts))

    # Remove empty fields
    entities = {k: v for k, v in entities.items() if v}

    return category, entities
