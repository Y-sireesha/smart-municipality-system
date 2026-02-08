# ======================================================
# SMART MUNICIPALITY COMPLAINT SYSTEM
# Dataset: NYC 311 Service Requests (Kaggle)
# ======================================================

import pandas as pd
import re
import spacy
from spacy.pipeline import EntityRuler

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ======================================================
# 1. LOAD spaCy MODEL
# ======================================================
nlp = spacy.load("en_core_web_sm")

# ======================================================
# 2. ADD CUSTOM ENTITY RULES (KEY PART)
# ======================================================
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = [
    # FACILITY
    {"label": "FACILITY", "pattern": "street light"},
    {"label": "FACILITY", "pattern": "traffic signal"},
    {"label": "FACILITY", "pattern": "garbage"},
    {"label": "FACILITY", "pattern": "road"},
    {"label": "FACILITY", "pattern": "water leak"},
    {"label": "FACILITY", "pattern": "drainage"},

    # ADDRESS
    {"label": "ADDRESS", "pattern": [{"IS_DIGIT": True}, {"LOWER": {"IN": ["street", "st", "road", "rd", "avenue", "ave"]}}]},
    {"label": "ADDRESS", "pattern": "park avenue"},
    {"label": "ADDRESS", "pattern": "main street"},
    {"label": "ADDRESS", "pattern": "broadway"},

    # ORGANIZATION
    {"label": "ORG", "pattern": "Department of Transportation"},
    {"label": "ORG", "pattern": "Department of Sanitation"},
    {"label": "ORG", "pattern": "Municipal Corporation"},
    {"label": "ORG", "pattern": "Water Department"}
]

ruler.add_patterns(patterns)

# ======================================================
# 3. TEXT CLEANING FUNCTION
# ======================================================
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

# ======================================================
# 4. LOAD NYC 311 DATASET
# ======================================================
print("Loading NYC 311 dataset...")

df = pd.read_csv("dataset/nyc_311.csv", low_memory=False)

df = df[[
    "Descriptor",
    "Complaint Type",
    "Created Date",
    "Agency Name",
    "Incident Address",
    "Borough"
]]

df = df.dropna()

# ======================================================
# 5. CREATE NATURAL COMPLAINT TEXT
# ======================================================
df["complaint_text"] = (
    df["Descriptor"] + " at " +
    df["Incident Address"] + " in " +
    df["Borough"] + " reported on " +
    df["Created Date"] + " to " +
    df["Agency Name"]
)

df["clean_text"] = df["complaint_text"].apply(clean_text)

print("Dataset prepared:", df.shape)

# ======================================================
# 6. TRAIN SERVICE CLASSIFICATION MODEL
# ======================================================
X = df["clean_text"]
y = df["Complaint Type"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, pred))

# ======================================================
# 7. ENTITY EXTRACTION FUNCTION
# ======================================================
def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

# ======================================================
# 8. FINAL PREDICTION FUNCTION
# ======================================================
def predict_complaint(text):
    vec = vectorizer.transform([clean_text(text)])
    category = model.predict(vec)[0]
    entities = extract_entities(text)
    return category, entities

# ======================================================
# 9. TEST THE SYSTEM
# ======================================================
print("\n--- SMART MUNICIPALITY SYSTEM OUTPUT ---\n")

test_complaint = (
    "Street light not working at 89 Park Avenue in Manhattan "
    "reported on 12 Jan 2026 to Department of Transportation"
)

category, entities = predict_complaint(test_complaint)

print("Complaint Text:")
print(test_complaint)

print("\nPredicted Service Type:")
print(category)

print("\nDetected Entities:")
for ent in entities:
    print(ent)
