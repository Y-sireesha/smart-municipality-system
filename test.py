import spacy

nlp = spacy.load("en_core_web_sm")

text = "Garbage issue near MG Road, Shivamogga on 12 Jan 2026 reported to Municipal Corporation"
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, ent.label_)
