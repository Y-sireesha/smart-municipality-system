print("🔥 app.py started")

from flask import Flask, render_template, request
from model import analyze_complaint
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    entities = None

    if request.method == "POST":
        complaint = request.form.get("complaint")
        result, entities = analyze_complaint(complaint)

    return render_template(
        "index.html",
        result=result,
        entities=entities
    )

if __name__ == "__main__":
    print("🚀 Flask server starting...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
