from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Inisialisasi pipeline untuk berbagai tugas
sentiment_pipeline = pipeline("sentiment-analysis")
generation_pipeline = pipeline("text-generation", model="distilgpt2")
translation_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-en-id")
summarization_pipeline = pipeline("summarization")
ner_pipeline = pipeline("ner")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    task = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        task = request.form["task"]

        if task == "sentiment":
            result = sentiment_pipeline(user_input)
        elif task == "generation":
            result = generation_pipeline(user_input, max_length=50, num_return_sequences=1)
        elif task == "translation":
            result = translation_pipeline(user_input)
        elif task == "summarization":
            result = summarization_pipeline(user_input)
        elif task == "named_entity_recognition":
            result = ner_pipeline(user_input)

    return render_template("nlp_pipeline.html", result=result, task=task)

if __name__ == "__main__":
    app.run(debug=True)