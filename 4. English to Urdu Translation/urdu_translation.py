from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Pydantic model for input validation
class TranslationRequest(BaseModel):
    text: str

# Load model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-ur"  # Ganti dengan model yang sesuai
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Translation function
def translate_text(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, num_beams=4, early_stopping=True)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

# Route for index page
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("urdu.html", {"request": request})

# API route for translation
@app.post("/translate")
async def translate(request: TranslationRequest):
    translation = translate_text(request.text)
    return {"translation": translation}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)