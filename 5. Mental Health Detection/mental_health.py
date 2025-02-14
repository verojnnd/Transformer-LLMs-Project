import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only needed once)
nltk.download('stopwords')

# load save model
model = AutoModelForSequenceClassification.from_pretrained('saved_mental_status_bert')
tokenizer = AutoTokenizer.from_pretrained('saved_mental_status_bert')
label_encoder = pickle.load(open('mental_health.pkl','rb'))


# custom function
# Get English stopwords from NLTK
stop_words = set(stopwords.words('english'))
def clean_statement(statement):
    # Convert to lowercase
    statement = statement.lower()

    # Remove special characters (punctuation, non-alphabetic characters)
    statement = re.sub(r'[^\w\s]', '', statement)

    # Remove numbers (optional, depending on your use case)
    statement = re.sub(r'\d+', '', statement)

    # Tokenize the statement (split into words)
    words = statement.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Rejoin words into a cleaned statement
    cleaned_statement = ' '.join(words)

    return cleaned_statement
# Detection System (Example)
def detect_mental_health(text):
    cleaned_text = clean_statement(text)
    inputs = tokenizer(cleaned_text, return_tensors="pt", padding=True, truncation=True, max_length=200)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([predicted_class])[0]
# UI app
st.title("Mental Health Status Detection Bert")

input_text = st.text_input("Enter your mental state here....")

if st.button("detect"):
    predicted_class = detect_mental_health(input_text)
    st.write("Predicted Status :", predicted_class)
