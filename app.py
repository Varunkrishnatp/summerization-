# pip install fastapi uvicorn transformers sentencepiece jinja2 torch
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import re
import torch

# Initialize FastAPI app
app = FastAPI(title="Text Summarization System", description="Summarize dialogues with T5!", version="1.0")

model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Mount templates
templates = Jinja2Templates(directory="templates")

# Input schema for requests
class DialogueInput(BaseModel):
    dialogue: str

# Clean text function
def clean_text(text: str) -> str:
    text = re.sub(r'\r\n', ' ', text)  # Remove carriage returns and line breaks
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'<.*?>', '', text)  # Remove any XML tags
    text = text.strip().lower()  # Strip and convert to lower case
    return text

# Summarization function
def summarize_dialogue(dialogue: str) -> dict:
    dialogue = clean_text(dialogue)
    inputs = tokenizer(dialogue, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate summary
    outputs = model.generate(
        inputs["input_ids"],
        max_length=300,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Calculate word count of the summary
    word_count = len(summary.split())
    
    return {"summary": summary, "word_count": word_count}

# API endpoint for text summarization
@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):
    summary_data = summarize_dialogue(dialogue_input.dialogue)
    return summary_data

# HTML UI
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
