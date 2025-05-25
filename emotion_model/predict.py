from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
import torch
import os
import pathlib

MODEL_PATH = str(pathlib.Path(__file__).parent / "roberta_emotion")


tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_PATH)
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH, torch_dtype=torch.float32, use_safetensors=True)
model.eval()

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    label_id = torch.argmax(outputs.logits, dim=1).item()
    label_map = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    return label_map[label_id]

def get_kaggle_id():
    return "angelbenito"
