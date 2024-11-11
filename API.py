import json
import re
import cv2
import clip
import torch
import pickle
import numpy as np
from PIL import Image
import urllib.request
from fastapi import FastAPI
from ultralytics import YOLO
from text2vec import SentenceModel
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
from pydantic import BaseModel
from typing import List, Optional

class ItemData(BaseModel):
    image_link: str
    name_product: str
    name_level_1: str
    name_level_2: str
    name_level_3: str
    price_product: float

app = FastAPI()

# Model text china
model_text_china = SentenceModel('shibing624/text2vec-base-chinese')

# Model CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load('ViT-B/32', device=device)

# Custom YOLOv8m
yolo8 = YOLO('D:\\project_capstone\\YOLO_result\\best_crawl_data.pt').to(device)
names = yolo8.names

# Catboost model for weight prediction
caboost = pickle.load(open("D:\\project_capstone\\weight_catboost\\catboost_model.pkl", "rb"))

# Load scaler once to avoid reloading in each function call
with open('D:\\project_capstone\\weight_catboost\\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# PhoBERT Embedding Reducer
class PhoBERTEmbeddingReducer(nn.Module):
    def __init__(self):
        super(PhoBERTEmbeddingReducer, self).__init__()
        self.pho_bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.fc = nn.Linear(768, 10)  
        torch.manual_seed(42) 
        nn.init.xavier_uniform_(self.fc.weight) 
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.pho_bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :]
        reduced_output = self.fc(last_hidden_state) 
        return reduced_output

tokenizer_phobert = AutoTokenizer.from_pretrained("vinai/phobert-base")
pho_bert_reducer = PhoBERTEmbeddingReducer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\sàáâãèéêìíòóôõùúýăđĩũơưạ-ỹ]', '', text)
    return text

def get_phobert_embedding(text):
    inputs = tokenizer_phobert(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():  
        outputs = pho_bert_reducer(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
    embeddings = outputs.squeeze().cpu().numpy() 
    return embeddings

def get_title_embedding(title):
    tensp_emb = model_text_china.encode(title)
    return tensp_emb

def fetch_image(url, timeout=20):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        print(f"Failed to fetch image from {url}: {e}")
        return None

def emb_largest_crop(img_link):
    img = fetch_image(img_link)
    if img is None:
        return None
    
    results = yolo8(img)
    largest_box = None
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if boxes.size > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_idx = np.argmax(areas)
            largest_box = boxes[max_idx]

    if largest_box is not None:
        xmin, ymin, xmax, ymax = map(int, largest_box)
        image_crop = img[ymin:ymax, xmin:xmax]
        crop_pil = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
        crop_preprocessed = preprocess(crop_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feature_emb = clip_model.encode_image(crop_preprocessed)
            feature = feature_emb.squeeze().cpu().numpy()
        return feature
    else:
        print("No bounding box found.")
        return None

def price_transform(price):
    price_reshaped = np.array(price).reshape(-1, 1)
    scaled_price = scaler.transform(price_reshaped)
    return scaled_price[0][0]

@app.get("/")
async def home():
    return "Product weight estimation"


@app.post("/weight_estimation/predict")
async def predict_weight(api_dev_return: ItemData):
    url = api_dev_return.image_link
    name_product = api_dev_return.name_product
    name_level_1 = api_dev_return.name_level_1
    name_level_2 = api_dev_return.name_level_2
    name_level_3 = api_dev_return.name_level_3
    price_product = api_dev_return.price_product

    emb_img = emb_largest_crop(url)
    if emb_img is None:
        return {"Error": "Cannot fetch image or bounding box."}

    emd_cat1 = get_phobert_embedding(name_level_1)
    emd_cat2 = get_phobert_embedding(name_level_2)
    emd_cat3 = get_phobert_embedding(name_level_3)
    emb_title = get_title_embedding(name_product)
    norm_price = price_transform(price_product)

    combined_features = np.hstack((emb_img, emb_title, emd_cat1, emd_cat2, emd_cat3, norm_price))
    final_combined_features = combined_features.reshape(1, -1)
    weight = caboost.predict(final_combined_features)
    return {"weight_estimation": weight[0]}
