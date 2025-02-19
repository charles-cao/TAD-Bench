from transformers import BertTokenizer, BertModel
# from pyod.models.iforest import iforest
import torch
import os
from sklearn.utils import shuffle
import json
from sklearn.metrics import roc_auc_score, roc_curve
from pyod.models.iforest import IForest
import time
import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import DataLoader


data_dir = 'data\movie_review.npz'

loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()

model_name = "bert-base-uncased"
cache_dir = "model/bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)

model = model.to(device)



start_time = time.time()
batch_size = 16  # 设置 batch size，例如 16
sentence_embeddings = []

dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)

for batch in dataloader:
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)    
    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    sentence_embeddings.append(batch_embeddings)

sentence_embeddings = np.vstack(sentence_embeddings)

end_time = time.time()

elapsed_time = end_time - start_time

np.save("dir...", sentence_embeddings)
