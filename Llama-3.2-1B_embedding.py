import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# from pyod.models.iforest import iforest
import json
from sklearn.metrics import roc_auc_score, roc_curve
from pyod.models.iforest import IForest
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
import transformers
import os
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = 'data\movie_review.npz'
loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()

model_name = "meta-llama/Llama-3.2-1B"  
cache_dir = "model/Llama-3.2-1B"  

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, device_map="auto")

start_time = time.time()


# 将 pad_token 设置为 eos_token
tokenizer.pad_token = tokenizer.eos_token

# 定义一个批量大小
batch_size = 16

# 使用 DataLoader 分批加载数据
dataloader = DataLoader(texts, batch_size=batch_size, shuffle=False)

all_embeddings = []

for batch in dataloader:
    encoded_inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
    input_ids = encoded_inputs["input_ids"].to(device)
    attention_mask = encoded_inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    valid_token_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
    batch_embeddings = valid_token_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    all_embeddings.append(batch_embeddings.cpu().numpy())
all_embeddings = np.vstack(all_embeddings)

end_time = time.time()

elapsed_time = end_time - start_time

print("All embeddings shape:", all_embeddings.shape)

np.save("dir...", all_embeddings)

