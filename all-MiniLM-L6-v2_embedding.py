import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer

import transformers
import os

import json
from sklearn.metrics import roc_auc_score, roc_curve
from pyod.models.iforest import IForest
import time
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


data_dir = 'data\movie_review.npz'

loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()


model_name = "sentence-transformers/all-MiniLM-L6-v2"  
cache_dir = "model/all-MiniLM-L6-v2" 

# Load pre-trained Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder = "model/all-MiniLM-L6-v2", trust_remote_code=True).cuda()

start_time = time.time()
all_embeddings = model.encode(texts)
end_time = time.time()
elapsed_time = end_time - start_time

np.save("embeddings\moviereview/minilm_moviewreview.npy", all_embeddings)
