import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from sentence_transformers import SentenceTransformer
import transformers
import os
# from pyod.models.iforest import iforest
import json
from sklearn.metrics import roc_auc_score, roc_curve
from pyod.models.iforest import IForest
import time
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_dir = 'dir...'

loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()

model_name = "dunzhang/stella_en_400M_v5" 
cache_dir = "model/stella_en_400M_v5" 

model = SentenceTransformer("dunzhang/stella_en_400M_v5", cache_folder = "model/stella_en_400M_v5", trust_remote_code=True).cuda()

start_time = time.time()


all_embeddings = model.encode(texts)

end_time = time.time()

elapsed_time = end_time - start_time
np.save("dir...", all_embeddings)