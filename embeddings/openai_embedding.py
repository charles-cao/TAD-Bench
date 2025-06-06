import json
from openai import OpenAI
import tiktoken

import json
from sklearn.metrics import roc_auc_score, roc_curve
from pyod.models.iforest import IForest
import time
import numpy as np


data_dir = 'dir...'

loaded = np.load(data_dir, allow_pickle=True)
texts = loaded['data'].tolist()
labels = loaded['label'].tolist()

start_time = time.time()

# 确保 API 密钥已正确配置
client = OpenAI(
  api_key = xxx #your key here
)


tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
# tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")
# tokenizer = tiktoken.encoding_for_model("text-embedding-3-large")

max_tokens = 8191

batches = []  
batch = [] 
batch_tokens = 0

for text in texts:
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        print(f"Text is too long ({len(tokens)} tokens). It will be truncated.")
        text = tokenizer.decode(tokens[:max_tokens])
        tokens = tokens[:max_tokens]

    if batch_tokens + len(tokens) > max_tokens:
        batches.append(batch)
        batch = []
        batch_tokens = 0

    batch.append(text)
    batch_tokens += len(tokens)

if batch:
    batches.append(batch)

print(f"Total batches created: {len(batches)}")

embeddings = []
for i, batch in enumerate(batches):
    print(f"Processing batch {i + 1}/{len(batches)} with {len(batch)} texts...")
    response = client.embeddings.create(
        input=batch,
        model="text-embedding-ada-002"
        # model = "text-embedding-3-small"
        # model = "text-embedding-3-large"
    )
    embeddings.extend([item.embedding for item in response.data])


embeddings = np.array(embeddings)
end_time = time.time()
np.save("dir...", embeddings)