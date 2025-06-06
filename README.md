# TAD-Bench: A Comprehensive Benchmark for Text Anomaly Detection

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## Overview

TAD-Bench is a comprehensive benchmark specifically designed for **Text Anomaly Detection**. Our objective is to enable systematic evaluation of state-of-the-art embedding models and anomaly detection algorithms, providing valuable insights for NLP applications requiring outlier detection.

### Key Features
- **Diverse Embedding Models**: Support for BERT, LLaMA, Qwen, OpenAI embeddings, and more
- **Multiple Detection Algorithms**: Integration with 7+ anomaly detection methods (KNN, OCSVM, Isolation Forest, etc.)
- **Standardized Evaluation**: Consistent metrics and experimental protocols
- **Real-world Datasets**: Curated datasets from various domains (movie reviews, social media, etc.)

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/charles-cao/TAD-Bench.git
cd TAD-Bench
```

### Basic Usage

1. **Generate Embeddings**:
```bash
# Using BERT
python embeddings/bert_embedding.py

# Using OpenAI API (requires API key)
python embeddings/openai_embedding.py
```

2. **Run Evaluation**:
```bash
jupyter notebook eval.ipynb
```

## 📊 Supported Models

### Embedding Models
| Model | Dimensions | Type | Paper/Source |
|-------|------------|------|--------------|
| BERT-base-uncased | 768 | Transformer | [Devlin et al., 2018] |
| all-MiniLM-L6-v2 | 384 | Sentence Transformer | [Sentence-BERT] |
| LLaMA-3.2-1B | 2048 | Large Language Model | [Meta AI] |
| Qwen2.5-1.5B | 1536 | Large Language Model | [Alibaba] |
| Stella-400M-v5 | 1024 | Sentence Transformer | [Dunzhang] |
| OpenAI text-embedding-ada-002 | 1536 | API-based | [OpenAI] |
| OpenAI text-embedding-3-small | 1536 | API-based | [OpenAI] |
| OpenAI text-embedding-3-large | 3072 | API-based | [OpenAI] |

### Anomaly Detection Algorithms
- **K-Nearest Neighbors (KNN)**
- **One-Class SVM (OCSVM)**
- **Isolation Forest (iForest)**
- **Local Outlier Factor (LOF)**
- **Empirical Cumulative Outlier Detection (ECOD)**
- **Isolation-based Nearest Neighbor Ensemble (iNNE)**
- **LUNAR**



## 🔧 Configuration

### OpenAI API Setup
For OpenAI embeddings, you need to set your API key:

```python
# In openai_embedding.py
client = OpenAI(
  api_key = xxx # your key here
)
```

### Custom Datasets
To use your own dataset:

1. Format your data as `.npz` file with `data` and `label` arrays
2. Update the `data_dir` path in embedding scripts
3. Run the embedding generation and evaluation pipeline

## 📚 Citation

If you use TAD-Bench in your research, please cite:

```bibtex
@misc{cao2025tadbenchcomprehensivebenchmarkembeddingbased,
      title={TAD-Bench: A Comprehensive Benchmark for Embedding-Based Text Anomaly Detection}, 
      author={Yang Cao and Sikun Yang and Chen Li and Haolong Xiang and Lianyong Qi and Bo Liu and Rongsheng Li and Ming Liu},
      year={2025},
      eprint={2501.11960},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.11960}, 
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
