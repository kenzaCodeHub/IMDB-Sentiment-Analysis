# IMDB Sentiment Analysis - CNN vs LSTM vs GloVe+LSTM

Comparative study of three deep learning architectures for binary sentiment classification on the IMDB movie reviews dataset (50K reviews).

## Approach

### Data Preprocessing
- Tokenization (top 1000 words), sequence padding to 100 tokens, 80/20 train-test split

### Models Compared

| Model | Architecture | Embeddings |
|---|---|---|
| **1D CNN** | Embedding → Conv1D(16) → AvgPool → Conv1D(32) → AvgPool → Dense(1) | Learned from scratch |
| **LSTM** | Embedding → LSTM(64) → Dense(256) → Dropout(0.5) → Dense(1) | Learned from scratch |
| **GloVe + LSTM** | Frozen GloVe(100d) → LSTM(64, L2) → Dense(256) → Dropout(0.5) → Dense(1) | Pre-trained GloVe 6B |

## Dataset

[IMDB Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — 50,000 movie reviews labeled as positive or negative.

## Tech Stack

Python, TensorFlow/Keras, scikit-learn, Pandas, NumPy, GloVe (6B 100d)
