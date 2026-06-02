# 🛡️ Anti-Hater Filter for Social Networks

> Deep Learning pipeline for automatic toxic content detection and multi-label classification of online comments.

---

## 📌 Overview

Online platforms face a constant challenge: detecting and moderating toxic content at scale. This project builds a **Deep Learning model with recurrent layers (LSTM/GRU)** that automatically classifies comments into 6 toxicity categories simultaneously, producing a binary vector per comment.

The solution is designed to **automate moderation**, reduce human workload, and scale efficiently on high-traffic platforms.

---

## 🎯 Obiettivo

For each comment, the model outputs a **binary vector of 6 elements** — one per toxicity category:

| Label | Description |
|---|---|
| `toxic` | General offensive content |
| `severe_toxic` | Highly aggressive or threatening |
| `obscene` | Explicit or vulgar language |
| `threat` | Direct threats of harm |
| `insult` | Personal attacks or demeaning language |
| `identity_hate` | Hate speech targeting identity groups |

---

## 🏗️ Pipeline

```
Raw Comments
     │
     ▼
Text Preprocessing (cleaning, tokenization, padding)
     │
     ▼
Embedding Layer (trainable word vectors)
     │
     ▼
Recurrent Layers (LSTM / GRU)
     │
     ▼
Dense + Sigmoid Output (6 binary predictions)
     │
     ▼
Multi-label Classification Vector [0/1, 0/1, 0/1, 0/1, 0/1, 0/1]
```

---

## 📁 Dataset

🔗 [Filter_Toxic_Comments_dataset](https://proai-datasets.s3.eu-west-3.amazonaws.com/Filter_Toxic_Comments_dataset.csv)

Multi-label dataset with 6 toxicity categories. Heavily imbalanced — handled with **iterative stratification** to preserve label co-occurrence distributions in train/test splits.

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Deep Learning | `tensorflow`, `keras` |
| NLP Preprocessing | `nltk` |
| Data Manipulation | `numpy`, `pandas` |
| Visualization | `matplotlib`, `seaborn`, `wordcloud` |
| Evaluation | `scikit-learn` (F1-score, ROC-AUC) |
| Stratified Splitting | `iterative-stratification` |

---

## 📊 Approach

### 1. Exploratory Data Analysis
- Label frequency distribution and co-occurrence heatmap
- Word cloud per toxicity category
- Comment length analysis

### 2. Text Preprocessing
- Lowercasing, punctuation removal, stopword filtering
- Tokenization and integer encoding
- Sequence padding to fixed length

### 3. Modeling
- Embedding layer → LSTM/GRU → Dense layers → Sigmoid activation
- Multi-label loss: `binary_crossentropy`
- Iterative stratified K-Fold cross-validation

### 4. Evaluation
- **Per-label**: Precision, Recall, F1-score
- **Global**: ROC-AUC, macro/micro averages
- Confusion matrices per category

---

## 📝 Notes

Developed as part of the **Master in Data Science @ ProfessionAI** (2025–2026).
This project demonstrates applied Deep Learning for NLP in a real-world content moderation scenario.

---

## 📫 Author

**Nicolò Discotto** · [LinkedIn](https://www.linkedin.com/in/nicolo-discotto/) · [GitHub](https://github.com/NicoloDiscotto)
