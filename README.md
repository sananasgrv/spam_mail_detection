# Email Spam Classification using Deep Learning (LSTM & CNN)

This repository contains end-to-end TensorFlow/Keras implementations for classifying emails as **Spam** or **Ham (Normal)** using Natural Language Processing (NLP) and Deep Learning architectures. 

Two distinct modeling approaches are explored across two Jupyter Notebooks using the same dataset:
1. **LSTM-based Sequential Architecture** (Recurrent Neural Network)
2. **1D CNN-based Architecture** (Convolutional Neural Network for Text)

---

## 📊 Dataset Overview

* **Source File:** `emails.csv`
* **Total Samples:** 5,728 emails
* **Columns:** 
  * `text`: Raw text body of the email (starts with "Subject: ...")
  * `spam`: Binary target label (`1` = Spam, `0` = Ham)
* **Class Balancing:** The dataset exhibits a class imbalance (~1,368 spam vs. ~4,360 ham). Both notebooks implement random undersampling of the majority class to create a perfectly balanced dataset (1:1 ratio) before model training.

---

## 🛠️ Data Preprocessing & Pipeline

Across both implementations, the text processing pipeline consists of:
1. **Stopword Removal:** Filtering out common English stopwords using `nltk.corpus.stopwords`.
2. **Train/Test Split:** Stratified 80/20 train-test split using `scikit-learn`.
3. **Tokenization:** Text sequence tokenization using Keras `Tokenizer`.
4. **Sequence Padding:** Standardizing input lengths using `pad_sequences` (`maxlen=100`, `padding='post'`, `truncating='post'`).

---

## 🚀 Model Architectures

### 1. LSTM Pipeline (`Notebook 1`)
Designed to capture long-term sequential dependencies in text sequences.

* **Input Layer:** `(batch_size, 100)`
* **Embedding Layer:** Vocabulary Size + 1 $\rightarrow$ Dimension 32
* **LSTM Layer:** 16 Units
* **Dense Layer:** 16 Units (ReLU)
* **Output Layer:** 1 Unit (Sigmoid)
* **Optimization:** Adam Optimizer, Binary Cross-Entropy Loss

### 2. 1D CNN Pipeline (`Notebook 2`)
Designed to extract local n-gram feature maps rapidly across text sequences.

* **Input Layer:** `(batch_size, 100)`
* **Embedding Layer:** Vocabulary Size + 1 $\rightarrow$ Dimension 32
* **Conv1D Layer:** 32 Filters, Kernel Size = 3 (ReLU)
* **GlobalMaxPool1D Layer:** Feature map reduction
* **Dense Layer:** 16 Units (ReLU)
* **Output Layer:** 1 Unit (Sigmoid)
* **Optimization:** Adam Optimizer, Binary Cross-Entropy Loss

---

## 💻 Tech Stack & Requirements

* **Language:** Python 3.x
* **Deep Learning Framework:** TensorFlow / Keras
* **NLP Tools:** NLTK
* **Data Handling & Visualization:** Pandas, NumPy, Seaborn, Matplotlib, Scikit-Learn

To install the required dependencies:

```bash
pip install numpy pandas seaborn matplotlib nltk tensorflow scikit-learn
