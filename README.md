# Email Spam Detection using LSTM

This repository contains a Machine Learning and Deep Learning project focused on classifying emails as **Spam** or **Ham** (Normal) using Natural Language Processing (NLP) techniques and a Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras.

## 📌 Overview

Email spam is a persistent issue in modern digital communication. This project demonstrates an end-to-end NLP pipeline that:
1. Loads and inspects an email dataset.
2. Balances the class distribution between spam and normal messages.
3. Preprocesses textual data by removing stop words.
4. Tokenizes and pads email text sequences.
5. Trains a Recurrent Neural Network (LSTM) with Early Stopping and Learning Rate Reduction callbacks.
6. Evaluates the model's accuracy and loss on unseen test data.

---

## 📁 Dataset

The project uses the `emails.csv` dataset, which contains:
* **`text`**: The body/subject of the email.
* **`spam`**: Target binary indicator (`1` for Spam, `0` for Normal/Ham).

To address class imbalance, the dataset is balanced by downsampling the normal messages to match the total count of spam messages.

---

## 🛠️ Project Structure & Workflow

1. **Data Loading & EDA**:
   * Inspect data shape, missing values, and target distributions using `pandas` and `seaborn`.
2. **Data Balancing**:
   * Downsample non-spam emails to match the number of spam emails, creating a balanced dataset.
3. **Text Preprocessing**:
   * Remove English stop words using `nltk.corpus.stopwords`.
4. **Tokenization & Padding**:
   * Convert text into numerical sequences using `tensorflow.keras.preprocessing.text.Tokenizer`.
   * Pad sequences to a fixed maximum length (`max_len = 100`).
5. **Model Architecture**:
   * **Embedding Layer**: Converts input tokens to dense vectors.
   * **LSTM Layer**: Captures sequential patterns in the text.
   * **Dense (ReLU)**: Hidden layer for feature representation.
   * **Dense (Sigmoid)**: Output layer for binary classification.
6. **Training & Callbacks**:
   * Optimizer: `Adam`
   * Loss Function: `binary_crossentropy`
   * Callbacks: `EarlyStopping` and `ReduceLROnPlateau` based on validation accuracy.

---

## 🚀 Installation & Requirements

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install numpy pandas seaborn matplotlib nltk tensorflow scikit-learn