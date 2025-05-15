# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + endofcell="--"
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # +
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import logging
import joblib
import os
import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.pytorch

from transformers import BertTokenizer, BertForSequenceClassification
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Chargement des données
df = pd.read_csv("data/spam.csv", encoding="ISO-8859-1")
df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "message"})

# 2. Conversion des labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Nettoyage
df.dropna(subset=['message'], inplace=True)

# 4. Features & target
X = df['message']
y = df['label']

# 5. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Filtrage
X_train = X_train[X_train.str.len() > 2]
y_train = y_train[X_train.index]
X_test = X_test[X_test.str.len() > 2]
y_test = y_test[X_test.index]

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Hate Speech Detection")



logging.basicConfig(level=logging.INFO)

def log_evaluation(func):
    @functools.wraps(func)
    def wrapper(model, name):
        logging.info(f"🚀 Début de l'évaluation du modèle {name}")
        result = func(model, name)
        logging.info(f"✅ Fin de l'évaluation du modèle {name} avec précision {result:.4f}")
        return result
    return wrapper

class TextClassifier(ABC):
    @abstractmethod
    def train(self, X, y): pass
    @abstractmethod
    def predict(self, X): pass

class LogisticTextClassifier(TextClassifier):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        X_vec = self.vectorizer.fit_transform(X)
        self.model.fit(X_vec, y)

    def predict(self, X):
        X_vec = self.vectorizer.transform(X)
        return self.model.predict(X_vec)

class SVMTextClassifier(LogisticTextClassifier):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LinearSVC()

class NaiveBayesTextClassifier(LogisticTextClassifier):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB()


class LSTMTextClassifier(TextClassifier):
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.model = None

    def train(self, X, y):
        self.tokenizer.fit_on_texts(X)
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=100)
        self.model = Sequential([
            Embedding(len(self.tokenizer.word_index) + 1, 128),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(2, activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_pad, y, epochs=3, batch_size=32)

    def predict(self, X):
        X_seq = self.tokenizer.texts_to_sequences(X)
        X_pad = pad_sequences(X_seq, maxlen=100)
        return np.argmax(self.model.predict(X_pad), axis=1)


def plot_confusion_matrix(y_true, y_pred, name):
    plt.figure(figsize=(4, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"{name}_conf.png")
    plt.close()

def plot_roc_curve(model, X, y, name):
    try:
        if isinstance(model, LSTMTextClassifier):
            X_seq = model.tokenizer.texts_to_sequences(X)
            X_pad = pad_sequences(X_seq, maxlen=100)
            probs = model.model.predict(X_pad)
            scores = probs[:, 1]  # Probabilité de la classe 1 (spam)
        else:
            if hasattr(model.model, 'decision_function'):
                scores = model.model.decision_function(model.vectorizer.transform(X))
            else:
                scores = model.model.predict_proba(model.vectorizer.transform(X))[:, 1]
        
        fpr, tpr, _ = roc_curve(y, scores)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {name}")
        plt.legend(loc="lower right")
        plt.savefig(f"{name}_roc.png")
        plt.close()
    except Exception as e:
        print(f"ROC non applicable pour {name}: {e}")
@log_evaluation
@log_evaluation
def evaluate_model(model, name):
    with mlflow.start_run(run_name=name):
        # 1. Log des paramètres (exemple simple ici)
        mlflow.log_param("model_name", name)
        
        # 2. Entraînement du modèle
        model.train(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 3. Log de métriques
        mlflow.log_metric("accuracy", acc)

        # 4. Sauvegarde des visualisations
        plot_confusion_matrix(y_test, y_pred, name)
        plot_roc_curve(model, X_test, y_test, name)
        mlflow.log_artifact(f"{name}_conf.png")
        mlflow.log_artifact(f"{name}_roc.png")

        # 5. Sauvegarde des modèles
        model_dir = f"models/{name.lower()}"
        os.makedirs(model_dir, exist_ok=True)

        if isinstance(model, LSTMTextClassifier):
            model.model.save(f"{model_dir}/model.h5")
            joblib.dump(model.tokenizer, f"{model_dir}/tokenizer.pkl")
            mlflow.keras.log_model(model.model, artifact_path="model")
            mlflow.log_artifact(f"{model_dir}/tokenizer.pkl")
        else:
            joblib.dump(model.vectorizer, f"{model_dir}/vectorizer.pkl")
            joblib.dump(model.model, f"{model_dir}/model.pkl")
            mlflow.sklearn.log_model(model.model, artifact_path="model")
            mlflow.log_artifact(f"{model_dir}/vectorizer.pkl")

        return acc

def model_generator():
    yield ("Logistic", LogisticTextClassifier())
    yield ("SVM", SVMTextClassifier())
    yield ("NaiveBayes", NaiveBayesTextClassifier())
    yield ("LSTM", LSTMTextClassifier())
class ModelIterator:
    def __init__(self, models):
        self.models = models
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.models):
            raise StopIteration
        model = self.models[self.index]
        self.index += 1
        return model


os.makedirs("models", exist_ok=True)

accuracies = {}
models = ModelIterator(list(model_generator()))
for name, model in models:
    acc = evaluate_model(model, name)
    accuracies[name] = acc



with open("cml-report.md", "w") as f:
    for name, acc in accuracies.items():
        f.write(f"## {name}\n- Accuracy: {acc:.4f}\n![{name} Confusion]({name}_conf.png)\n\n")

with open("metrics.txt", "w") as f:
    f.write("Accuracy Scores:\n")
    for name, acc in accuracies.items():
        f.write(f"{name}: {acc:.4f}\n")


print("✅ Tous les modèles ont été entraînés et évalués avec succès.")
print("📄 Rapport Markdown généré : cml-report.md")
print("📁 Modèles sauvegardés dans le dossier 'models'")
print("📊 Fichier de métriques généré : metrics.txt") 
# -




# # + endofcell="--"






# --

# + endofcell="---"
# # # +
import os

# Répertoire courant du notebook
print("Répertoire courant :", os.getcwd())
# -

# --

# ---
