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

# +
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
import joblib
import os

# 1. Chargement des données
df = pd.read_csv("data/spam.csv", encoding="ISO-8859-1")

# 2. Conversion des labels (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Vérification et nettoyage de base
df.dropna(subset=['message'], inplace=True)

# 4. Séparation des features et de la cible
X = df['message']
y = df['label']

# 5. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Nettoyage simple
def clean_texts(X, y):
    mask = X.str.strip().notna() & (X.str.len() > 2)
    return X[mask], y[mask]

X_train, y_train = clean_texts(X_train, y_train)
X_test, y_test = clean_texts(X_test, y_test)

# 7. Interface abstraite
class TextClassifier(ABC):
    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

# 8. Modèles
class LogisticTextClassifier(TextClassifier):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = TfidfVectorizer()

    def train(self, X, y):
        X_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_vect, y)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect)

class SVMTextClassifier(TextClassifier):
    def __init__(self):
        self.model = LinearSVC()
        self.vectorizer = TfidfVectorizer()

    def train(self, X, y):
        X_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_vect, y)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect)

class NaiveBayesTextClassifier(TextClassifier):
    def __init__(self):
        self.model = MultinomialNB()
        self.vectorizer = TfidfVectorizer()

    def train(self, X, y):
        X_vect = self.vectorizer.fit_transform(X)
        self.model.fit(X_vect, y)

    def predict(self, X):
        X_vect = self.vectorizer.transform(X)
        return self.model.predict(X_vect)

# 9. Évaluation
def evaluate_model(model: TextClassifier, X_train, y_train, X_test, y_test):
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    return y_pred, acc

# 10. Création dossier modèles
os.makedirs("models", exist_ok=True)

# Entraînement des modèles
model_log = LogisticTextClassifier()
y_pred_log, acc_log = evaluate_model(model_log, X_train, y_train, X_test, y_test)
joblib.dump(model_log, os.path.join("models", "logistic_model.pkl"))

model_svm = SVMTextClassifier()
y_pred_svm, acc_svm = evaluate_model(model_svm, X_train, y_train, X_test, y_test)
joblib.dump(model_svm, os.path.join("models", "svm_model.pkl"))

model_nb = NaiveBayesTextClassifier()
y_pred_nb, acc_nb = evaluate_model(model_nb, X_train, y_train, X_test, y_test)
joblib.dump(model_nb, os.path.join("models", "naive_bayes_model.pkl"))

# 11. Sauvegarde rapport markdown
with open("cml-report.md", "w") as f:
    f.write(f"# Rapport CML\n")
    f.write(f"## Logistic Regression\n")
    f.write(f"- Accuracy: **{acc_log:.4f}**\n")
    f.write(f"![Logistic Confusion](logistic_confusion.png)\n")
    f.write(f"![Logistic ROC](logistic_roc.png)\n\n")
    f.write(f"## SVM\n")
    f.write(f"- Accuracy: **{acc_svm:.4f}**\n")
    f.write(f"![SVM Confusion](svm_confusion.png)\n")
    f.write(f"![SVM ROC](svm_roc.png)\n\n")
    f.write(f"## Naive Bayes\n")
    f.write(f"- Accuracy: **{acc_nb:.4f}**\n")
    f.write(f"![NB Confusion](nb_confusion.png)\n")
    f.write(f"![NB ROC](nb_roc.png)\n")

# 12. Sauvegarde des métriques dans un fichier texte
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy Logistic: {acc_log:.2f}\n")
    f.write(f"Accuracy SVM: {acc_svm:.2f}\n")
    f.write(f"Accuracy Naive Bayes: {acc_nb:.2f}\n")

# 13. Visualisation
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", filename="confusion.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)
    plt.close()

def plot_roc_curve(model: TextClassifier, X_test, y_test, title="ROC Curve", filename="roc.png"):
    y_scores = None
    if hasattr(model.model, "decision_function"):
        y_scores = model.model.decision_function(model.vectorizer.transform(X_test))
    elif hasattr(model.model, "predict_proba"):
        y_scores = model.model.predict_proba(model.vectorizer.transform(X_test))[:, 1]

    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.savefig(filename)
        plt.close()

# Génération des courbes et matrices
plot_confusion_matrix(y_test, y_pred_log, "Logistic Regression - Confusion", "logistic_confusion.png")
plot_roc_curve(model_log, X_test, y_test, "Logistic Regression - ROC", "logistic_roc.png")

plot_confusion_matrix(y_test, y_pred_svm, "SVM - Confusion", "svm_confusion.png")
plot_roc_curve(model_svm, X_test, y_test, "SVM - ROC", "svm_roc.png")

plot_confusion_matrix(y_test, y_pred_nb, "Naive Bayes - Confusion", "nb_confusion.png")
plot_roc_curve(model_nb, X_test, y_test, "Naive Bayes - ROC", "nb_roc.png")

# + endofcell="--"

# # +
import os

# Répertoire courant du notebook
print("Répertoire courant :", os.getcwd())
# -

# --



