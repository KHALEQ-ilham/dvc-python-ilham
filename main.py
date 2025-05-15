from fastapi import FastAPI, HTTPException, Depends,Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from auth import create_access_token, verify_token, authenticate_user
from prometheus_client import Gauge

# Initialisation de l'app
app = FastAPI(title="Multi-Model Spam Detector API")

# Chargement des modèles
models = {
    "logistic": {
        "vectorizer": joblib.load("models/logistic_vectorizer.pkl"),
        "model": joblib.load("models/logistic_model.pkl")
    },
    "svm": {
        "vectorizer": joblib.load("models/svm_vectorizer.pkl"),
        "model": joblib.load("models/svm_model.pkl")
    },
    "naivebayes": {
        "vectorizer": joblib.load("models/naivebayes_vectorizer.pkl"),
        "model": joblib.load("models/naivebayes_model.pkl")
    },
    "lstm": {
        "model": load_model("models/lstm_model.h5"),
        "tokenizer": joblib.load("models/lstm_tokenizer.pkl")
    }
}
# Métriques de précision par modèle
model_accuracy = Gauge("model_accuracy", "Accuracy of ML models", ["model"])

# Exemple statique : tu peux automatiser ça plus tard
model_accuracy.labels(model="logistic").set(0.91)
model_accuracy.labels(model="svm").set(0.88)
model_accuracy.labels(model="naivebayes").set(0.86)
model_accuracy.labels(model="lstm").set(0.89)

# Schéma de la requête
class Message(BaseModel):
    text: str
    model_name: str  # "logistic", "svm", "naivebayes", "lstm"

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API multi-modèles de détection de spam !"}

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Nom d'utilisateur ou mot de passe incorrect")
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
def predict_spam(message: Message, username: str = Depends(verify_token)):
    model_name = message.model_name.lower()

    if model_name not in models:
        raise HTTPException(status_code=400, detail="Modèle non reconnu")

    model_data = models[model_name]
    
    if model_name == "lstm":
        tokenizer = model_data["tokenizer"]
        model = model_data["model"]
        
        X_seq = tokenizer.texts_to_sequences([message.text])
        X_pad = pad_sequences(X_seq, maxlen=100)
        pred = np.argmax(model.predict(X_pad), axis=1)[0]
    else:
        vectorizer = model_data["vectorizer"]
        model = model_data["model"]
        
        vect = vectorizer.transform([message.text])
        pred = model.predict(vect)[0]

    return {"user": username, "model": model_name, "prediction": "spam" if pred == 1 else "ham"}

@app.get("/models")
def get_models():
    return {"models": list(models.keys())}



prediction_counter = Counter("prediction_requests", "Nombre de requêtes de prédiction")
prediction_latency = Histogram("prediction_latency_seconds", "Temps de latence des prédictions")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
