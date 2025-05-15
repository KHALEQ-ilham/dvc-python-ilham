from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# Clé secrète pour signer les tokens (à garder confidentielle !)
SECRET_KEY = "your_secret_key_here"  # Mets une vraie clé forte ici !
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Simuler une base d'utilisateurs
fake_users_db = {
    "admin": {
        "username": "admin",
        "password": "admin123"  # Ne pas stocker des mots de passe en clair dans la vraie vie !
    }
}

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Créer un token JWT
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Vérifier un token JWT
def verify_token(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None or username not in fake_users_db:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception

# Authentifier un utilisateur
def authenticate_user(username: str, password: str):
    user = fake_users_db.get(username)
    if not user or user["password"] != password:
        return False
    return user
