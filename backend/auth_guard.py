# auth_guard.py
from fastapi import Depends, HTTPException, Header
from firebase_admin import auth
from typing import Optional

def verify_firebase_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = auth.verify_id_token(token)  # valida firma, exp, aud, etc.
        return decoded  # contiene 'uid', 'email', ...
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
