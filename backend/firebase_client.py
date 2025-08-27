import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import firebase_admin
from firebase_admin import credentials, firestore, storage

# Inicializa Firebase
def init_firebase():
    if not firebase_admin._apps:
        cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("FIREBASE_PROJECT_ID")
        if not cred_path:
            raise RuntimeError("Falta GOOGLE_APPLICATION_CREDENTIALS en .env")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {
            "projectId": project_id,
            # Si usarás Storage, setea bucket por defecto (opcional)
            # "storageBucket": f"{project_id}.appspot.com"
        })

db = None

def firestore_client():
    global db
    if db is None:
        init_firebase()
        db = firestore.client()
    return db

def now_ts():
    return datetime.now(timezone.utc)

# Helpers de escritura ---------------------------------

def create_upload(user_id: str, image_path: str, sex: int, goals: Dict[str, Any], status: str="pending") -> str:
    doc = {
        "userId": user_id,
        "imagePath": image_path,
        "sex": sex,
        "goals": goals,
        "status": status,
        "createdAt": now_ts()
    }
    ref = firestore_client().collection("uploads").add(doc)[1]
    return ref.id

def update_upload(upload_id: str, data: Dict[str, Any]):
    firestore_client().collection("uploads").document(upload_id).update(data)

def create_prediction(upload_id: str, height_m: float, weight_kg: float, class_idx: int, class_name: str) -> str:
    doc = {
        "uploadId": upload_id,
        "height_m": height_m,
        "weight_kg": weight_kg,
        "class_idx": class_idx,
        "class_name": class_name,
        "createdAt": now_ts()
    }
    ref = firestore_client().collection("predictions").add(doc)[1]
    return ref.id

def create_plan(pred_id: str, user_id: str, plan: Dict[str, Any]) -> str:
    doc = {
        "predId": pred_id,
        "userId": user_id,
        "plan": plan,
        "createdAt": now_ts()
    }
    ref = firestore_client().collection("plans").add(doc)[1]
    return ref.id
