# main.py — pipeline End-to-End (CNN -> Planner -> Firestore)
import io, os, json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

from firebase_client import (
    create_upload, update_upload, create_prediction, create_plan
)

load_dotenv()

# =========================
# RUTAS – AJUSTA AQUÍ
# =========================
BASE_DIR = r"E:\TESIS\Redes neuronales\RED CONVOLUCIONAL"

# Red 1 (CNN antropométrica)
MODEL_PATH  = fr"{BASE_DIR}\modelo_antropometrico.keras"
SCALER_PATH = fr"{BASE_DIR}\scaler_medidas.pkl"

# Red 2 (Planner MLP)
PL_DIR          = fr"{BASE_DIR}\ckpts_planner"
PL_MODEL_PATH   = fr"{PL_DIR}\planner_best.keras"      # o planner_mlp.keras si prefieres
PL_XSC_PATH     = fr"{PL_DIR}\planner_x_scaler.pkl"
PL_YSC_PATH     = fr"{PL_DIR}\planner_y_scaler.pkl"

IMG_SIZE = (224, 224)
CLASS_NAMES = ["desnutricion", "riesgo", "normal", "sobrepeso", "obesidad"]

# =========================
# Custom objects usados al entrenar la CNN
# =========================
try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.saving import register_keras_serializable

huber_loss = tf.keras.losses.Huber(delta=1.5)
mae_loss   = tf.keras.losses.MeanAbsoluteError()
def mixed_reg_loss(y_true, y_pred):
    return 0.7 * huber_loss(y_true, y_pred) + 0.3 * mae_loss(y_true, y_pred)

@register_keras_serializable(package="custom")
def apply_weight_correction(t):
    reg_base, w_delta = t
    altura_z = reg_base[:, :1]
    peso_z   = reg_base[:, 1:2] + 0.1 * w_delta
    return tf.concat([altura_z, peso_z], axis=1)

@register_keras_serializable(package="custom")
def preprocess_input_registered(x):
    return preprocess_input(x)

@register_keras_serializable(package="custom")
class BrightnessContrastLayer(tf.keras.layers.Layer):
    def __init__(self, max_delta_brightness=0.05, lower_c=0.9, upper_c=1.1, **kwargs):
        super().__init__(**kwargs)
        self.max_delta_brightness = max_delta_brightness
        self.lower_c = lower_c
        self.upper_c = upper_c
    def call(self, x, training=None):
        if training:
            x = tf.image.random_brightness(x, max_delta=self.max_delta_brightness)
            x = tf.image.random_contrast(x, lower=self.lower_c, upper=self.upper_c)
        return x

# =========================
# Utilidades Planner
# =========================
FEATURE_COLS = [
    "sex","h_m","w_kg","bmi","bmi_cls","pt_cat_idx",
    "ideal_w_kg","delta_to_ideal",
    "goal_3200_s","goal_push","goal_sit",
    "knee","shoulder","back","vegan","lactose_free","gluten_free"
]

def bmi_class(b):
    return 0 if b < 18.5 else 1 if b < 20 else 2 if b < 25 else 3 if b < 30 else 4

EXERCISES = {
    "upper": ["Flexiones", "Remo mancuerna", "Press militar", "Remo invertido"],
    "lower": ["Sentadilla goblet", "Zancadas", "Puente de glúteo", "Peso muerto rumano"],
    "core" : ["Plancha", "Dead bug", "Bird-dog", "Crunch"]
}
RECIPES = [
    {"title":"Avena con yogurt y frutas","kcal":450,"protein_g":20,"fat_g":12,"carbs_g":65,"tags":[]},
    {"title":"Tofu salteado + arroz","kcal":650,"protein_g":35,"fat_g":18,"carbs_g":85,"tags":["vegano","sin_lactosa"]},
    {"title":"Pollo + quinoa + ensalada","kcal":700,"protein_g":55,"fat_g":20,"carbs_g":60,"tags":["sin_gluten","sin_lactosa"]},
    {"title":"Ensalada de garbanzos","kcal":550,"protein_g":25,"fat_g":18,"carbs_g":70,"tags":["vegano","sin_lactosa","sin_gluten"]},
    {"title":"Salmón + camote + brócoli","kcal":720,"protein_g":45,"fat_g":30,"carbs_g":55,"tags":["sin_gluten","sin_lactosa"]},
]

def choose_meals(kcal, vegan, lact_free, glu_free):
    allowed = []
    for r in RECIPES:
        if vegan and "vegano" not in r["tags"]: continue
        if lact_free and "sin_lactosa" not in r["tags"]: continue
        if glu_free and "sin_gluten" not in r["tags"]: continue
        allowed.append(r)
    if not allowed:
        allowed = RECIPES
    per = kcal / 3.0
    allowed = sorted(allowed, key=lambda r: abs(r["kcal"] - per))
    return allowed[:3]

def split_minutes(total, parts):
    if total <= 0 or parts <= 0: return [0]*parts
    base = int(total // parts); rem = int(total - base*parts)
    arr = [base] * parts
    for i in range(rem): arr[i] += 1
    return arr

def plan_from_outputs(out, vegan, lact_free, glu_free):
    kcal     = out["kcal"]
    protein  = out["protein_g"]
    fat      = out["fat_g"]
    carbs    = out["carbs_g"]
    easy     = max(0, out["easy_runs_per_wk"])
    inter    = max(0, out["intervals_per_wk"])
    strength = max(0, out["strength_per_wk"])
    push_sets = int(round(out["push_sets"]))
    sit_sets  = int(round(out["sit_sets"]))

    blocks = []
    if easy > 0:
        blocks += [("Easy run", m) for m in split_minutes(int(round(45*easy)), max(1, int(round(easy))))]
    if inter > 0:
        blocks += [("Intervals", m) for m in split_minutes(int(round(20*inter)), max(1, int(round(inter))))]
    blocks += [("Long run", 40)]

    days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    sched = []
    up = int(round(strength/2)); low = int(round(strength - up)); i = 0
    for d in days:
        day = {"day": d, "sessions": []}
        if i < len(blocks):
            n, m = blocks[i]; i += 1
            if m > 0:
                day["sessions"].append({"type":"run","name":n,"minutes":m})
        if up > 0:
            day["sessions"].append({"type":"strength","focus":"upper","exercises":EXERCISES["upper"][:3],"pushups_sets":push_sets})
            up -= 1
        elif low > 0:
            day["sessions"].append({"type":"strength","focus":"lower","exercises":EXERCISES["lower"][:3]})
            low -= 1
        else:
            day["sessions"].append({"type":"core","exercises":EXERCISES["core"][:2],"situps_sets":sit_sets})
        sched.append(day)

    meals = choose_meals(kcal, vegan, lact_free, glu_free)
    return {
        "nutrition": {
            "targets_per_day": {
                "kcal": int(round(kcal)),
                "protein_g": int(round(protein)),
                "fat_g": int(round(fat)),
                "carbs_g": int(round(carbs)),
            },
            "meals_example": meals
        },
        "training": sched
    }

def pretty_json(d: dict) -> JSONResponse:
    return JSONResponse(content=json.loads(json.dumps(d, indent=2, ensure_ascii=False)))

# =========================
# Cargar modelos al iniciar
# =========================
app = FastAPI(title="Anthro + Planner", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # en prod restringe
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CNN = load_model(
    MODEL_PATH,
    custom_objects={
        "mixed_reg_loss": mixed_reg_loss,
        "apply_weight_correction": apply_weight_correction,
        "preprocess_input": preprocess_input_registered,
        "BrightnessContrastLayer": BrightnessContrastLayer,
    },
    compile=False,
)
SCALER_Y = joblib.load(SCALER_PATH)

PLANNER = load_model(PL_MODEL_PATH, compile=False)
PL_XSC   = joblib.load(PL_XSC_PATH)
PL_YSC   = joblib.load(PL_YSC_PATH)

def load_image_bytes(b: bytes):
    img = Image.open(io.BytesIO(b)).convert("RGB").resize(IMG_SIZE)
    return img_to_array(img).astype("float32")

# =========================
# Endpoint
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}
@app.post("/process")
async def process(
    file: UploadFile = File(...),
    sex: int = Form(...),  # 0=female, 1=male
    goal_3200_s: int = Form(...),
    goal_push: int = Form(...),
    goal_sit: int = Form(...),
    user_id: str = Form("demo-user"),
    knee: int = Form(0), shoulder: int = Form(0), back: int = Form(0),
    vegan: int = Form(0), lactose_free: int = Form(0), gluten_free: int = Form(0)
):
    # 1) Cargar upload en Firestore
    goals = {"run_s": goal_3200_s, "push": goal_push, "sit": goal_sit}
    upload_id = create_upload(
        user_id=user_id,
        image_path=file.filename, 
        sex=sex,
        goals=goals,
        status="pending"
    )
    # 2) Leer imagen en memoria
    img_bytes = await file.read()
    # 3) Red 1: antropométrica
    arr = load_image_bytes(img_bytes)              # (224,224,3) float32
    X = np.expand_dims(arr, 0)                     # (1,224,224,3)
    S = np.array([[sex]], dtype="float32")         # (1,1)
    reg_z, cls_prob = CNN.predict([X, S], verbose=0)
    h_m, w_kg = SCALER_Y.inverse_transform(reg_z)[0].tolist()
    cls_idx = int(cls_prob.argmax(axis=1)[0])
    class_name = CLASS_NAMES[cls_idx]
    # 4) Guardar prediction en Firestore
    pred_id = create_prediction(upload_id, h_m, w_kg, cls_idx, class_name)
    update_upload(upload_id, {"status": "predicted", "predId": pred_id})

    # 5) Red 2: preparar features (17 exactas y en orden)
    bmi = w_kg / (h_m * h_m)
    bcls = bmi_class(bmi)
    ideal = 22.0 * (h_m * h_m)
    delta = w_kg - ideal

    feats = pd.DataFrame([{
        "sex": 1.0 if sex >= 0.75 else 0.0,
        "h_m": h_m, "w_kg": w_kg, "bmi": bmi, "bmi_cls": bcls,
        "pt_cat_idx": bcls, "ideal_w_kg": ideal, "delta_to_ideal": delta,
        "goal_3200_s": float(goal_3200_s), "goal_push": int(goal_push), "goal_sit": int(goal_sit),
        "knee": int(knee), "shoulder": int(shoulder), "back": int(back),
        "vegan": int(vegan), "lactose_free": int(lactose_free), "gluten_free": int(gluten_free)
    }])[FEATURE_COLS].astype("float32").values

    Xz = PL_XSC.transform(feats)
    Yz = PLANNER.predict(Xz, verbose=0)
    Y  = PL_YSC.inverse_transform(Yz)[0]

    target_keys = [
        "run_km_wk","runs_per_wk","intervals_per_wk","easy_runs_per_wk",
        "strength_per_wk","push_sets","sit_sets","kcal","protein_g","fat_g","carbs_g"
    ]
    out = {k: float(Y[i]) for i, k in enumerate(target_keys)}

    # 6) Armar plan y guardar
    plan = plan_from_outputs(out, bool(vegan), bool(lactose_free), bool(gluten_free))
    plan_id = create_plan(pred_id=pred_id, user_id=user_id, plan=plan)
    update_upload(upload_id, {"status": "planned", "planId": plan_id})

    # 7) Respuesta
    resultado = {
        "height_m": round(h_m, 3),
        "weight_kg": round(w_kg, 1),
        "class_idx": cls_idx,
        "class_name": class_name,
        "plan": plan,
        "uploadId": upload_id,
        "predId": pred_id,
        "planId": plan_id
    }
    return JSONResponse(content=json.loads(json.dumps(resultado, indent=2, ensure_ascii=False)))
