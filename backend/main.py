# main.py — pipeline End-to-End (CNN -> Planner -> Firestore)
import io, os, json, datetime
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from PIL import Image
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field
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
# RUTAS 
# =========================
BASE_DIR = r"E:\TESIS\Redes neuronales\RED CONVOLUCIONAL"

# Red 1 (CNN antropométrica)
MODEL_PATH  = fr"{BASE_DIR}\modelo_antropometrico.keras"
SCALER_PATH = fr"{BASE_DIR}\scaler_medidas.pkl"

# Red 2 (Planner MLP)
PL_DIR          = fr"{BASE_DIR}\ckpts_planner"
PL_MODEL_PATH   = fr"{PL_DIR}\planner_best.keras"      # o planner_mlp.keras
PL_XSC_PATH     = fr"{PL_DIR}\planner_x_scaler.pkl"
PL_YSC_PATH     = fr"{PL_DIR}\planner_y_scaler.pkl"

# Salidas locales en disco
OUTPUT_DIR      = fr"{BASE_DIR}\outputs\plans"         # <<-- aquí se guardarán los .json

IMG_SIZE    = (224, 224)
CLASS_NAMES = ["desnutricion", "riesgo", "normal", "sobrepeso", "obesidad"]  # head CNN (si existe)

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
# Clasificación UI (fallback IMC) y mapeos
# =========================
UI_CLASS_NAMES = ["desnutricion", "bajo_peso", "normal", "sobrepeso", "obesidad"]
CNN_TO_UI = {
    "desnutricion": "desnutricion",
    "riesgo":       "bajo_peso",
    "normal":       "normal",
    "sobrepeso":    "sobrepeso",
    "obesidad":     "obesidad",
}

def bmi_ui_idx(bmi: float) -> int:
    # UI: <17 desnutrición; [17,18.5) bajo_peso; [18.5,25) normal; [25,30) sobrepeso; >=30 obesidad
    if bmi < 17.0: return 0
    if bmi < 18.5: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

# === Importante ===
# Para el PLANNER mantén la codificación con umbrales de entrenamiento: 18.5/20/25/30
def bmi_class_planner(b):
    return 0 if b < 18.5 else 1 if b < 20 else 2 if b < 25 else 3 if b < 30 else 4

def ui_class_from_outputs(h_m: float, w_kg: float, cls_name_from_cnn: Optional[str]):
    bmi = w_kg / (h_m * h_m)
    if cls_name_from_cnn:
        ui_name = CNN_TO_UI.get(cls_name_from_cnn, cls_name_from_cnn)
        ui_idx  = UI_CLASS_NAMES.index(ui_name)
        source  = "cnn"
    else:
        ui_idx  = bmi_ui_idx(bmi)
        ui_name = UI_CLASS_NAMES[ui_idx]
        source  = "bmi"
    return ui_idx, ui_name, source, bmi

# =========================
# Utilidades Planner
# =========================
FEATURE_COLS = [
    "sex","h_m","w_kg","bmi","bmi_cls","pt_cat_idx",
    "ideal_w_kg","delta_to_ideal",
    "goal_3200_s","goal_push","goal_sit",
    "knee","shoulder","back","vegan","lactose_free","gluten_free"
]

EXERCISES = {
    "upper": ["Flexiones", "Remo mancuerna", "Press militar", "Remo invertido"],
    "lower": ["Sentadilla goblet", "Zancadas", "Puente de glúteo", "Peso muerto rumano"],
    "core" : ["Plancha", "Dead bug", "Bird-dog", "Crunch"]
}

def _fallback_meals():
    # Catálogo mínimo por si falta el JSON (no te quedas sin comidas)
    return [
        {"title":"Api con pastel","category":"breakfast","kcal":500,"protein_g":8,"fat_g":10,"carbs_g":95,"tags":[]},
        {"title":"Huminta al horno","category":"breakfast","kcal":420,"protein_g":12,"fat_g":14,"carbs_g":60,"tags":[]},
        {"title":"Trucha a la plancha con ensalada","category":"lunch","kcal":650,"protein_g":45,"fat_g":25,"carbs_g":45,"tags":["sin_gluten","sin_lactosa"]},
        {"title":"Charque de llama con mote","category":"lunch","kcal":700,"protein_g":55,"fat_g":18,"carbs_g":65,"tags":["sin_lactosa"]},
        {"title":"Guiso de quinua con verduras","category":"lunch","kcal":600,"protein_g":22,"fat_g":14,"carbs_g":90,"tags":["vegano","sin_lactosa"]},
        {"title":"Pollo a la plancha con arroz integral","category":"dinner","kcal":620,"protein_g":45,"fat_g":15,"carbs_g":60,"tags":["sin_lactosa"]},
        {"title":"Sopa de quinua","category":"dinner","kcal":400,"protein_g":18,"fat_g":8,"carbs_g":60,"tags":["vegano","sin_lactosa"]},
        {"title":"Sándwich de palta","category":"snack","kcal":350,"protein_g":8,"fat_g":18,"carbs_g":35,"tags":["vegano"]},
        {"title":"Ensalada de frutas con quinua inflada","category":"snack","kcal":300,"protein_g":6,"fat_g":6,"carbs_g":55,"tags":["vegano"]},
    ]

def load_meals_catalog(path: str):
    if not os.path.isfile(path):
        return _fallback_meals()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["meals"] if isinstance(data, dict) and "meals" in data else data
    except Exception:
        return _fallback_meals()

def _filter_meals(meals, vegan: int, lact_free: int, glu_free: int):
    def ok(m):
        tags = set(m.get("tags", []))
        if vegan and "vegano" not in tags: return False
        if lact_free and "sin_lactosa" not in tags: return False
        if glu_free and "sin_gluten" not in tags: return False
        return True
    out = [m for m in meals if ok(m)]
    return out if out else meals

def choose_meals_for_day(meals, kcal_target, prot_t, fat_t, carb_t, vegan, lact_free, glu_free):
    meals = _filter_meals(meals, vegan, lact_free, glu_free)
    by_cat = {"breakfast":[], "lunch":[], "dinner":[], "snack":[]}
    for m in meals:
        by_cat.setdefault(m.get("category","lunch"), []).append(m)

    split = dict(MEAL_SPLIT)
    if kcal_target < 2000 and "snack" in split:
        snack = split.pop("snack")
        split["lunch"]  = split.get("lunch",0)  + snack*0.6
        split["dinner"] = split.get("dinner",0) + snack*0.4

    def score(m, share):
        dk = abs(m["kcal"] - kcal_target*share)
        dp = abs(m["protein_g"] - prot_t*share)*0.5
        df = abs(m["fat_g"]     - fat_t*share)*0.25
        dc = abs(m["carbs_g"]   - carb_t*share)*0.25
        return dk + dp + df + dc

    plan = {}
    for cat, share in split.items():
        cand = by_cat.get(cat) or meals
        cand.sort(key=lambda m: score(m, share))
        if cand: plan[cat] = cand[0]

    totals = {"kcal":0,"protein_g":0,"fat_g":0,"carbs_g":0}
    for item in plan.values():
        totals["kcal"]      += int(round(item["kcal"]))
        totals["protein_g"] += int(round(item["protein_g"]))
        totals["fat_g"]     += int(round(item["fat_g"]))
        totals["carbs_g"]   += int(round(item["carbs_g"]))
    return plan, totals


def split_minutes(total, parts):
    if total <= 0 or parts <= 0: return [0]*parts
    base = int(total // parts); rem = int(total - base*parts)
    arr = [base] * parts
    for i in range(rem): arr[i] += 1
    return arr

def plan_from_outputs(out, vegan, lact_free, glu_free):
    kcal     = float(out["kcal"])
    protein  = float(out["protein_g"])
    fat      = float(out["fat_g"])
    carbs    = float(out["carbs_g"])
    easy     = max(0, out["easy_runs_per_wk"])
    inter    = max(0, out["intervals_per_wk"])
    strength = max(0, out["strength_per_wk"])
    push_sets = int(round(out["push_sets"]))
    sit_sets  = int(round(out["sit_sets"]))

    # --- Running/Fuerza (igual que tenías)
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

    # --- Selección de comidas desde catálogo
    meals_catalog = load_meals_catalog(MEALS_CATALOG_PATH)
    meals_plan, meals_totals = choose_meals_for_day(
        meals_catalog,
        kcal_target=kcal,
        prot_t=protein,
        fat_t=fat,
        carb_t=carbs,
        vegan=int(vegan), lact_free=int(lact_free), glu_free=int(glu_free)
    )

    return {
        "nutrition": {
            "targets_per_day": {
                "kcal": int(round(kcal)),
                "protein_g": int(round(protein)),
                "fat_g": int(round(fat)),
                "carbs_g": int(round(carbs)),
            },
            "meals": meals_plan,           # breakfast/lunch/dinner(/snack)
            "meals_totals": meals_totals   # suma de macros de lo elegido
        },
        "training": sched
    }

def pretty_json(d: dict) -> JSONResponse:
    return JSONResponse(content=json.loads(json.dumps(d, indent=2, ensure_ascii=False)))

def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_json_local(prefix: str, data: dict) -> str:
    """Guarda un JSON en OUTPUT_DIR con timestamp; devuelve ruta."""
    ensure_dir(OUTPUT_DIR)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(OUTPUT_DIR, f"{prefix}_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

# =========================
# Cargar modelos al iniciar
# =========================
app = FastAPI(title="Anthro + Planner", version="1.1")

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

# Robust predictor (CNN puede devolver solo regresión o multitarea)
def predict_anthro_from_image(img_bytes: bytes, sex: int):
    arr = load_image_bytes(img_bytes)        # (224,224,3)
    X = np.expand_dims(arr, 0)
    S = np.array([[sex]], dtype="float32")   # (1,1)
    out = CNN.predict([X, S], verbose=0)
    if isinstance(out, (list, tuple)) and len(out) == 2:
        reg_z, cls_prob = out
        cls_idx = int(cls_prob.argmax(axis=1)[0])
        cls_name_cnn = CLASS_NAMES[cls_idx]
    else:
        reg_z, cls_prob, cls_name_cnn = out, None, None
    h_m, w_kg = SCALER_Y.inverse_transform(reg_z)[0].tolist()
    return float(h_m), float(w_kg), cls_name_cnn

# =========================
# Modelos pydantic para /process/json
# =========================
class Goals(BaseModel):
    goal_3200_s: int
    goal_push: int
    goal_sit: int

class ManualAnthro(BaseModel):
    height_m: float
    weight_kg: float

class ProcessJSON(BaseModel):
    input_mode: Literal["manual","image"] = "manual"
    sex: int = Field(..., description="0=female,1=male")
    goals: Goals
    manual: Optional[ManualAnthro] = None
    user_id: str = "demo-user"
    knee: int = 0; shoulder: int = 0; back: int = 0
    vegan: int = 0; lactose_free: int = 0; gluten_free: int = 0

# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "planner": True, "cnn": True}

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
    # 1) Crear upload
    goals = {"run_s": goal_3200_s, "push": goal_push, "sit": goal_sit}
    upload_id = create_upload(
        user_id=user_id,
        image_path=file.filename,
        sex=sex,
        goals=goals,
        status="pending"
    )

    # 2) Leer imagen y predecir
    img_bytes = await file.read()
    h_m, w_kg, cls_name_cnn = predict_anthro_from_image(img_bytes, sex)

    # 3) Clasificación UI (CNN o fallback IMC)
    ui_idx, ui_name, class_source, bmi = ui_class_from_outputs(h_m, w_kg, cls_name_cnn)

    # 4) Guardar predicción
    pred_id = create_prediction(upload_id, h_m, w_kg, ui_idx, ui_name)
    update_upload(upload_id, {"status": "predicted", "predId": pred_id})

    # 5) Features EXACTAS y en ORDEN para el planner
    bcls_pl = bmi_class_planner(bmi)             # umbrales del entrenamiento (18.5/20/25/30)
    ideal   = 22.0 * (h_m * h_m)
    delta   = w_kg - ideal
    feats = pd.DataFrame([{
        "sex": 1.0 if sex >= 0.75 else 0.0,
        "h_m": h_m, "w_kg": w_kg, "bmi": bmi, "bmi_cls": bcls_pl,
        "pt_cat_idx": bcls_pl, "ideal_w_kg": ideal, "delta_to_ideal": delta,
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

    # 6) Armar plan, guardar en Firestore y en DISCO
    plan = plan_from_outputs(out, bool(vegan), bool(lactose_free), bool(gluten_free))
    plan_id = create_plan(pred_id=pred_id, user_id=user_id, plan=plan)
    update_upload(upload_id, {"status": "planned", "planId": plan_id})

    # Guardado local (solo el plan) en JSON aparte
    saved_path = save_json_local(prefix=f"plan_{upload_id}", data=plan)

    # 7) Respuesta
    resultado = {
        "height_m": round(h_m, 3),
        "weight_kg": round(w_kg, 1),
        "bmi": round(bmi, 2),
        "class_idx": ui_idx,
        "class_name": ui_name,
        "class_source": class_source,  # "cnn" o "bmi"
        "plan": plan,
        "plan_json_path": saved_path,
        "uploadId": upload_id,
        "predId": pred_id,
        "planId": plan_id
    }

    # Guardado local del JSON completo (resultado)
    saved_result_path = save_json_local(prefix=f"result_{upload_id}", data=resultado)
    resultado["result_json_path"] = saved_result_path

    return pretty_json(resultado)



@app.post("/process/json")
def process_json(payload: ProcessJSON):
    # Solo modo manual aquí
    if payload.input_mode != "manual":
        return JSONResponse(status_code=400, content={"detail": "input_mode debe ser 'manual' en /process/json"})
    if payload.manual is None:
        return JSONResponse(status_code=422, content={"detail": "manual {height_m, weight_kg} es requerido"})

    h_m = float(payload.manual.height_m)
    w_kg = float(payload.manual.weight_kg)
    ui_idx, ui_name, class_source, bmi = ui_class_from_outputs(h_m, w_kg, cls_name_from_cnn=None)

    # Guardar upload/pred (sin imagen)
    upload_id = create_upload(
        user_id=payload.user_id,
        image_path="manual",
        sex=payload.sex,
        goals=payload.goals.dict(),
        status="predicted"
    )
    pred_id = create_prediction(upload_id, h_m, w_kg, ui_idx, ui_name)

    # Planner (mantener bcls entrenado)
    bcls_pl = bmi_class_planner(bmi)
    ideal   = 22.0 * (h_m * h_m)
    delta   = w_kg - ideal
    feats = pd.DataFrame([{
        "sex": 1.0 if payload.sex >= 0.75 else 0.0,
        "h_m": h_m, "w_kg": w_kg, "bmi": bmi, "bmi_cls": bcls_pl,
        "pt_cat_idx": bcls_pl, "ideal_w_kg": ideal, "delta_to_ideal": delta,
        "goal_3200_s": float(payload.goals.goal_3200_s),
        "goal_push": int(payload.goals.goal_push),
        "goal_sit": int(payload.goals.goal_sit),
        "knee": int(payload.knee), "shoulder": int(payload.shoulder), "back": int(payload.back),
        "vegan": int(payload.vegan), "lactose_free": int(payload.lactose_free), "gluten_free": int(payload.gluten_free),
    }])[FEATURE_COLS].astype("float32").values

    Xz = PL_XSC.transform(feats)
    Yz = PLANNER.predict(Xz, verbose=0)
    Y  = PL_YSC.inverse_transform(Yz)[0]
    target_keys = [
        "run_km_wk","runs_per_wk","intervals_per_wk","easy_runs_per_wk",
        "strength_per_wk","push_sets","sit_sets","kcal","protein_g","fat_g","carbs_g"
    ]
    out = {k: float(Y[i]) for i, k in enumerate(target_keys)}

    plan = plan_from_outputs(out, bool(payload.vegan), bool(payload.lactose_free), bool(payload.gluten_free))
    plan_id = create_plan(pred_id=pred_id, user_id=payload.user_id, plan=plan)
    update_upload(upload_id, {"status": "planned", "planId": plan_id})

    # Guardado local (solo el plan) en JSON aparte
    saved_path = save_json_local(prefix=f"plan_{upload_id}", data=plan)

    resultado = {
        "height_m": round(h_m, 3),
        "weight_kg": round(w_kg, 1),
        "bmi": round(bmi, 2),
        "class_idx": ui_idx,
        "class_name": ui_name,
        "class_source": class_source,  # "bmi"
        "plan": plan,
        "plan_json_path": saved_path,
        "uploadId": upload_id,
        "predId": pred_id,
        "planId": plan_id
    }

    saved_result_path = save_json_local(prefix=f"result_{upload_id}", data=resultado)
    resultado["result_json_path"] = saved_result_path

    return pretty_json(resultado)

class GoalsAutoIn(BaseModel):
    sex: Literal["M","F"]
    ageYears: int = Field(..., ge=15, le=70)
    targetScore: int = Field(45, ge=60, le=100)  # 60..100

class GoalsAutoOut(BaseModel):
    goal_3200_s: int
    goal_push: int
    goal_sit: int
    meta: dict

# ===== Metas automáticas (Anexo A) =====
ANCHOR_SCORES = [60, 70, 80, 90, 100]

# Tabla ejemplo por sexo y rango etario.
# Reemplaza con tu Anexo A (valores por anclas 60/70/80/90/100).
GOALS_TABLE = {
    "M": {
        "18-24": {
            "push": [42, 50, 60, 70, 80],
            "sit":  [50, 60, 70, 80, 90],
            "run_s": [1100, 1000, 920, 880, 840],  # 3200 m (s)
        },
        "25-29": {
            "push": [40, 48, 58, 68, 78],
            "sit":  [48, 58, 68, 78, 88],
            "run_s": [1120, 1015, 935, 895, 855],
        },
        "30-34": {
            "push": [38, 46, 56, 66, 76],
            "sit":  [46, 56, 66, 76, 86],
            "run_s": [1140, 1030, 950, 910, 870],
        },
        "35-39": {
            "push": [36, 44, 54, 64, 74],
            "sit":  [44, 54, 64, 74, 84],
            "run_s": [1160, 1045, 965, 925, 885],
        },
        "40-44": {
            "push": [34, 42, 52, 62, 72],
            "sit":  [42, 52, 62, 72, 82],
            "run_s": [1180, 1060, 980, 940, 900],
        },
        "45-49": {
            "push": [30, 38, 48, 58, 68],
            "sit":  [38, 48, 58, 68, 78],
            "run_s": [1210, 1090, 1010, 970, 930],
        },
        "50-54": {
            "push": [26, 34, 44, 54, 64],
            "sit":  [34, 44, 54, 64, 74],
            "run_s": [1240, 1120, 1040, 1000, 960],
        },
        "55-59": {
            "push": [22, 30, 40, 50, 60],
            "sit":  [30, 40, 50, 60, 70],
            "run_s": [1280, 1150, 1070, 1030, 990],
        },
        "60+": {
            "push": [18, 26, 36, 46, 56],
            "sit":  [26, 36, 46, 56, 66],
            "run_s": [1320, 1180, 1100, 1060, 1020],
        },
    },
    "F": {
        "18-24": {
            "push": [18, 24, 32, 40, 48],
            "sit":  [40, 50, 60, 70, 80],
            "run_s": [1200, 1100, 1020, 980, 940],
        },
        "25-29": {
            "push": [16, 22, 30, 38, 46],
            "sit":  [38, 48, 58, 68, 78],
            "run_s": [1220, 1115, 1035, 995, 955],
        },
        "30-34": {
            "push": [14, 20, 28, 36, 44],
            "sit":  [36, 46, 56, 66, 76],
            "run_s": [1240, 1130, 1050, 1010, 970],
        },
        "35-39": {
            "push": [12, 18, 26, 34, 42],
            "sit":  [34, 44, 54, 64, 74],
            "run_s": [1260, 1145, 1065, 1025, 985],
        },
        "40-44": {
            "push": [10, 16, 24, 32, 40],
            "sit":  [32, 42, 52, 62, 72],
            "run_s": [1280, 1160, 1080, 1040, 1000],
        },
        "45-49": {
            "push": [8, 14, 22, 30, 38],
            "sit":  [30, 40, 50, 60, 70],
            "run_s": [1310, 1190, 1110, 1070, 1030],
        },
        "50-54": {
            "push": [6, 12, 20, 28, 36],
            "sit":  [28, 38, 48, 58, 68],
            "run_s": [1340, 1220, 1140, 1100, 1060],
        },
        "55-59": {
            "push": [4, 10, 18, 26, 34],
            "sit":  [26, 36, 46, 56, 66],
            "run_s": [1380, 1250, 1170, 1130, 1090],
        },
        "60+": {
            "push": [2, 8, 16, 24, 32],
            "sit":  [24, 34, 44, 54, 64],
            "run_s": [1420, 1280, 1200, 1160, 1120],
        },
    }
}

def _age_band(age: int) -> str:
    if age < 25: return "18-24"
    if age < 30: return "25-29"
    if age < 35: return "30-34"
    if age < 40: return "35-39"
    if age < 45: return "40-44"
    if age < 50: return "45-49"
    if age < 55: return "50-54"
    if age < 60: return "55-59"
    return "60+"

def _interp(x: float, xs: list, ys: list) -> float:
    """Interpolación lineal entre anclas xs (scores) y ys (valores)."""
    if x <= xs[0]: return float(ys[0])
    if x >= xs[-1]: return float(ys[-1])
    for i in range(len(xs)-1):
        if xs[i] <= x <= xs[i+1]:
            t = (x - xs[i]) / (xs[i+1] - xs[i])
            return float(ys[i] + t*(ys[i+1] - ys[i]))
    return float(ys[-1])

@app.post("/goals/auto", response_model=GoalsAutoOut)
def goals_auto(req: GoalsAutoIn):
    sex = req.sex
    age = req.ageYears
    score = req.targetScore

    if sex not in GOALS_TABLE:
      return JSONResponse(status_code=422, content={"detail": f"sex debe ser 'M' o 'F'"})

    band = _age_band(age)
    table = GOALS_TABLE[sex].get(band)
    if table is None:
      return JSONResponse(status_code=404, content={"detail": f"Sin tabla para edad {age} ({band})"})

    # Interpolar por puntaje
    push = int(round(_interp(score, ANCHOR_SCORES, table["push"])))
    sit  = int(round(_interp(score, ANCHOR_SCORES, table["sit"])))
    run_s = int(round(_interp(score, ANCHOR_SCORES, table["run_s"])))

    return GoalsAutoOut(
        goal_3200_s=run_s,
        goal_push=push,
        goal_sit=sit,
        meta={"sex": sex, "age_band": band, "score": score}
    )
