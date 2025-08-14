
import os, json, math
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from datetime import timedelta

# Load artifacts
ART_DIR = os.environ.get("ART_DIR", ".")

with open(os.path.join(ART_DIR, "planner_columns.json"), "r", encoding="utf-8") as f:
    cols = json.load(f)
INPUT_COLS = cols["input_cols"]
TARGET_COLS = cols["target_cols"]

scaler_x = joblib.load(os.path.join(ART_DIR, "scaler_inputs.pkl"))
scaler_y = joblib.load(os.path.join(ART_DIR, "scaler_targets.pkl"))
model = tf.keras.models.load_model(os.path.join(ART_DIR, "planner_model.keras"))

# Minimal catalogs (expand later or point to your CSVs)
EXERCISES_CSV = os.environ.get("EXERCISES_CSV", "exercises_min.csv")
RECIPES_CSV   = os.environ.get("RECIPES_CSV",   "recipes_min.csv")

if not os.path.isfile(EXERCISES_CSV):
    # Fallback tiny DataFrame
    exercises = pd.DataFrame([
        # id, name, group, equipment, contraindications (pipe-separated)
        [1,"Sentadilla goblet","piernas","dumbbell","rodilla|espalda"],
        [2,"Zancadas caminando","piernas","none","rodilla"],
        [3,"Puente de glúteo","piernas","none","espalda"],
        [4,"Flexiones de brazos","pecho","none","hombro|muñeca"],
        [5,"Remo con mancuernas","espalda","dumbbell","espalda"],
        [6,"Plancha","core","none","espalda"],
        [7,"Press banca mancuernas","pecho","dumbbell","hombro"],
        [8,"Press militar mancuernas","hombro","dumbbell","hombro"],
        [9,"Remo invertido","espalda","barra","hombro|codo"],
        [10,"Saltos de cuerda","cardio","rope","rodilla|tobillo"]
    ], columns=["id","name","group","equipment","contra"])
else:
    exercises = pd.read_csv(EXERCISES_CSV)

if not os.path.isfile(RECIPES_CSV):
    recipes = pd.DataFrame([
        # id, title, kcal, protein_g, fat_g, carbs_g, tags (comma)
        [1,"Avena con yogurt y frutas",450,20,12,65,"lactosa"],
        [2,"Tofu salteado con verduras + arroz",650,35,18,85,"vegano|sin_lactosa"],
        [3,"Pollo a la plancha + quinoa + ensalada",700,55,20,60,"sin_gluten|sin_lactosa"],
        [4,"Ensalada de garbanzos",550,25,18,70,"vegano|sin_lactosa|sin_gluten"],
        [5,"Tortilla de claras + pan integral + palta",520,35,18,50,"lactosa_free"],
        [6,"Salmón + camote + brócoli",720,45,30,55,"sin_gluten|sin_lactosa"],
    ], columns=["id","title","kcal","protein_g","fat_g","carbs_g","tags"])
else:
    recipes = pd.read_csv(RECIPES_CSV)

def seconds_to_hhmm(x):
    m = int(x//60); s = int(x%60)
    return f"{m:02d}:{s:02d}"

def plan_from_outputs(y, constraints):
    """
    y: vector (12) in real scale
    constraints: dict with dietary and injury constraints
    Returns structured weekly plan.
    """
    (kcal, protein, fat, carbs,
     run_easy, run_tempo, run_interval,
     up_days, low_days, pushups, situps, long_run) = y.tolist()

    # --- Diet: naive 3 meals close to targets ---
    meals = []
    allowed = recipes.copy()
    if constraints.get("vegan", False):
        allowed = allowed[allowed["tags"].str.contains("vegano", na=False)]
    if constraints.get("gluten_free", False):
        allowed = allowed[~allowed["tags"].str.contains("gluten", na=False)]
    if constraints.get("lactose_free", False):
        allowed = allowed[~allowed["tags"].str.contains("lactosa", na=False)]
    if len(allowed)==0: allowed = recipes

    kcal_per_meal = kcal/3.0
    # Pick 3 meals nearest kcal_per_meal
    allowed["diff"] = (allowed["kcal"]-kcal_per_meal).abs()
    allowed = allowed.sort_values("diff")
    pick = allowed.head(3).to_dict(orient="records")
    for p in pick:
        meals.append({
            "title": p["title"],
            "kcal": float(p["kcal"]), "protein_g": float(p["protein_g"]),
            "fat_g": float(p["fat_g"]), "carbs_g": float(p["carbs_g"]),
        })

    # --- Training split (7 days) ---
    inj_shoulder = constraints.get("inj_shoulder", False)
    inj_knee     = constraints.get("inj_knee", False)
    inj_back     = constraints.get("inj_back", False)

    def filter_ex(group, avoid_tags):
        df = exercises[exercises["group"]==group]
        if len(df)==0: return []
        if avoid_tags:
            df = df[~df["contra"].fillna("").str.contains("|".join(avoid_tags))]
        return df["name"].head(5).tolist()

    avoid = []
    if inj_shoulder: avoid.append("hombro")
    if inj_knee:     avoid.append("rodilla")
    if inj_back:     avoid.append("espalda")

    upper_list = filter_ex("pecho", avoid) + filter_ex("espalda", avoid) + filter_ex("hombro", avoid)
    lower_list = filter_ex("piernas", avoid)
    core_list  = filter_ex("core", avoid)

    # Construct schedule
    days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    sched = []
    # Distribute running minutes
    run_blocks = [
        ("Easy run", run_easy, 3),
        ("Tempo run", run_tempo, 2),
        ("Intervals", run_interval, 2),
        ("Long run", long_run, 1),
    ]

    def split_minutes(total, parts):
        if total<=0 or parts<=0: return [0]*parts
        base = int(total//parts)
        rem = int(total - base*parts)
        arr = [base]*parts
        for i in range(rem): arr[i]+=1
        return arr

    runs_assigned = []
    for name, mins, parts in run_blocks:
        arr = split_minutes(int(round(mins)), parts)
        for m in arr:
            if m>0: runs_assigned.append((name, m))

    # Strength sessions
    up_days = int(round(up_days)); low_days = int(round(low_days))
    pushups = int(round(pushups)); situps = int(round(situps))

    # Simple weekly distribution
    i = 0
    for d in days:
        day_plan = {"day": d, "sessions": []}
        # Assign a run if pending
        if i < len(runs_assigned):
            rname, m = runs_assigned[i]; i+=1
            day_plan["sessions"].append({"type":"run", "name": rname, "minutes": int(m)})
        # Alternate upper/lower
        if up_days>0:
            day_plan["sessions"].append({"type":"strength","focus":"upper","exercises": upper_list[:3] or ["Flexiones"], "pushups_target": pushups})
            up_days -= 1
        elif low_days>0:
            day_plan["sessions"].append({"type":"strength","focus":"lower","exercises": lower_list[:3] or ["Sentadilla goblet"]})
            low_days -= 1
        else:
            # core maintenance
            if len(core_list)>0:
                day_plan["sessions"].append({"type":"core","exercises": core_list[:2], "situps_target": situps})
        sched.append(day_plan)

    return {
        "nutrition": {
            "targets_per_day": {
                "kcal": round(float(kcal)),
                "protein_g": round(float(protein)),
                "fat_g": round(float(fat)),
                "carbs_g": round(float(carbs)),
            },
            "meals_example": meals
        },
        "training": sched
    }

def predict_plan(
    sex:str, height_cm:float, weight_kg:float,
    grade_run:int, grade_push:int, grade_abs:int,
    vegan=False, gluten_free=False, lactose_free=False, nut_allergy=False,
    inj_shoulder=False, inj_knee=False, inj_back=False,
    equip_barbell=False, equip_dumbbell=True, equip_machines=False, equip_track=True
):
    """Returns weekly plan dict."""
    bmi = weight_kg / ((height_cm/100.0)**2)
    row = [
        0 if sex=="damas" else 1,
        height_cm, weight_kg, bmi,
        grade_run, grade_push, grade_abs,
        int(vegan), int(gluten_free), int(lactose_free), int(nut_allergy),
        int(inj_shoulder), int(inj_knee), int(inj_back),
        int(equip_barbell), int(equip_dumbbell), int(equip_machines), int(equip_track)
    ]
    X = np.array(row, dtype=np.float32)[None, :]
    Xs = scaler_x.transform(X)
    Ys = model.predict(Xs, verbose=0)
    Y = scaler_y.inverse_transform(Ys)[0]
    constraints = {
        "vegan": vegan, "gluten_free": gluten_free, "lactose_free": lactose_free, "nut_allergy": nut_allergy,
        "inj_shoulder": inj_shoulder, "inj_knee": inj_knee, "inj_back": inj_back
    }
    plan = plan_from_outputs(Y, constraints)
    return plan

if __name__ == "__main__":
    # Demo
    plan = predict_plan(
        sex="varones", height_cm=175, weight_kg=82,
        grade_run=4, grade_push=3, grade_abs=3,
        lactose_free=True, inj_knee=False, vegan=False
    )
    import json
    print(json.dumps(plan, ensure_ascii=False, indent=2))
