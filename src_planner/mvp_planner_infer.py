# mvp_planner_infer.py
import os, json, argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ====== Config ======
# Cambia este default si quieres dejar ruta fija:
DEFAULT_ART_DIR = r"E:\TESIS\Redes neuronales\RED CONVOLUCIONAL\ckpts_planner"

# --- catálogo mínimo para armar plan (puedes reemplazar por tus CSV) ---
EXERCISES = {
    "upper": ["Flexiones", "Remo mancuerna", "Press militar", "Remo invertido"],
    "lower": ["Sentadilla goblet", "Zancadas", "Puente de glúteo", "Peso muerto rumano"],
    "core":  ["Plancha", "Dead bug", "Bird-dog", "Crunch"]
}
RECIPES = [
    {"title":"Avena con yogurt y frutas", "kcal":450, "protein_g":20, "fat_g":12, "carbs_g":65, "tags":[]},
    {"title":"Tofu salteado + arroz",     "kcal":650, "protein_g":35, "fat_g":18, "carbs_g":85, "tags":["vegano","sin_lactosa"]},
    {"title":"Pollo + quinoa + ensalada", "kcal":700, "protein_g":55, "fat_g":20, "carbs_g":60, "tags":["sin_gluten","sin_lactosa"]},
    {"title":"Ensalada de garbanzos",     "kcal":550, "protein_g":25, "fat_g":18, "carbs_g":70, "tags":["vegano","sin_lactosa","sin_gluten"]},
    {"title":"Salmón + camote + brócoli", "kcal":720, "protein_g":45, "fat_g":30, "carbs_g":55, "tags":["sin_gluten","sin_lactosa"]},
]

def choose_meals(kcal_target, vegan=0, lactose_free=0, gluten_free=0):
    allowed = []
    for r in RECIPES:
        if vegan and "vegano" not in r["tags"]: continue
        if lactose_free and "sin_lactosa" not in r["tags"]: continue
        if gluten_free and "sin_gluten" not in r["tags"]: continue
        allowed.append(r)
    if not allowed: allowed = RECIPES
    per_meal = kcal_target/3.0
    allowed = sorted(allowed, key=lambda r: abs(r["kcal"]-per_meal))
    return allowed[:3]

def split_minutes(total, parts):
    if total<=0 or parts<=0: return [0]*parts
    base = int(total//parts); rem = int(total - base*parts)
    out = [base]*parts
    for i in range(rem): out[i]+=1
    return out

def plan_from_outputs(out, constraints=None):
    c = constraints or {}
    kcal = out["kcal"]; protein=out["protein_g"]; fat=out["fat_g"]; carbs=out["carbs_g"]
    runs_per_wk = max(0, out["runs_per_wk"])
    easy = max(0, out["easy_runs_per_wk"]); intervals = max(0, out["intervals_per_wk"])
    long_run_min = max(30, int(round(0.25*60)))  # placeholder si quieres largo fijo

    # Distribuir sesiones de carrera en la semana
    blocks = []
    if easy>0:      blocks += [("Easy run", m) for m in split_minutes(int(round(45*easy)), max(1,int(round(easy))))]
    if intervals>0: blocks += [("Intervals", m) for m in split_minutes(int(round(20*intervals)), max(1,int(round(intervals))))]
    blocks += [("Long run", long_run_min)]

    days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    sched = []
    up_left = int(round(out["strength_per_wk"]/2))
    low_left = int(round(out["strength_per_wk"] - up_left))
    push_sets = int(round(out["push_sets"])); sit_sets=int(round(out["sit_sets"]))
    i=0
    for d in days:
        day = {"day": d, "sessions": []}
        if i < len(blocks):
            name, mins = blocks[i]; i+=1
            if mins>0: day["sessions"].append({"type":"run","name":name,"minutes":mins})
        if up_left>0:
            day["sessions"].append({"type":"strength","focus":"upper","exercises":EXERCISES["upper"][:3],"pushups_sets":push_sets})
            up_left-=1
        elif low_left>0:
            day["sessions"].append({"type":"strength","focus":"lower","exercises":EXERCISES["lower"][:3]})
            low_left-=1
        else:
            day["sessions"].append({"type":"core","exercises":EXERCISES["core"][:2],"situps_sets":sit_sets})
        sched.append(day)

    meals = choose_meals(kcal, vegan=c.get("vegan",0), lactose_free=c.get("lactose_free",0), gluten_free=c.get("gluten_free",0))
    return {
        "nutrition": {
            "targets_per_day": {"kcal": int(round(kcal)), "protein_g": int(round(protein)),
                                "fat_g": int(round(fat)), "carbs_g": int(round(carbs))},
            "meals_example": meals
        },
        "training": sched
    }


FEATURE_COLS = [
    "sex","h_m","w_kg","bmi","bmi_cls","pt_cat_idx",
    "ideal_w_kg","delta_to_ideal",
    "goal_3200_s","goal_push","goal_sit",
    "knee","shoulder","back","vegan","lactose_free","gluten_free",
]

TARGET_COLS = [
    "run_km_wk","runs_per_wk","intervals_per_wk","easy_runs_per_wk",
    "strength_per_wk","push_sets","sit_sets","kcal","protein_g","fat_g","carbs_g",
]

# ====== Utils ======
def bmi_class(bmi: float) -> int:
    if bmi < 18.5: return 0
    if bmi < 20.0: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

def build_features(sex, height_cm, weight_kg,
                   goal_3200_s, goal_push, goal_sit,
                   knee, shoulder, back, vegan, lactose_free, gluten_free):
    h_m = float(height_cm) / 100.0
    w_kg = float(weight_kg)
    bmi = w_kg / (h_m**2)
    bcls = bmi_class(bmi)

    # Si no tienes tabla peso-talla en infer, usa defaults coherentes al train
    ideal_w_kg = 22.0 * (h_m**2)
    pt_cat_idx = bcls
    delta_to_ideal = w_kg - ideal_w_kg

    row = {
        "sex": 1.0 if (str(sex).lower() in ["1","varones","hombre","male","m"]) else 0.0,
        "h_m": h_m, "w_kg": w_kg, "bmi": bmi, "bmi_cls": bcls,
        "pt_cat_idx": pt_cat_idx, "ideal_w_kg": ideal_w_kg, "delta_to_ideal": delta_to_ideal,
        "goal_3200_s": float(goal_3200_s),
        "goal_push": int(goal_push),
        "goal_sit": int(goal_sit),
        "knee": int(knee), "shoulder": int(shoulder), "back": int(back),
        "vegan": int(vegan), "lactose_free": int(lactose_free), "gluten_free": int(gluten_free),
    }
    X = pd.DataFrame([row])[FEATURE_COLS].astype("float32").values
    return X

# ====== Main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", default=DEFAULT_ART_DIR, help="Carpeta con artefactos del planner")
    ap.add_argument("--sex", default="varones", choices=["varones","damas","0","1","M","F","male","female"])
    ap.add_argument("--height_cm", type=float, required=True)
    ap.add_argument("--weight_kg", type=float, required=True)
    ap.add_argument("--goal_3200_s", type=float, default=18*60.0)  # 3200m (segundos)
    ap.add_argument("--goal_push", type=int, default=45)
    ap.add_argument("--goal_sit", type=int, default=50)
    ap.add_argument("--knee", type=int, default=0)
    ap.add_argument("--shoulder", type=int, default=0)
    ap.add_argument("--back", type=int, default=0)
    ap.add_argument("--vegan", type=int, default=0)
    ap.add_argument("--lactose_free", type=int, default=0)
    ap.add_argument("--gluten_free", type=int, default=0)
    args = ap.parse_args()

    ART = args.art_dir

    # Cargar scalers y modelo (usa best si existe; si no, mlp)
    x_path = os.path.join(ART, "planner_x_scaler.pkl")
    y_path = os.path.join(ART, "planner_y_scaler.pkl")
    best_path = os.path.join(ART, "planner_best.keras")
    mlp_path  = os.path.join(ART, "planner_mlp.keras")

    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError("No encuentro planner_x_scaler.pkl / planner_y_scaler.pkl en " + ART)

    model_path = best_path if os.path.isfile(best_path) else mlp_path
    if not os.path.isfile(model_path):
        raise FileNotFoundError("No encuentro planner_best.keras ni planner_mlp.keras en " + ART)

    xsc = joblib.load(x_path)
    ysc = joblib.load(y_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Construir features
    X = build_features(
        sex=args.sex, height_cm=args.height_cm, weight_kg=args.weight_kg,
        goal_3200_s=args.goal_3200_s, goal_push=args.goal_push, goal_sit=args.goal_sit,
        knee=args.knee, shoulder=args.shoulder, back=args.back,
        vegan=args.vegan, lactose_free=args.lactose_free, gluten_free=args.gluten_free
    )

    # Inferencia
    Xz = xsc.transform(X)
    Yz = model.predict(Xz, verbose=0)
    Y  = ysc.inverse_transform(Yz)[0]

    out = {k: float(Y[i]) for i, k in enumerate(TARGET_COLS)}
    # redondeos suaves para mostrar
    for k in ["runs_per_wk","intervals_per_wk","easy_runs_per_wk","strength_per_wk","push_sets","sit_sets"]:
        out[k] = round(out[k], 1)
    for k in ["run_km_wk"]:
        out[k] = round(out[k], 1)
    for k in ["kcal","protein_g","fat_g","carbs_g"]:
        out[k] = round(out[k], 0)

    constraints = {"vegan": args.vegan, "lactose_free": args.lactose_free, "gluten_free": args.gluten_free}
    plan = plan_from_outputs(out, constraints)
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
