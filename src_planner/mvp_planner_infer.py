# mvp_planner_infer.py
import os, json, argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# ===== Config =====
DEFAULT_ART_DIR = r"E:\TESIS\Redes neuronales\RED CONVOLUCIONAL\ckpts_planner"
DEFAULT_MEALS_PATH = os.environ.get("MEALS_CATALOG", "")  # opcional por env var

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

# ===== Utilidades =====
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
    return X, row  # row = constraints para comidas

# ====== Catálogo de comidas ======
def _fallback_meals():
    # Catálogo mínimo (boliviano) para contingencia
    # Estructura: category ∈ {breakfast,lunch,dinner,snack}
    # tags: ["vegano","sin_lactosa","sin_gluten"]
    return [
        {"title":"Api con pastel","category":"breakfast","kcal":500,"protein_g":8,"fat_g":10,"carbs_g":95,"tags":[]},
        {"title":"Huminta al horno","category":"breakfast","kcal":420,"protein_g":12,"fat_g":14,"carbs_g":60,"tags":[]},
        {"title":"Sándwich de palta","category":"snack","kcal":350,"protein_g":8,"fat_g":18,"carbs_g":35,"tags":["vegano"]},
        {"title":"Ensalada andina (quinua+habas)","category":"dinner","kcal":520,"protein_g":20,"fat_g":12,"carbs_g":80,"tags":["vegano","sin_lactosa","sin_gluten"]},
        {"title":"Trucha a la plancha con ensalada","category":"lunch","kcal":650,"protein_g":45,"fat_g":25,"carbs_g":45,"tags":["sin_gluten","sin_lactosa"]},
        {"title":"Charque de llama con mote","category":"lunch","kcal":700,"protein_g":55,"fat_g":18,"carbs_g":65,"tags":["sin_lactosa"]},
        {"title":"Sopa de quinua","category":"dinner","kcal":400,"protein_g":18,"fat_g":8,"carbs_g":60,"tags":["vegano","sin_lactosa"]},
        {"title":"Majadito cruceño","category":"lunch","kcal":740,"protein_g":35,"fat_g":20,"carbs_g":95,"tags":[]},
        {"title":"Ensalada de frutas con quinua inflada","category":"snack","kcal":300,"protein_g":6,"fat_g":6,"carbs_g":55,"tags":["vegano"]},
        {"title":"Pollo a la plancha con arroz integral","category":"dinner","kcal":620,"protein_g":45,"fat_g":15,"carbs_g":60,"tags":["sin_lactosa"]},
        {"title":"Choclo con queso","category":"snack","kcal":330,"protein_g":14,"fat_g":10,"carbs_g":45,"tags":[]},
        {"title":"Guiso de quinua con verduras","category":"lunch","kcal":600,"protein_g":22,"fat_g":14,"carbs_g":90,"tags":["vegano","sin_lactosa"]},
        {"title":"Sopa de maní (pollo)","category":"lunch","kcal":720,"protein_g":35,"fat_g":35,"carbs_g":65,"tags":[]},
        {"title":"Sonso de yuca con queso","category":"snack","kcal":380,"protein_g":14,"fat_g":16,"carbs_g":45,"tags":[]},
    ]

def load_meals_catalog(path: str):
    if not path or not os.path.isfile(path):
        return _fallback_meals()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Se admite lista o dict con key "meals"
    if isinstance(data, dict) and "meals" in data:
        return data["meals"]
    return data

def filter_meals(meals, vegan, lactose_free, gluten_free):
    def ok(m):
        tags = set(m.get("tags", []))
        if vegan and "vegano" not in tags: return False
        if lactose_free and "sin_lactosa" not in tags: return False
        if gluten_free and "sin_gluten" not in tags: return False
        return True
    allowed = [m for m in meals if ok(m)]
    return allowed if allowed else meals

# distribución de kcal por comida (puedes cambiar los pesos)
MEAL_SPLIT = {
    "breakfast": 0.25,
    "lunch":     0.40,
    "dinner":    0.30,
    "snack":     0.05,  # se puede omitir si kcal objetivo es baja
}

def choose_meals_for_day(meals, kcal_target, prot_t, fat_t, carb_t, constraints):
    # Filtrar por restricciones
    vegan = constraints.get("vegan", 0)
    lactose_free = constraints.get("lactose_free", 0)
    gluten_free  = constraints.get("gluten_free", 0)
    allowed = filter_meals(meals, vegan, lactose_free, gluten_free)

    # Separar por categoría
    by_cat = {"breakfast":[], "lunch":[], "dinner":[], "snack":[]}
    for m in allowed:
        cat = m.get("category", "lunch")
        if cat not in by_cat: continue
        by_cat[cat].append(m)

    def pick_for(cat, share):
        # Objetivo por comida
        kc = kcal_target * share
        # Greedy por cercanía en kcal (y ligera penalización por macros)
        cand = by_cat.get(cat, [])
        if not cand:
            # si no hay, usar cualquiera
            cand = allowed
        def score(m):
            dk = abs(m["kcal"] - kc)
            dp = abs(m["protein_g"] - prot_t*share)*0.5
            df = abs(m["fat_g"] - fat_t*share)*0.25
            dc = abs(m["carbs_g"] - carb_t*share)*0.25
            return dk + dp + df + dc
        cand = sorted(cand, key=score)
        return cand[0] if cand else None

    plan = {}
    # Si kcal diaria es baja < 2000, podemos omitir snack
    split = dict(MEAL_SPLIT)
    if kcal_target < 2000:
        snack_share = split.pop("snack")
        # Redistribuir snack a lunch/dinner
        split["lunch"] += snack_share * 0.6
        split["dinner"] += snack_share * 0.4

    for cat, share in split.items():
        chosen = pick_for(cat, share)
        if chosen:
            plan[cat] = chosen

    # Cálculo de totales estimados (suma de las comidas elegidas)
    totals = {"kcal":0,"protein_g":0,"fat_g":0,"carbs_g":0}
    for item in plan.values():
        totals["kcal"]      += item["kcal"]
        totals["protein_g"] += item["protein_g"]
        totals["fat_g"]     += item["fat_g"]
        totals["carbs_g"]   += item["carbs_g"]

    return plan, totals

# ====== Entrenamiento semanal (igual que antes, sólo empaquetado) ======
EXERCISES = {
    "upper": ["Flexiones","Remo mancuerna","Press militar","Remo invertido"],
    "lower": ["Sentadilla goblet","Zancadas","Puente de glúteo","Peso muerto rumano"],
    "core":  ["Plancha","Dead bug","Bird-dog","Crunch"]
}

def split_minutes(total, parts):
    if total<=0 or parts<=0: return [0]*parts
    base = int(total//parts); rem = int(total - base*parts)
    out = [base]*parts
    for i in range(rem): out[i]+=1
    return out

def plan_training_from_outputs(out):
    runs_per_wk = max(0, out["runs_per_wk"])
    easy = max(0, out["easy_runs_per_wk"])
    intervals = max(0, out["intervals_per_wk"])
    long_run_min = max(30, int(round(0.25*60)))  # placeholder

    blocks = []
    if easy>0:      blocks += [("Easy run", m) for m in split_minutes(int(round(45*easy)), max(1,int(round(easy)))) ]
    if intervals>0: blocks += [("Intervals", m) for m in split_minutes(int(round(20*intervals)), max(1,int(round(intervals)))) ]
    blocks += [("Long run", long_run_min)]

    days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    sched = []
    up_left  = int(round(out["strength_per_wk"]/2))
    low_left = int(round(out["strength_per_wk"] - up_left))
    push_sets = int(round(out["push_sets"]))
    sit_sets  = int(round(out["sit_sets"]))

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
    return sched

# ===== Main infer =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art_dir", default=DEFAULT_ART_DIR)
    ap.add_argument("--meals_catalog", default=DEFAULT_MEALS_PATH, help="Ruta JSON con catálogo de platos bolivianos")
    ap.add_argument("--sex", default="varones", choices=["varones","damas","0","1","M","F","male","female"])
    ap.add_argument("--height_cm", type=float, required=True)
    ap.add_argument("--weight_kg", type=float, required=True)
    ap.add_argument("--goal_3200_s", type=float, default=18*60.0)
    ap.add_argument("--goal_push", type=int, default=45)
    ap.add_argument("--goal_sit", type=int, default=50)
    ap.add_argument("--knee", type=int, default=0)
    ap.add_argument("--shoulder", type=int, default=0)
    ap.add_argument("--back", type=int, default=0)
    ap.add_argument("--vegan", type=int, default=0)
    ap.add_argument("--lactose_free", type=int, default=0)
    ap.add_argument("--gluten_free", type=int, default=0)
    args = ap.parse_args()

    # Artefactos del modelo
    x_path = os.path.join(args.art_dir, "planner_x_scaler.pkl")
    y_path = os.path.join(args.art_dir, "planner_y_scaler.pkl")
    best_path = os.path.join(args.art_dir, "planner_best.keras")
    mlp_path  = os.path.join(args.art_dir, "planner_mlp.keras")
    if not os.path.isfile(x_path) or not os.path.isfile(y_path):
        raise FileNotFoundError("Faltan planner_x_scaler.pkl / planner_y_scaler.pkl en " + args.art_dir)
    model_path = best_path if os.path.isfile(best_path) else mlp_path
    if not os.path.isfile(model_path):
        raise FileNotFoundError("Faltan planner_best.keras y planner_mlp.keras en " + args.art_dir)

    # Cargar
    xsc = joblib.load(x_path)
    ysc = joblib.load(y_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    # Features
    X, raw = build_features(
        sex=args.sex, height_cm=args.height_cm, weight_kg=args.weight_kg,
        goal_3200_s=args.goal_3200_s, goal_push=args.goal_push, goal_sit=args.goal_sit,
        knee=args.knee, shoulder=args.shoulder, back=args.back,
        vegan=args.vegan, lactose_free=args.lactose_free, gluten_free=args.gluten_free
    )

    # Infer numérica
    Xz = xsc.transform(X)
    Yz = model.predict(Xz, verbose=0)
    Y  = ysc.inverse_transform(Yz)[0]
    out = {k: float(Y[i]) for i, k in enumerate(TARGET_COLS)}

    # redondeos suaves
    for k in ["runs_per_wk","intervals_per_wk","easy_runs_per_wk","strength_per_wk","push_sets","sit_sets"]:
        out[k] = round(out[k], 1)
    out["run_km_wk"] = round(out["run_km_wk"], 1)
    for k in ["kcal","protein_g","fat_g","carbs_g"]:
        out[k] = round(out[k], 0)

    # Entrenamiento semanal
    training_sched = plan_training_from_outputs(out)

    # Selección de comidas
    meals_catalog = load_meals_catalog(args.meals_catalog)
    constraints = {"vegan": args.vegan, "lactose_free": args.lactose_free, "gluten_free": args.gluten_free}
    meals_plan, meals_totals = choose_meals_for_day(
        meals_catalog,
        kcal_target=out["kcal"],
        prot_t=out["protein_g"], fat_t=out["fat_g"], carb_t=out["carbs_g"],
        constraints=constraints
    )

    result = {
        "nutrition": {
            "targets_per_day": {
                "kcal": int(out["kcal"]), "protein_g": int(out["protein_g"]),
                "fat_g": int(out["fat_g"]), "carbs_g": int(out["carbs_g"])
            },
            "meals": meals_plan,              # dict con breakfast/lunch/dinner(/snack)
            "meals_totals": meals_totals      # suma de macros de lo elegido
        },
        "training": training_sched,
        "predicted": out                     # deja también los targets numéricos por transparencia
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
