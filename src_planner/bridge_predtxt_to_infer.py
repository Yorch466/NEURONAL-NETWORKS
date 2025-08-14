# bridge_predtxt_to_planner_grades.py
import os, re, json, argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

# --------- Parsers ----------
LINE_RE = re.compile(
    r"Imagen:\s*(?P<fname>.+?)\s*\|\s*Altura:\s*(?P<h>[0-9.]+)\s*m.*?\|\s*Peso:\s*(?P<w>[0-9.]+)\s*kg",
    re.IGNORECASE
)

def parse_time_to_seconds(s: str) -> float:
    s = str(s).strip()
    if ":" not in s:
        return float(s)
    mm, ss = s.split(":")
    return int(mm)*60 + float(ss)

def guess_sex_from_name(name: str, default_sex: int):
    n = name.lower()
    if any(k in n for k in ["dama","female","f_","_f","mujer","femenino","girl","woman"]):
        return 0  # damas
    if any(k in n for k in ["varon","male","m_","_m","hombre","masculino","boy","man"]):
        return 1  # varones
    return default_sex

def bmi_class(bmi):
    if bmi < 18.5: return 0
    if bmi < 20.0: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

# --------- Grade mapping from goals (uses your CSVs) ----------
def load_grade_tables(folder="."):
    paths = {
        ("damas","run"): os.path.join(folder,"damas_aerobica_3200m.csv"),
        ("varones","run"): os.path.join(folder,"varones_aerobica_3200m.csv"),
        ("damas","push"): os.path.join(folder,"damas_flexiones_2min.csv"),
        ("varones","push"): os.path.join(folder,"varones_flexiones_2min.csv"),
        ("damas","sit"): os.path.join(folder,"damas_abdominales_2min.csv"),
        ("varones","sit"): os.path.join(folder,"varones_abdominales_2min.csv"),
    }
    dfs = {}
    for k,p in paths.items():
        if os.path.isfile(p):
            df = pd.read_csv(p)
            # Normalizamos nombres de columnas comunes
            cols = {c.lower():c for c in df.columns}
            # intentamos detectar 'time' o 'time_s' (aeróbica), y 'reps' (fuerza) y 'grade'
            if "time_s" in cols:
                df["time_s_norm"] = df[cols["time_s"]].astype(float)
            elif "time" in cols:
                # convertir MM:SS a s
                df["time_s_norm"] = df[cols["time"]].astype(str).apply(parse_time_to_seconds)
            if "reps" in cols:
                df["reps_norm"] = pd.to_numeric(df[cols["reps"]], errors="coerce")
            if "grade" in cols:
                df["grade_norm"] = pd.to_numeric(df[cols["grade"]], errors="coerce")
            dfs[k] = df
    return dfs

def goal_to_grade(sex_int:int, goal_3200=None, goal_push=None, goal_sit=None, tables=None):
    """Maps goals to grade using CSVs. Falls back to sensible defaults if missing."""
    sex_name = "varones" if sex_int==1 else "damas"
    grade_run = grade_push = grade_sit = 3  # defaults

    if tables is None:
        tables = load_grade_tables(".")

    # run
    if goal_3200 is not None and ("{}","run"):
        df = tables.get((sex_name,"run"))
        if df is not None and "time_s_norm" in df and "grade_norm" in df:
            t = parse_time_to_seconds(goal_3200)
            # tomamos el grade de la fila con tiempo más cercano
            ix = (df["time_s_norm"] - t).abs().idxmin()
            gv = df.loc[ix,"grade_norm"]
            if pd.notna(gv):
                grade_run = int(gv)

    # push
    if goal_push is not None:
        df = tables.get((sex_name,"push"))
        if df is not None and "reps_norm" in df and "grade_norm" in df:
            reps = float(goal_push)
            ix = (df["reps_norm"] - reps).abs().idxmin()
            gv = df.loc[ix,"grade_norm"]
            if pd.notna(gv):
                grade_push = int(gv)

    # sit
    if goal_sit is not None:
        df = tables.get((sex_name,"sit"))
        if df is not None and "reps_norm" in df and "grade_norm" in df:
            reps = float(goal_sit)
            ix = (df["reps_norm"] - reps).abs().idxmin()
            gv = df.loc[ix,"grade_norm"]
            if pd.notna(gv):
                grade_abs = int(gv)
                return grade_run, grade_push, grade_abs

    # si no hubo tabla para situps:
    return grade_run, grade_push, int(goal_sit if goal_sit is not None else grade_push)

# --------- Planner artifacts & plan builder (same logic as tu script) ----------
def load_artifacts(art_dir):
    with open(os.path.join(art_dir,"planner_columns.json"),"r",encoding="utf-8") as f:
        cols = json.load(f)
    input_cols  = cols["input_cols"]
    target_cols = cols["target_cols"]
    xsc = joblib.load(os.path.join(art_dir,"scaler_inputs.pkl"))
    ysc = joblib.load(os.path.join(art_dir,"scaler_targets.pkl"))
    model = tf.keras.models.load_model(os.path.join(art_dir,"planner_model.keras"))
    return input_cols, target_cols, xsc, ysc, model

def simple_catalogs(ex_csv=None, rec_csv=None):
    if ex_csv and os.path.isfile(ex_csv):
        exercises = pd.read_csv(ex_csv)
    else:
        exercises = pd.DataFrame([
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
    if rec_csv and os.path.isfile(rec_csv):
        recipes = pd.read_csv(rec_csv)
    else:
        recipes = pd.DataFrame([
            [1,"Avena con yogurt y frutas",450,20,12,65,"lactosa"],
            [2,"Tofu salteado con verduras + arroz",650,35,18,85,"vegano|sin_lactosa"],
            [3,"Pollo a la plancha + quinoa + ensalada",700,55,20,60,"sin_gluten|sin_lactosa"],
            [4,"Ensalada de garbanzos",550,25,18,70,"vegano|sin_lactosa|sin_gluten"],
            [5,"Tortilla de claras + pan integral + palta",520,35,18,50,"lactosa_free"],
            [6,"Salmón + camote + brócoli",720,45,30,55,"sin_gluten|sin_lactosa"],
        ], columns=["id","title","kcal","protein_g","fat_g","carbs_g","tags"])
    return exercises, recipes

def build_plan_from_outputs(Y, constraints, exercises, recipes):
    (kcal, protein, fat, carbs,
     run_easy, run_tempo, run_interval,
     up_days, low_days, pushups, situps, long_run) = Y.tolist()

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
    allowed["diff"] = (allowed["kcal"]-kcal_per_meal).abs()
    for _,p in allowed.sort_values("diff").head(3).iterrows():
        meals.append({
            "title": p["title"],
            "kcal": float(p["kcal"]), "protein_g": float(p["protein_g"]),
            "fat_g": float(p["fat_g"]), "carbs_g": float(p["carbs_g"]),
        })

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

    days = ["Lun","Mar","Mié","Jue","Vie","Sáb","Dom"]
    sched = []
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

    up_days = int(round(up_days)); low_days = int(round(low_days))
    pushups = int(round(pushups)); situps = int(round(situps))

    i = 0
    for d in days:
        day_plan = {"day": d, "sessions": []}
        if i < len(runs_assigned):
            rname, m = runs_assigned[i]; i+=1
            day_plan["sessions"].append({"type":"run", "name": rname, "minutes": int(m)})
        if up_days>0:
            day_plan["sessions"].append({"type":"strength","focus":"upper",
                                         "exercises": upper_list[:3] or ["Flexiones"], "pushups_target": pushups})
            up_days -= 1
        elif low_days>0:
            day_plan["sessions"].append({"type":"strength","focus":"lower",
                                         "exercises": lower_list[:3] or ["Sentadilla goblet"]})
            low_days -= 1
        else:
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

# --------- Main ---------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_txt", required=True, help="predicciones_altura_peso.txt")
    ap.add_argument("--art_dir", default=".", help="Directorio con planner_model.keras y scalers")
    ap.add_argument("--outdir", default="plans_from_preds_grades")
    # Opción A: poner grades directos
    ap.add_argument("--grade_run", type=int)
    ap.add_argument("--grade_push", type=int)
    ap.add_argument("--grade_abs",  type=int)
    # Opción B: metas → grade (usa CSVs)
    ap.add_argument("--goal_3200", help='MM:SS o segundos (opcional)')
    ap.add_argument("--goal_push", type=float, help="reps en 2 min (opcional)")
    ap.add_argument("--goal_sit",  type=float, help="reps en 2 min (opcional)")
    ap.add_argument("--tables_dir", default=".", help="Carpeta de CSVs damas_/varones_*.csv")
    # sexo por defecto y restricciones globales
    ap.add_argument("--default_sex", type=int, default=1, help="1 varones, 0 damas")
    ap.add_argument("--vegan", action="store_true")
    ap.add_argument("--gluten_free", action="store_true")
    ap.add_argument("--lactose_free", action="store_true")
    ap.add_argument("--inj_knee", action="store_true")
    ap.add_argument("--inj_shoulder", action="store_true")
    ap.add_argument("--inj_back", action="store_true")
    # catálogos opcionales
    ap.add_argument("--exercises_csv")
    ap.add_argument("--recipes_csv")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    INPUT_COLS, TARGET_COLS, xsc, ysc, model = load_artifacts(args.art_dir)
    exercises, recipes = simple_catalogs(args.exercises_csv, args.recipes_csv)
    tables = load_grade_tables(args.tables_dir)

    with open(args.pred_txt, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    done = 0
    for ln in lines:
        m = LINE_RE.search(ln)
        if not m:  # intenta una forma más corta si existiese (altura/peso solos)
            continue
        fname = m.group("fname")
        h_m   = float(m.group("h"))
        w_kg  = float(m.group("w"))
        sex   = guess_sex_from_name(os.path.basename(fname), args.default_sex)

        # Grades
        if args.grade_run is not None and args.grade_push is not None and args.grade_abs is not None:
            g_run, g_push, g_abs = int(args.grade_run), int(args.grade_push), int(args.grade_abs)
        else:
            g_run, g_push, g_abs = goal_to_grade(
                sex, args.goal_3200, args.goal_push, args.goal_sit, tables
            )

        # Construir features desde nombres (según planner_columns.json)
        h_cm = h_m * 100.0
        bmi  = w_kg / (h_m**2)
        feats = {
            "sex": sex,
            "height_cm": h_cm,
            "weight_kg": w_kg,
            "bmi": bmi,
            "grade_run": g_run,
            "grade_push": g_push,
            "grade_abs": g_abs,
            "vegan": int(args.vegan),
            "gluten_free": int(args.gluten_free),
            "lactose_free": int(args.lactose_free),
            "nut_allergy": 0,
            "inj_shoulder": int(args.inj_shoulder),
            "inj_knee": int(args.inj_knee),
            "inj_back": int(args.inj_back),
            "equip_barbell": 0,
            "equip_dumbbell": 1,
            "equip_machines": 0,
            "equip_track": 1,
        }
        # Rellena faltantes con 0 si hiciera falta
        for k in INPUT_COLS:
            feats.setdefault(k, 0)

        X = np.array([[feats[k] for k in INPUT_COLS]], dtype="float32")
        Xz = xsc.transform(X)
        Yz = model.predict(Xz, verbose=0)
        Y  = ysc.inverse_transform(Yz).ravel()

        plan = build_plan_from_outputs(
            Y,
            constraints={
                "vegan": bool(args.vegan),
                "gluten_free": bool(args.gluten_free),
                "lactose_free": bool(args.lactose_free),
                "inj_shoulder": bool(args.inj_shoulder),
                "inj_knee": bool(args.inj_knee),
                "inj_back": bool(args.inj_back),
            },
            exercises=exercises, recipes=recipes
        )

        out = {
            "image": fname,
            "sex": int(sex),
            "height_m": float(h_m),
            "weight_kg": float(w_kg),
            "grades": {"run": int(g_run), "push": int(g_push), "abs": int(g_abs)},
            "constraints": {
                "vegan": bool(args.vegan), "gluten_free": bool(args.gluten_free),
                "lactose_free": bool(args.lactose_free),
                "inj_knee": bool(args.inj_knee), "inj_shoulder": bool(args.inj_shoulder), "inj_back": bool(args.inj_back)
            },
            "plan": plan
        }
        base = os.path.splitext(os.path.basename(fname))[0]
        out_path = os.path.join(args.outdir, f"plan_{base}.json")
        with open(out_path, "w", encoding="utf-8") as g:
            json.dump(out, g, ensure_ascii=False, indent=2)
        done += 1

    print(f"✅ Planes generados: {done} | Carpeta: {args.outdir}")

if __name__ == "__main__":
    main()
