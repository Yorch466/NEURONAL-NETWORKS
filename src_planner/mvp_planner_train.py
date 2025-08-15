# mvp_planner_train.py
import os, json, math, random, argparse
import numpy as np, pandas as pd
from typing import List, Tuple, Dict
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, losses
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# ---------- utilidades ----------
def clip(a, lo, hi): return max(lo, min(float(a), hi))

def parse_time_to_seconds(s: str) -> float:
    try:
        mm, ss = str(s).split(":")
        return int(mm) * 60 + float(ss)
    except Exception:
        return np.nan

def bmi_class(bmi: float) -> int:
    if bmi < 18.5: return 0
    if bmi < 20.0: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

def mifflin_st_jeor(weight_kg, height_m, sex, age=25):
    h_cm = height_m * 100.0
    return (10*weight_kg + 6.25*h_cm - 5*age + (5 if sex==1 else -161))

def tdee_from_sessions(bmr, runs_per_wk, strength_per_wk):
    weekly_sesh = runs_per_wk + strength_per_wk
    fa = 1.35 + 0.02*clip(weekly_sesh,0,10) + 0.03*clip(runs_per_wk,0,6)
    return bmr * clip(fa, 1.35, 1.9)

# ---------- política (reglas -> etiquetas Y) ----------
def policy(head: dict) -> dict:
    sex = int(head["sex"])
    h_m = float(head["h_m"])
    w_kg = float(head["w_kg"])
    goal_3200_s = float(head["goal_3200_s"])
    goal_push   = float(head["goal_push"])
    goal_sit    = float(head["goal_sit"])
    knee   = int(head.get("knee",0))
    shoulder = int(head.get("shoulder",0))
    back   = int(head.get("back",0))
    vegan  = int(head.get("vegan",0))
    lactose_free = int(head.get("lactose_free",0))
    gluten_free  = int(head.get("gluten_free",0))

    bmi = w_kg/(h_m**2); cls = bmi_class(bmi)

    base_time = (15*60 if sex==1 else 16*60) + 16.0*(bmi-22.0)
    base_time = clip(base_time, 12*60, 28*60)
    base_push = (40 if sex==1 else 25) - 0.9*(bmi-22.0); base_push = clip(base_push, 10, 70)
    base_sit  = 40 - 0.5*(bmi-22.0); base_sit = clip(base_sit, 15, 70)

    gap_run  = max(0.0, (base_time - goal_3200_s)/60.0)
    gap_push = max(0.0, goal_push - base_push)
    gap_sit  = max(0.0, goal_sit  - base_sit)

    runs_per_wk = 2.5 + 0.35*gap_run
    if cls>=3: runs_per_wk -= 0.3
    runs_per_wk = clip(runs_per_wk, 2, 5)

    base_km = 10 if sex==0 else 12
    run_km_wk = base_km + 2.2*gap_run
    if cls>=4: run_km_wk *= 0.9
    run_km_wk = clip(run_km_wk, 8, 45)

    intervals = clip(1.0 + 0.25*gap_run, 1, 3)
    easy_runs = clip(runs_per_wk - intervals, 1, 4)

    strength_per_wk = clip(2.0 + 0.12*(gap_push+gap_sit), 2, 4)
    push_sets = clip(3.0 + 0.05*gap_push, 3, 6)
    sit_sets  = clip(3.0 + 0.04*gap_sit , 3, 6)

    if knee:
        intervals = max(1.0, intervals-1.0)
        easy_runs = max(1.0, easy_runs)
        run_km_wk = max(8.0, run_km_wk*0.9)
    if shoulder:
        push_sets = max(2.0, push_sets*0.7)
    if back:
        sit_sets  = max(2.0, sit_sets*0.8)

    bmr  = mifflin_st_jeor(w_kg, h_m, sex)
    tdee = tdee_from_sessions(bmr, runs_per_wk, strength_per_wk)
    if cls>=4:   kcal = tdee*0.78
    elif cls==3: kcal = tdee*0.85
    elif cls<=1: kcal = tdee*1.05
    else:        kcal = tdee*0.98

    prot_gkg  = clip(1.6 + 0.15*strength_per_wk, 1.6, 2.2)
    carbs_gkg = clip(3.0 + 0.4*runs_per_wk, 3.0, 6.0)
    fat_gkg   = 0.8
    protein_g = prot_gkg * w_kg
    carbs_g   = carbs_gkg * w_kg
    fat_g     = fat_gkg * w_kg

    return dict(
        run_km_wk=run_km_wk,
        runs_per_wk=runs_per_wk,
        intervals_per_wk=intervals,
        easy_runs_per_wk=easy_runs,
        strength_per_wk=strength_per_wk,
        push_sets=push_sets,
        sit_sets=sit_sets,
        kcal=kcal,
        protein_g=protein_g,
        fat_g=fat_g,
        carbs_g=carbs_g
    )

# ---------- carga pools desde tus CSVs ----------
def load_goal_pools(goals_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Devuelve:
    {
      'varones': {'run_s': array([...]), 'push': array([...]), 'sit': array([...])},
      'damas'  : { ... }
    }
    """
    def load_run_csv(path):
        df = pd.read_csv(path)
        if "time_s" in df.columns:
            vals = pd.to_numeric(df["time_s"], errors="coerce").dropna().values
        else:
            # intenta parsear "time" o "time_str"
            time_col = None
            for c in df.columns:
                if "time" in c.lower():
                    time_col = c; break
            if time_col is None:  # último recurso: primera col tipo mm:ss que parsea ok
                for c in df.columns:
                    if df[c].astype(str).str.contains(":").any():
                        time_col = c; break
            vals = df[time_col].astype(str).map(parse_time_to_seconds).dropna().values
        # limpia extremos absurdos
        vals = vals[(vals>=9*60) & (vals<=40*60)]
        return np.unique(vals)

    def load_reps_csv(path):
        df = pd.read_csv(path)
        # si hay "reps", úsala directo
        cols_lower = [c.lower() for c in df.columns]
        if "reps" in cols_lower:
            rcol = df.columns[cols_lower.index("reps")]
            vals = pd.to_numeric(df[rcol], errors="coerce").dropna().values
        else:
            # junta todos los valores numéricos plausibles (ignorando 'score' o tiempos)
            vals_all = []
            for c in df.columns:
                if "score" in c.lower(): continue
                if "time"  in c.lower(): continue
                s = pd.to_numeric(df[c], errors="coerce").dropna()
                vals_all.append(s)
            if len(vals_all)==0:
                vals = np.array([20,30,40,50,60], dtype=float)  # fallback
            else:
                vals = pd.concat(vals_all).values
        vals = vals[(vals>=5) & (vals<=120)]
        return np.unique(vals)

    pools = {"varones":{}, "damas":{}}
    pools["varones"]["run_s"] = load_run_csv(os.path.join(goals_dir, "varones_aerobica_3200m_clean.csv"))
    pools["damas"]["run_s"]   = load_run_csv(os.path.join(goals_dir, "damas_aerobica_3200m_clean.csv"))
    pools["varones"]["push"]  = load_reps_csv(os.path.join(goals_dir, "varones_flexiones_2min_clean.csv"))
    pools["damas"]["push"]    = load_reps_csv(os.path.join(goals_dir, "damas_flexiones_2min_clean.csv"))
    pools["varones"]["sit"]   = load_reps_csv(os.path.join(goals_dir, "varones_abdominales_2min_clean.csv"))
    pools["damas"]["sit"]     = load_reps_csv(os.path.join(goals_dir, "damas_abdominales_2min_clean.csv"))
    # seguridad: si algo quedó vacío, mete defaults razonables
    for sex in ["varones","damas"]:
        if pools[sex]["run_s"].size==0:
            pools[sex]["run_s"] = np.unique(np.array([12*60, 14*60, 16*60, 18*60, 20*60, 22*60], dtype=float))
        if pools[sex]["push"].size==0: pools[sex]["push"] = np.unique(np.array([15,25,35,45,55]))
        if pools[sex]["sit"].size==0:  pools[sex]["sit"]  = np.unique(np.array([20,30,40,50,60]))
    return pools

def load_pesotalla(pth: str) -> pd.DataFrame:
    pt = pd.read_csv(pth)
    nm = {c.lower().strip(): c for c in pt.columns}

    def get(*cands):
        for k in cands:
            if k in nm: return nm[k]
        return None

    # --- sex ---
    sex_col = get("sex","sexo")
    if sex_col is None:
        raise ValueError("CSV pesotalla debe tener columna 'sex' (damas/varones) o 'sexo'.")
    sex_map = {"damas":0,"mujeres":0,"female":0,"femenino":0,
               "varones":1,"hombres":1,"male":1,"masculino":1,0:0,1:1}
    sex = pt[sex_col].astype(str).str.lower().map(sex_map)
    if sex.isna().any():  # por si ya viene 0/1 numérico
        sex_num = pd.to_numeric(pt[sex_col], errors="coerce")
        sex = sex.fillna(sex_num).astype(int)

    # --- categoría ---
    cat_col = get("status","categoria","category","clasificacion")
    if cat_col is None:
        raise ValueError("CSV pesotalla debe tener 'status'/'categoria'.")
    cat_norm = pt[cat_col].astype(str).str.lower().str.replace(" ", "", regex=False)
    cat_map = {"desnutricion":0,"desnutrición":0,"riesgo":1,"riesgodesnutricion":1,"riesgodesnutrición":1,
               "normal":2,"sobrepeso":3,"sobrepeso.":3,"obesidad":4}
    cat_idx = cat_norm.map(cat_map).astype("Int64")

    # --- altura: soporta height_cm O min/max ---
    h_cm_col = get("height_cm","estatura_cm")
    hmin_col = get("estaturamin_cm","estatura_min_cm","height_min_cm")
    hmax_col = get("estaturamax_cm","estatura_max_cm","height_max_cm")

    if h_cm_col is not None:
        est_min = pd.to_numeric(pt[h_cm_col], errors="coerce")
        est_max = pd.to_numeric(pt[h_cm_col], errors="coerce")
    else:
        if hmin_col is None or hmax_col is None:
            raise ValueError("Faltan columnas de altura: use 'height_cm' o 'estatura_min_cm' + 'estatura_max_cm'.")
        est_min = pd.to_numeric(pt[hmin_col], errors="coerce")
        est_max = pd.to_numeric(pt[hmax_col], errors="coerce")

    # --- peso ideal (si está, úsalo; si no, intenta derivarlo con rango NORMAL) ---
    ideal_col = get("peso_ideal_kg","ideal_kg","pesoideal_kg")
    if ideal_col is not None:
        peso_ideal = pd.to_numeric(pt[ideal_col], errors="coerce")
    else:
        wmin_col = get("peso_min_kg","min_kg","peso_min")
        wmax_col = get("peso_max_kg","max_kg","peso_max")
        if wmin_col and wmax_col:
            wmin = pd.to_numeric(pt[wmin_col], errors="coerce")
            wmax = pd.to_numeric(pt[wmax_col], errors="coerce")
            # si la fila es NORMAL, toma punto medio; si no hay info, NaN
            peso_ideal = np.where(cat_idx==2, (wmin+wmax)/2.0, np.nan)
        else:
            peso_ideal = pd.Series(np.nan, index=pt.index)

    out = pd.DataFrame({
        "sex": sex.astype(int),
        "estatura_min_cm": est_min,
        "estatura_max_cm": est_max,
        "cat_idx": cat_idx,
        "peso_ideal_kg": pd.to_numeric(peso_ideal, errors="coerce")
    })
    return out


def annotate_with_pesotalla(A: pd.DataFrame, pt: pd.DataFrame) -> pd.DataFrame:
    """
    A: DataFrame con columnas ['sex','h_m','w_kg', ...]
    pt: salida de load_pesotalla(...) con columnas:
        ['sex','estatura_min_cm','estatura_max_cm','cat_idx','peso_ideal_kg']
    Devuelve A con columnas extra: 'h_cm','ideal_w_kg','pt_cat_idx','delta_to_ideal'
    """
    A = A.copy()
    A["h_cm"] = (A["h_m"] * 100).round().astype(int)

    # Para cada (sex, h_cm), elige fila de pt que contenga h_cm en [min,max].
    # Si hay varias, prioriza categoría NORMAL (cat_idx==2); si no, la primera.
    ideals = []
    cats   = []
    for sx, hc in zip(A["sex"].astype(int).tolist(), A["h_cm"].tolist()):
        sub = pt[(pt["sex"]==sx) &
                 (pt["estatura_min_cm"]<=hc) &
                 (pt["estatura_max_cm"]>=hc)]
        if len(sub)==0:
            ideals.append(np.nan); cats.append(np.nan); continue
        # prioriza NORMAL
        sub = sub.sort_values(by=["cat_idx"], key=lambda s: (s!=2))
        ideals.append(pd.to_numeric(sub["peso_ideal_kg"], errors="coerce").iloc[0])
        cats.append(int(sub["cat_idx"].iloc[0]))
    A["ideal_w_kg"] = ideals
    A["pt_cat_idx"] = cats
    A["delta_to_ideal"] = A["w_kg"] - A["ideal_w_kg"]
    return A


# ---------- generación X a partir de anthro_seed o sintético ----------
def sample_anthro(n: int) -> pd.DataFrame:
    sex = np.random.binomial(1, 0.5, size=n)
    h_m  = np.where(sex==1,
                    np.random.normal(1.73, 0.07, size=n),
                    np.random.normal(1.61, 0.07, size=n))
    h_m = np.clip(h_m, 1.45, 2.0)
    bmi = np.random.choice([18.0,19.0,22.0,27.0,32.0], size=n, p=[0.08,0.12,0.5,0.2,0.1]) \
          + np.random.normal(0, 1.2, size=n)
    bmi = np.clip(bmi, 16.0, 38.0)
    w_kg = bmi*(h_m**2)
    return pd.DataFrame(dict(sex=sex, h_m=h_m, w_kg=w_kg))

def build_dataset(pools, anthro_csv=None, pesotalla_csv=None, n_synth=20000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if anthro_csv and os.path.exists(anthro_csv):
        A = pd.read_csv(anthro_csv)
        A = A[["sex","h_m","w_kg"]].dropna()
        A["sex"] = A["sex"].astype(int)
    else:
        A = sample_anthro(n_synth)

    # restricciones aleatorias (puedes cambiar probabilidades)
    A["knee"] = np.random.binomial(1, 0.08, size=len(A))
    A["shoulder"] = np.random.binomial(1, 0.08, size=len(A))
    A["back"] = np.random.binomial(1, 0.10, size=len(A))
    A["vegan"] = np.random.binomial(1, 0.06, size=len(A))
    A["lactose_free"] = np.random.binomial(1, 0.10, size=len(A))
    A["gluten_free"]  = np.random.binomial(1, 0.05, size=len(A))

    # metas muestreadas de tus CSVs por sexo
    run_s, push, sit = [], [], []
    for _, r in A.iterrows():
        sex_key = "varones" if int(r["sex"])==1 else "damas"
        run_s.append(np.random.choice(pools[sex_key]["run_s"]))
        push.append(int(np.random.choice(pools[sex_key]["push"])))
        sit.append(int(np.random.choice(pools[sex_key]["sit"])))
    A["goal_3200_s"] = np.array(run_s, dtype=float)
    A["goal_push"]   = np.array(push, dtype=int)
    A["goal_sit"]    = np.array(sit, dtype=int)
    A["bmi"] = A["w_kg"]/(A["h_m"]**2)
    A["bmi_cls"] = A["bmi"].map(bmi_class).astype(int)
    A["bmi"] = A["w_kg"]/(A["h_m"]**2)
    A["bmi_cls"] = A["bmi"].map(bmi_class).astype(int)

    # ... dentro de build_dataset, justo después del if pesotalla_csv ...
    if pesotalla_csv is not None and os.path.exists(pesotalla_csv):
        pt = load_pesotalla(pesotalla_csv)
        A = annotate_with_pesotalla(A, pt)  # crea: ideal_w_kg, pt_cat_idx, delta_to_ideal
    else:
        # Si no hay tabla, crea columnas con los mismos NOMBRES que usa annotate_with_pesotalla
        A["ideal_w_kg"]    = 22.0*(A["h_m"]**2)
        A["pt_cat_idx"]    = A["bmi_cls"]
        A["delta_to_ideal"] = A["w_kg"] - A["ideal_w_kg"]




    # construir Y aplicando la política
    Ys = [policy(row.to_dict()) for _, row in A.iterrows()]
    Y = pd.DataFrame(Ys)

    return A, Y

# ---------- modelo ----------
def build_model(in_dim, out_dim):
    inputs = layers.Input(shape=(in_dim,))
    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.10)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(out_dim, activation="linear")(x)
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(1e-3),
        loss=losses.Huber(delta=1.0),
        metrics=["mae"]
    )
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals_dir", required=True, help="Carpeta con CSVs de objetivos")
    ap.add_argument("--anthro_csv", default=None, help="Opcional: CSV con sex,h_m,w_kg")
    ap.add_argument("--outdir", default="ckpts_planner", help="Salida (modelo y scalers)")
    ap.add_argument("--pesotalla_csv", default=None, help="CSV peso-talla largo")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logdir = "logs_planner"; os.makedirs(logdir, exist_ok=True)

    print("📥 Cargando pools desde CSVs:", args.goals_dir)
    pools = load_goal_pools(args.goals_dir)
    print({k:{kk:int(len(vv)) for kk,vv in d.items()} for k,d in pools.items()})

    print("🧪 Construyendo dataset…")
    Xdf, Ydf = build_dataset(pools, anthro_csv=args.anthro_csv, pesotalla_csv=args.pesotalla_csv)

    feature_cols = [
    "sex","h_m","w_kg","bmi","bmi_cls","pt_cat_idx",
    "ideal_w_kg","delta_to_ideal",
    "goal_3200_s","goal_push","goal_sit",
    "knee","shoulder","back","vegan","lactose_free","gluten_free"]

    target_cols  = ["run_km_wk","runs_per_wk","intervals_per_wk","easy_runs_per_wk",
                    "strength_per_wk","push_sets","sit_sets","kcal","protein_g","fat_g","carbs_g"]



    X = Xdf[feature_cols].astype("float32").values
    Y = Ydf[target_cols].astype("float32").values


    with open(os.path.join(args.outdir, "planner_columns.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_cols": feature_cols,
            "target_cols": target_cols
        }, f, ensure_ascii=False, indent=2)

    xsc, ysc = StandardScaler(), StandardScaler()
    Xz = xsc.fit_transform(X).astype("float32")
    Yz = ysc.fit_transform(Y).astype("float32")
    joblib.dump(xsc, os.path.join(args.outdir, "planner_x_scaler.pkl"))
    joblib.dump(ysc, os.path.join(args.outdir, "planner_y_scaler.pkl"))

    Xtr, Xva, Ytr, Yva = train_test_split(Xz, Yz, test_size=0.2, random_state=SEED)
    print("📐 Shapes:", Xtr.shape, Ytr.shape)

    model = build_model(Xtr.shape[1], Ytr.shape[1])

    cbs = [
        callbacks.EarlyStopping(monitor="val_mae", mode="min", patience=25, min_delta=1e-4,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(os.path.join(args.outdir, "planner_best.keras"),
                                  monitor="val_mae", mode="min", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_mae", mode="min",
                                    factor=0.5, patience=10, min_lr=5e-5, verbose=1),
        callbacks.CSVLogger(os.path.join(logdir, "train_log.csv"), append=True),
        callbacks.BackupAndRestore(backup_dir=os.path.join(logdir, "backup"))
    ]

    model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
              epochs=args.epochs, batch_size=args.batch, shuffle=True, callbacks=cbs, verbose=1)

    model.save(os.path.join(args.outdir, "planner_mlp.keras"), include_optimizer=False)
    print("✅ Guardado:", os.path.join(args.outdir, "planner_mlp.keras"))

    # Demo (construido con el mismo pipeline que el dataset)
    demo = pd.DataFrame([dict(sex=1, h_m=1.75, w_kg=92.0)])
    demo["bmi"] = demo["w_kg"] / (demo["h_m"]**2)
    demo["bmi_cls"] = demo["bmi"].map(bmi_class).astype(int)

    # Añade anotaciones de peso-talla si hay CSV, si no usa defaults consistentes
    if args.pesotalla_csv and os.path.exists(args.pesotalla_csv):
        pt_demo = load_pesotalla(args.pesotalla_csv)
        demo = annotate_with_pesotalla(demo, pt_demo)  # crea ideal_w_kg, pt_cat_idx, delta_to_ideal
    else:
        demo["ideal_w_kg"]     = 22.0 * (demo["h_m"]**2)
        demo["pt_cat_idx"]     = demo["bmi_cls"]
        demo["delta_to_ideal"] = demo["w_kg"] - demo["ideal_w_kg"]

    # Metas y restricciones del ejemplo
    demo["goal_3200_s"] = 18*60 + 30
    demo["goal_push"]   = 45
    demo["goal_sit"]    = 50
    demo["knee"] = 0; demo["shoulder"] = 0; demo["back"] = 0
    demo["vegan"] = 0; demo["lactose_free"] = 0; demo["gluten_free"] = 0

    # Mismo orden de features que en entrenamiento
    feature_cols = [
        "sex","h_m","w_kg","bmi","bmi_cls","pt_cat_idx",
        "ideal_w_kg","delta_to_ideal",
        "goal_3200_s","goal_push","goal_sit",
        "knee","shoulder","back","vegan","lactose_free","gluten_free"
    ]

    x_demo = demo[feature_cols].astype("float32").values
    y_pred = ysc.inverse_transform(model.predict(xsc.transform(x_demo))).ravel()
    print("🧪 Demo:", dict(zip(target_cols, y_pred.tolist())))


if __name__ == "__main__":
    main()
