# predict_cli.py
# Inferencia local del modelo antropométrico:
# - Carga .keras/.h5 y scaler_medidas.pkl
# - Acepta 1..N imágenes y sexo (auto/F/M/0/1/0.5)
# - Devuelve altura (m), peso (kg) y clase (NET y por REGLA peso-talla/IMC)
# - Opcional: exporta CSV

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# ========================
# Config básica
# ========================
CLASS_NAMES = ["desnutricion","riesgo","normal","sobrepeso","obesidad"]
IMG_SIZE = (224, 224)

# ========================
# Custom objects / pérdidas
# ========================
huber_loss = tf.keras.losses.Huber(delta=1.5)
mae_loss   = tf.keras.losses.MeanAbsoluteError()
def mixed_reg_loss(y_true, y_pred):
    # Debe existir para deserializar modelos guardados con esta loss
    return 0.7 * huber_loss(y_true, y_pred) + 0.3 * mae_loss(y_true, y_pred)

class BrightnessContrastLayer(tf.keras.layers.Layer):
    """En inferencia no aplica jitter, pero debe existir para deserializar."""
    def __init__(self, max_delta_brightness=0.05, lower_c=0.9, upper_c=1.1, **kwargs):
        super().__init__(**kwargs)
        self.max_delta_brightness = max_delta_brightness
        self.lower_c = lower_c
        self.upper_c = upper_c
    def call(self, x, training=None):
        return x

# --- Registro para Keras 3 (y compatibilidad 2.x) ---
try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable(package="custom")
def apply_weight_correction(t):
    """Debe coincidir EXACTO con lo usado al entrenar."""
    reg_base, w_delta = t
    altura_z = reg_base[:, :1]
    peso_z   = reg_base[:, 1:2] + 0.1 * w_delta
    return tf.concat([altura_z, peso_z], axis=1)

@register_keras_serializable(package="custom")
def preprocess_input_registered(x):
    """Para deserializar Lambda(preprocess_input) si está dentro del grafo."""
    return preprocess_input(x)

# ========================
# Reglas de clasificación
# ========================
def bmi_to_class5(bmi: float) -> int:
    # 0: <18.5 | 1: 18.5-19.9 | 2: 20.0-24.9 | 3: 25.0-29.9 | 4: >=30
    if bmi < 18.5: return 0
    if bmi < 20.0: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

def load_pesotalla_table(csv_path: str):
    if not (csv_path and os.path.exists(csv_path)): return None
    pt = pd.read_csv(csv_path)
    cols_lc = {c.lower().strip(): c for c in pt.columns}
    def pick(*cands):
        for k in cands:
            if k in cols_lc: return cols_lc[k]
        return None
    sex_col = pick("sex","sexo")
    hmin = pick("estatura_min_cm","estaturamin_cm","height_min_cm")
    hmax = pick("estatura_max_cm","estaturamax_cm","height_max_cm")
    wmin = pick("peso_min_kg","min_kg","peso_min")
    wmax = pick("peso_max_kg","max_kg","peso_max")
    cat_col = pick("categoria","status","clasificacion","category")

    if sex_col is None or (hmin is None and hmax is None) or (wmin is None or wmax is None) or cat_col is None:
        return None

    sex_map={"damas":0,"mujeres":0,"female":0,"femenino":0,"varones":1,"hombres":1,"male":1,"masculino":1,0:0,1:1}
    pt["sex"] = pt[sex_col].map(lambda x: sex_map.get(str(x).lower(), np.nan)).astype("Int64")
    pt["estatura_min_cm"] = pd.to_numeric(pt[hmin], errors="coerce")
    pt["estatura_max_cm"] = pd.to_numeric(pt[hmax], errors="coerce")

    cat_norm = pt[cat_col].astype(str).str.lower().str.replace(" ", "", regex=False)
    cat_map = {"desnutricion":0,"desnutrición":0,"riesgo":1,"riesgodesnutricion":1,"riesgodesnutrición":1,"normal":2,"sobrepeso":3,"obesidad":4}
    pt["cat_idx"] = cat_norm.map(cat_map).astype("Int64")

    pt["peso_min_kg"] = pd.to_numeric(pt[wmin], errors="coerce")
    pt["peso_max_kg"] = pd.to_numeric(pt[wmax], errors="coerce")

    keep = ["sex","estatura_min_cm","estatura_max_cm","cat_idx","peso_min_kg","peso_max_kg"]
    pt = pt[keep].dropna()
    return pt

def class_from_pesotalla(height_m: float, weight_kg: float, sex_num: int, pt_table: pd.DataFrame) -> int:
    """Devuelve índice 0..4 usando tabla peso-talla + sexo; si no hay tabla o sexo, cae a IMC."""
    if pt_table is None or sex_num not in (0,1):
        return bmi_to_class5(weight_kg/(height_m**2))
    h_cm = int(round(height_m*100))
    sub = pt_table[(pt_table["sex"]==sex_num) &
                   (pt_table["estatura_min_cm"]<=h_cm) &
                   (pt_table["estatura_max_cm"]>=h_cm)]
    if len(sub)==0:
        return bmi_to_class5(weight_kg/(height_m**2))
    inside = sub[(sub["peso_min_kg"]<=weight_kg) & (weight_kg<=sub["peso_max_kg"])]
    if len(inside)>0:
        return int(inside.sort_values("cat_idx").iloc[0]["cat_idx"])
    if weight_kg < sub["peso_min_kg"].min(): return 0
    if weight_kg > sub["peso_max_kg"].max(): return 4
    mids = (sub["peso_min_kg"] + sub["peso_max_kg"])/2.0
    idx = (mids - weight_kg).abs().idxmin()
    return int(sub.loc[idx, "cat_idx"])

# ========================
# Utilidades
# ========================
def detect_sex_from_filename(path):
    base = os.path.basename(path)
    if "_F_" in base: return 0.0
    if "_M_" in base: return 1.0
    return 0.5  # neutro si no hay pista

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    arr = img_to_array(img)
    return arr

# ========================
# Main
# ========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="modelo_antropometrico.keras", help="Ruta al modelo (.keras o .h5)")
    ap.add_argument("--scaler", default="scaler_medidas.pkl", help="Ruta al scaler Joblib")
    ap.add_argument("--pesotalla_csv", default="", help="Ruta a comb_pesotalla_long_finite.csv (opcional)")
    ap.add_argument("--sex", default="auto", choices=["auto","F","M","0","1","0.5"], help="Sexo global para imágenes")
    ap.add_argument("--out_csv", default="", help="Si se indica, guarda resultados en CSV")
    ap.add_argument("images", nargs="+", help="Rutas de imagen a evaluar")
    args = ap.parse_args()

    # --- Cargar modelo con custom_objects registrados ---
    model = load_model(
        args.model,
        custom_objects={
            "mixed_reg_loss": mixed_reg_loss,
            "BrightnessContrastLayer": BrightnessContrastLayer,
            "apply_weight_correction": apply_weight_correction,
            "preprocess_input": preprocess_input_registered,
        },
        compile=False
    )

    # --- Cargar scaler y tabla opcional ---
    scaler = joblib.load(args.scaler)
    pt_table = load_pesotalla_table(args.pesotalla_csv) if args.pesotalla_csv else None

    # --- Resolver sexo global ---
    sex_global = None
    if args.sex in ["F","0"]: sex_global = 0.0
    elif args.sex in ["M","1"]: sex_global = 1.0
    elif args.sex == "0.5": sex_global = 0.5

    # --- Preparar batch ---
    rows = []
    batch_imgs, batch_sex, names = [], [], []
    for p in args.images:
        if not os.path.exists(p):
            raise FileNotFoundError(f"No existe la imagen: {p}")
        names.append(os.path.basename(p))
        batch_imgs.append(load_image(p))
        if sex_global is None:
            batch_sex.append(detect_sex_from_filename(p))
        else:
            batch_sex.append(sex_global)

    X = np.stack(batch_imgs).astype("float32")
    S = np.array(batch_sex, dtype="float32").reshape(-1,1)

    # Si tu modelo YA incluye Lambda(preprocess_input), no lo apliques aquí.
    # Mantén Xp = X para evitar doble normalización.
    Xp = X
    # Si tu modelo NO incluyera el Lambda, descomenta:
    # Xp = preprocess_input(X)

    # --- Inferencia ---
    pred_reg_z, pred_cls_prob = model.predict([Xp, S], batch_size=8, verbose=0)
    pred_reg_real = scaler.inverse_transform(pred_reg_z)
    pred_cls_idx  = pred_cls_prob.argmax(axis=1)

    # --- Postproceso / salida ---
    for name, (h,w), c_idx, sx in zip(names, pred_reg_real, pred_cls_idx, batch_sex):
        rule_idx = class_from_pesotalla(h, w, int(sx) if sx in (0.0,1.0) else -1, pt_table)
        rows.append({
            "image": name,
            "sex_input": float(sx),
            "height_m": float(round(h, 3)),
            "weight_kg": float(round(w, 1)),
            "class_net_idx": int(c_idx),
            "class_net_name": CLASS_NAMES[int(c_idx)],
            "class_rule_idx": int(rule_idx),
            "class_rule_name": CLASS_NAMES[int(rule_idx)],
        })

    df = pd.DataFrame(rows)
    # Imprime tabla en consola
    print(df.to_string(index=False))

    # Exporta CSV si se pidió
    if args.out_csv:
        df.to_csv(args.out_csv, index=False, encoding="utf-8")
        print(f"\n✅ Guardado: {args.out_csv}")

if __name__ == "__main__":
    main()
