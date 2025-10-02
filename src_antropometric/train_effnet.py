import os, sys, math, warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Lambda, LeakyReLU, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard, BackupAndRestore
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Input, Lambda, LeakyReLU, BatchNormalization, Dropout)
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Input, Lambda, LeakyReLU, BatchNormalization, Dropout, Concatenate
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# --- Pérdida mixta para REG (más robusta a outliers) ---
huber_loss = tf.keras.losses.Huber(delta=1.5)
mae_loss   = tf.keras.losses.MeanAbsoluteError()

def mixed_reg_loss(y_true, y_pred):
    # 70% Huber + 30% MAE
    return 0.7 * huber_loss(y_true, y_pred) + 0.3 * mae_loss(y_true, y_pred)


# ========================
# CONFIG
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FASE1 = 8
EPOCHS_FASE2 = 120

IMG_DIR = "data/images/datasets/datasets/ALLIMAGES"
CSV_PATH = "data/2DImage2BMI/ALL_feature/Image_train.csv"

# CSV PESO-TALLA (largo) OPCIONAL
PESOTALLA_CSV = "data/goals/comb_pesotalla_long_finite.csv"

LR_FASE1 = 1e-3
LR_FASE2 = 3e-4
SEED = 42

os.makedirs("ckpts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

CLASS_NAMES = ["desnutricion","riesgo","normal","sobrepeso","obesidad"]
N_CLASSES = 5

print("exe:", sys.executable)
print("TF:", tf.__version__)
print("Dispositivos CPU/GPU visibles:", tf.config.list_physical_devices())

# ========================
# UTILIDADES
# ========================
def load_image(file):
    path = os.path.join(IMG_DIR, file)
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)   # sin /255 — usamos preprocess_input
    return img

def bmi_to_class5(bmi: float) -> int:
    # 0: <18.5 | 1: 18.5-19.9 | 2: 20.0-24.9 | 3: 25.0-29.9 | 4: >=30
    if bmi < 18.5: return 0
    if bmi < 20.0: return 1
    if bmi < 25.0: return 2
    if bmi < 30.0: return 3
    return 4

def load_pesotalla_table(pth: str):
    if not (pth and os.path.exists(pth)):
        print("⚠️ No se encontró PESOTALLA_CSV; se usará fallback por IMC.")
        return None
    pt = pd.read_csv(pth)
    # normaliza nombres
    cols_lc = {c.lower().strip(): c for c in pt.columns}

    def pick(*cands):
        for k in cands:
            if k in cols_lc: return cols_lc[k]
        return None

    # sex
    sex_col = pick("sex","sexo")
    sex_map = {"damas":0,"mujeres":0,"female":0,"femenino":0,
               "varones":1,"hombres":1,"male":1,"masculino":1,0:0,1:1}
    sex = pt[sex_col].map(lambda x: sex_map.get(str(x).lower(), np.nan)).astype("Int64")

    # estaturas
    hmin = pick("estatura_min_cm","estaturamin_cm","height_min_cm")
    hmax = pick("estatura_max_cm","estaturamax_cm","height_max_cm")
    if hmin is None or hmax is None:
        # por si viene height_cm
        hc = pick("height_cm","estatura_cm")
        if hc is None: raise ValueError("CSV peso-talla debe tener min/max de estatura o height_cm.")
        pt["estatura_min_cm"] = pd.to_numeric(pt[hc], errors="coerce")
        pt["estatura_max_cm"] = pd.to_numeric(pt[hc], errors="coerce")
    else:
        pt["estatura_min_cm"] = pd.to_numeric(pt[hmin], errors="coerce")
        pt["estatura_max_cm"] = pd.to_numeric(pt[hmax], errors="coerce")

    # categoria
    cat_col = pick("categoria","status","clasificacion","category")
    cat_norm = pt[cat_col].astype(str).str.lower().str.replace(" ", "", regex=False)
    cat_map = {"desnutricion":0,"desnutrición":0,"riesgo":1,"riesgodesnutricion":1,"riesgodesnutrición":1,
               "normal":2,"sobrepeso":3,"obesidad":4}
    pt["cat_idx"] = cat_norm.map(cat_map).astype("Int64")

    # pesos
    wmin = pick("peso_min_kg","min_kg","peso_min")
    wmax = pick("peso_max_kg","max_kg","peso_max")
    if wmin is None or wmax is None:
        raise ValueError("CSV peso-talla debe tener peso_min_kg y peso_max_kg (rango por categoría).")
    pt["peso_min_kg"] = pd.to_numeric(pt[wmin], errors="coerce")
    pt["peso_max_kg"] = pd.to_numeric(pt[wmax], errors="coerce")

    # ideal
    ideal_col = pick("peso_ideal_kg","ideal_kg","pesoideal_kg")
    if ideal_col is not None:
        pt["peso_ideal_kg"] = pd.to_numeric(pt[ideal_col], errors="coerce")
    else:
        pt["peso_ideal_kg"] = np.nan

    pt["sex"] = sex
    keep = ["sex","estatura_min_cm","estatura_max_cm","cat_idx","peso_min_kg","peso_max_kg","peso_ideal_kg"]
    pt = pt[keep].dropna(subset=["sex","estatura_min_cm","estatura_max_cm","cat_idx","peso_min_kg","peso_max_kg"])
    return pt

def class_from_pesotalla(height_m: float, weight_kg: float, sex_num: int, pt_table: pd.DataFrame) -> int:
    """
    Devuelve índice de clase 0..4 usando la tabla peso-talla y el sexo (0 damas / 1 varones).
    Si no hay match exacto, asigna extremos; si no hay tabla, cae a IMC.
    """
    if pt_table is None or pd.isna(sex_num):
        return bmi_to_class5(weight_kg/(height_m**2))

    h_cm = int(round(height_m*100))
    sub = pt_table[(pt_table["sex"]==sex_num) &
                   (pt_table["estatura_min_cm"]<=h_cm) &
                   (pt_table["estatura_max_cm"]>=h_cm)]
    if len(sub)==0:
        return bmi_to_class5(weight_kg/(height_m**2))

    # match si cae dentro de algún rango
    inside = sub[(sub["peso_min_kg"]<=weight_kg) & (weight_kg<=sub["peso_max_kg"])]
    if len(inside)>0:
        return int(inside.sort_values("cat_idx").iloc[0]["cat_idx"])

    # si está fuera de todos los rangos de esa estatura, decide por extremos
    if weight_kg < sub["peso_min_kg"].min(): return 0
    if weight_kg > sub["peso_max_kg"].max(): return 4

    # si llegó aquí, el peso está entre rangos pero no cayó en ninguno (huecos):
    # toma el rango más cercano por distancia al centro
    mids = (sub["peso_min_kg"] + sub["peso_max_kg"])/2.0
    idx = (mids - weight_kg).abs().idxmin()
    return int(sub.loc[idx, "cat_idx"])

# ========================
# CARGA CSV + DETECCIÓN ALTURA/PESO (+ SEX opcional)
# ========================
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["img_name"] + [f"col_{i}" for i in range(1, df.shape[1])]

# Detectar columnas numéricas plausibles
numdf = df.drop(columns=["img_name"]).apply(pd.to_numeric, errors="coerce")
height_cands = [c for c in numdf.columns if numdf[c].between(1.2, 2.2).mean() > 0.90]
weight_cands = [c for c in numdf.columns if numdf[c].between(30, 200).mean() > 0.60]

# Fallbacks
hcol = height_cands[-1] if height_cands else df.columns[-2]
wcol = weight_cands[-1] if weight_cands else df.columns[-1]
if hcol == wcol:
    hcol, wcol = df.columns[-2], df.columns[-1]

df["altura"] = pd.to_numeric(df[hcol], errors="coerce").astype("float32")
df["peso"]   = pd.to_numeric(df[wcol], errors="coerce").astype("float32")

# Intentar detectar sexo desde el CSV o desde el nombre del archivo
sex_col = None
for cand in ["sex", "sexo", "genero", "género"]:
    if cand in df.columns:
        sex_col = cand
        break

if sex_col is not None:
    # Mapeo explícito desde la columna del CSV
    sex_map = {"damas": 0, "mujeres": 0, "female": 0, "femenino": 0,
               "varones": 1, "hombres": 1, "male": 1, "masculino": 1, 0: 0, 1: 1}
    df["sex_num"] = df[sex_col].map(lambda x: sex_map.get(str(x).lower(), np.nan)).astype("float32")
else:
    # Fallback: detectar desde nombre del archivo (ejemplo: 0_F_15_162560_6123497.jpg)
    def detect_sex_from_filename(fname):
        base = os.path.basename(fname)
        if "_F_" in base:
            return 0.0  # mujer
        elif "_M_" in base:
            return 1.0  # hombre
        else:
            return np.nan
    df["sex_num"] = df["img_name"].map(detect_sex_from_filename).astype("float32")

print(f"Cols detectadas -> altura: {hcol} | peso: {wcol}")
print(df[["altura", "peso"]].describe())
assert df["altura"].std() > 1e-3, "⚠️ Varianza de ALTURA ~ 0 (columna mal mapeada)."
assert df["peso"].std()   > 1e-3, "⚠️ Varianza de PESO ~ 0 (columna mal mapeada)."

# Imputación: si no se detectó sexo, usar 0.5 (valor neutro entre 0 y 1)
df["sex_num_imputed"] = df["sex_num"].fillna(0.5).astype("float32")

# Carga tabla peso-talla (opcional)
pt_table = load_pesotalla_table(PESOTALLA_CSV)

# Etiquetas de clase (5) a partir de tabla peso-talla (si hay sexo) o fallback IMC
def row_to_class(r):
    h, w, sx = float(r["altura"]), float(r["peso"]), r["sex_num"]
    if np.isnan(sx):
        return bmi_to_class5(w/(h**2))
    else:
        return class_from_pesotalla(h, w, int(sx), pt_table)

df["clase5"] = df.apply(row_to_class, axis=1).astype(int)
print("Distribución de clases (5):", df["clase5"].value_counts().sort_index().to_dict())

# ========================
# IMÁGENES + TARGETS
# ========================
print("📷 Cargando imágenes...")
X = np.stack([load_image(fname) for fname in df["img_name"]], axis=0).astype("float32")

y_reg_full = df[["altura","peso"]].values.astype("float32")
y_class_full = tf.keras.utils.to_categorical(df["clase5"].values, num_classes=N_CLASSES).astype("float32")

# ========================
# TRAIN / VAL
# ========================
train_idx, val_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=df["clase5"].values
)
X_train, X_val = X[train_idx], X[val_idx]
y_reg_train, y_reg_val = y_reg_full[train_idx], y_reg_full[val_idx]
y_class_train, y_class_val = y_class_full[train_idx], y_class_full[val_idx]
val_filenames = df["img_name"].iloc[val_idx].tolist()
val_sex = df["sex_num"].iloc[val_idx].tolist()  # para la regla al final

sex_full  = df["sex_num_imputed"].values.astype("float32").reshape(-1, 1)
sex_train = sex_full[train_idx]
sex_val   = sex_full[val_idx]


# === Sample weights por salida (en vez de class_weight) ===
from sklearn.utils.class_weight import compute_class_weight

y_train_cls_idx = y_class_train.argmax(axis=1)
y_val_cls_idx   = y_class_val.argmax(axis=1)

present_classes = np.unique(y_train_cls_idx)
cw = compute_class_weight(class_weight="balanced",
                          classes=present_classes, y=y_train_cls_idx)
cw_map = {int(c): float(w) for c, w in zip(present_classes, cw)}

# Pesos por muestra para la salida de CLASIFICACIÓN
sw_cls_train = np.array([cw_map.get(int(c), 1.0) for c in y_train_cls_idx], dtype="float32")
sw_cls_val   = np.array([cw_map.get(int(c), 1.0) for c in y_val_cls_idx],   dtype="float32")

# Para la salida de REGRESIÓN: todo 1.0
sw_reg_train = np.ones(len(y_reg_train), dtype="float32")
sw_reg_val   = np.ones(len(y_reg_val),   dtype="float32")


# ========================
# ESCALADO (REGRESIÓN)
# ========================
scaler_y = StandardScaler()
y_reg_train = scaler_y.fit_transform(y_reg_train).astype("float32")
y_reg_val   = scaler_y.transform(y_reg_val).astype("float32")
print("📏 Escalador y guardado en scaler_medidas.pkl")
joblib.dump(scaler_y, "scaler_medidas.pkl")

# ========================
# MODELO
# ========================

class BrightnessContrastLayer(tf.keras.layers.Layer):
    """Pequeño ajuste de brillo/contraste solo en entrenamiento."""
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

data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    BrightnessContrastLayer(name="rand_brightness_contrast"),
], name="augment")

base_model = EfficientNetB0(include_top=False, weights="imagenet",
                            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Fase 1

# ===== Entradas =====
inputs_img = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")
inputs_sex = Input(shape=(1,), name="sex_input")  # 0=female, 1=male, 0.5=desconocido

# ===== Rama imagen (EffNet) =====
x = data_aug(inputs_img)
x = Lambda(preprocess_input, name="effnet_preproc")(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)

# ===== Rama sexo (pequeña proyección) =====
sex_feat = Dense(4, activation="relu", name="sex_dense")(inputs_sex)

# ===== Fusionar =====
x_combined = Concatenate(name="concat_img_sex")([x, sex_feat])

# ===== Cabeza REG (z-space [altura, peso]) con corrección de peso =====
x_reg = Dense(256, kernel_initializer="he_normal")(x_combined)
x_reg = BatchNormalization()(x_reg)
x_reg = LeakyReLU(0.1)(x_reg)

out_reg_base = Dense(2, activation="linear", name="regression_base")(x_reg)
w_corr = Dense(1, activation="linear", name="w_corr",
               kernel_initializer="zeros", bias_initializer="zeros")(x_reg)

def apply_weight_correction(t):
    reg_base, w_delta = t
    altura_z = reg_base[:, :1]
    peso_z   = reg_base[:, 1:2] + 0.1 * w_delta
    return tf.concat([altura_z, peso_z], axis=1)

out_reg = Lambda(apply_weight_correction, name="regression")([out_reg_base, w_corr])

# ===== Cabeza CLS (5) OUTPUTS =====
x_cls = Dense(128, activation="relu")(x_combined)
x_cls = Dropout(0.15)(x_cls)
out_cls = Dense(N_CLASSES, activation="softmax", name="classification")(x_cls)

model = Model(inputs=[inputs_img, inputs_sex], outputs=[out_reg, out_cls])


# ========================
# FASE 1 (head-only)
# ========================
model.compile(
    optimizer=Adam(learning_rate=LR_FASE1, clipnorm=1.0),
    loss=[mixed_reg_loss, CategoricalCrossentropy(label_smoothing=0.05)],
    metrics=[["mae"], ["accuracy"]],
    loss_weights=[1.0, 0.35],
)


cbs_f1 = [
    EarlyStopping(monitor="val_regression_mae", mode="min",
                  patience=12, min_delta=1e-4, restore_best_weights=True, verbose=1),
    ModelCheckpoint("ckpts/best_model_f1.keras", monitor="val_regression_mae",
                    mode="min", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_regression_loss", mode="min",
                      factor=0.5, patience=6, cooldown=2, min_lr=1e-6, verbose=1),
    CSVLogger("logs/f1.csv", append=True),
    BackupAndRestore(backup_dir="logs/backup_f1"),
]

print("🚀 Fase 1 (warm-up, base congelada) ...")
history1 = model.fit(
    [X_train, sex_train], [y_reg_train, y_class_train],
    validation_data=([X_val, sex_val], [y_reg_val, y_class_val], [sw_reg_val, sw_cls_val]),
    epochs=EPOCHS_FASE1,
    batch_size=BATCH_SIZE,
    callbacks=cbs_f1,
    shuffle=True, verbose=1,
    sample_weight=[sw_reg_train, sw_cls_train],
)


# ========================
# FASE 2 (fine-tuning con BN congeladas, solo últimas ~100 capas)
# ========================
base_model.trainable = True
for l in base_model.layers:
    if isinstance(l, BatchNormalization):
        l.trainable = False
# Libera un poco más feature extractor (≈ últimas 150 capas)
for l in base_model.layers[:-150]:
    l.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),  # más bajo en FT profundo
    loss=[mixed_reg_loss, CategoricalCrossentropy(label_smoothing=0.05)],
    metrics=[["mae"], ["accuracy"]],
    loss_weights=[1.0, 0.35],
)


cbs_f2 = [
    EarlyStopping(monitor="val_regression_mae", mode="min",
                  patience=30, min_delta=1e-4, restore_best_weights=True, verbose=1),
    ModelCheckpoint("ckpts/best_model_f2.keras", monitor="val_regression_mae",
                    mode="min", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_regression_loss", mode="min",
                      factor=0.5, patience=12, cooldown=3, min_lr=1e-6, verbose=1),
    CSVLogger("logs/f2.csv", append=True),
    TensorBoard(log_dir="logs/tb", update_freq="epoch"),
    BackupAndRestore(backup_dir="logs/backup_f2"),
]

print("🚀 Fase 2 (fine-tuning, base entrenable) ...")
history2 = model.fit(
    [X_train, sex_train], [y_reg_train, y_class_train],
    validation_data=([X_val, sex_val], [y_reg_val, y_class_val], [sw_reg_val, sw_cls_val]),
    epochs=EPOCHS_FASE2,
    batch_size=BATCH_SIZE,
    callbacks=cbs_f2,
    shuffle=True, verbose=1,
    sample_weight=[sw_reg_train, sw_cls_train],
)



# ========================
# CHEQUEOS + GUARDADO
# ========================
pred_reg_z, pred_cls_prob = model.predict([X_val, sex_val], batch_size=BATCH_SIZE)

print("STD pred z-space (altura_z, peso_z):", pred_reg_z.std(axis=0))

model.save("modelo_antropometrico.keras")
model.save("modelo_antropometrico.h5", include_optimizer=False)
print("✅ Guardado: .keras y .h5")

# ========================
# PREDICCIONES EN ESCALA REAL + CLASES
# ========================
scaler_y = joblib.load("scaler_medidas.pkl")
pred_reg_real = scaler_y.inverse_transform(pred_reg_z)  # [altura_m, peso_kg] reales

# clase por la red (argmax)
pred_cls_idx = pred_cls_prob.argmax(axis=1)
pred_cls_name = [CLASS_NAMES[i] for i in pred_cls_idx]

# clase por REGLA (pesotalla si hay sexo/tabla; si no, IMC)
rule_cls_idx = []
rule_cls_name = []
for (h, w), sx in zip(pred_reg_real, val_sex):
    ci = class_from_pesotalla(h, w, int(sx) if not np.isnan(sx) else -1, pt_table)
    rule_cls_idx.append(ci)
    rule_cls_name.append(CLASS_NAMES[ci])

# Métricas simples de regresión
y_true_real = scaler_y.inverse_transform(y_reg_val)
mae_altura = np.mean(np.abs(y_true_real[:,0]-pred_reg_real[:,0]))
mae_peso   = np.mean(np.abs(y_true_real[:,1]-pred_reg_real[:,1]))
print(f"MAE altura: {mae_altura:.2f} m  |  MAE peso: {mae_peso:.1f} kg")

# Accuracy de clasificación (siempre 5 clases)
y_true_cls = y_class_val.argmax(axis=1)
acc_net = (pred_cls_idx == y_true_cls).mean()
print(f"ACC clasificación (cabeza de red, 5 clases): {acc_net:.3f}")

# Guardar TXT
with open("predicciones_altura_peso_clase.txt", "w", encoding="utf-8") as f:
    for fname, (h, w), c_net, c_rule in zip(val_filenames, pred_reg_real, pred_cls_name, rule_cls_name):
        f.write(f"{fname} | Altura: {h:.3f} m | Peso: {w:.1f} kg | Clase_NET: {c_net} | Clase_REGLA: {c_rule}\n")
print("📄 Predicciones guardadas en predicciones_altura_peso_clase.txt")

# Plots rápidos
plt.figure(); plt.scatter(y_true_real[:,1], pred_reg_real[:,1], s=8); plt.plot([40,160],[40,160])
plt.xlabel("Peso real"); plt.ylabel("Peso predicho"); plt.title("VAL peso"); plt.savefig("val_peso.png", dpi=140); plt.close()

plt.figure(); plt.scatter(y_true_real[:,0], pred_reg_real[:,0], s=8); plt.plot([1.45,2.05],[1.45,2.05])
plt.xlabel("Altura real (m)"); plt.ylabel("Altura predicha (m)"); plt.title("VAL altura"); plt.savefig("val_altura.png", dpi=140); plt.close()
