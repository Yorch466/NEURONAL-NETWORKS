import os, sys, math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, BatchNormalization,
    Input, Lambda, LeakyReLU
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy, Huber
# ---- imports extra (dejalo junto con los demás) ----
from tensorflow.keras.callbacks import CSVLogger, TensorBoard, BackupAndRestore
import os



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import matplotlib.pyplot as plt

# ========================
# CONFIG
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_FASE1 = 8
EPOCHS_FASE2 = 120
IMG_DIR = "data/images/datasets/datasets/ALLIMAGES"
CSV_PATH = "data/2DImage2BMI/ALL_feature/Image_train.csv"
LR_FASE1 = 1e-3
LR_FASE2 = 3e-4
SEED = 42


os.makedirs("ckpts", exist_ok=True)
os.makedirs("logs", exist_ok=True)

np.random.seed(SEED)
tf.random.set_seed(SEED)

print("exe:", sys.executable)
print("TF:", tf.__version__)
print("Dispositivos CPU/GPU visibles:", tf.config.list_physical_devices())

# ========================
# UTILIDADES
# ========================
def load_image(file):
    """Carga imagen en rango [0..255], SIN dividir entre 255 (usamos preprocess_input)."""
    path = os.path.join(IMG_DIR, file)
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img)  # <- sin /255.0
    return img

# ========================
# CARGA CSV + DETECCIÓN ALTURA/PESO
# ========================
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["img_name"] + [f"col_{i}" for i in range(1, df.shape[1])]

# Detectar columnas numéricas y buscar candidatos por rangos plausibles
numdf = df.drop(columns=["img_name"]).apply(pd.to_numeric, errors="coerce")
height_cands = [c for c in numdf.columns if numdf[c].between(1.2, 2.2).mean() > 0.90]
weight_cands = [c for c in numdf.columns if numdf[c].between(30, 200).mean() > 0.60]

# Fallbacks (penúltima/última) si no encuentra
hcol = height_cands[-1] if height_cands else df.columns[-2]
wcol = weight_cands[-1] if weight_cands else df.columns[-1]
if hcol == wcol:
    hcol, wcol = df.columns[-2], df.columns[-1]

df["altura"] = pd.to_numeric(df[hcol], errors="coerce").astype("float32")
df["peso"]   = pd.to_numeric(df[wcol], errors="coerce").astype("float32")

print(f"Cols detectadas -> altura: {hcol} | peso: {wcol}")
print(df[["altura","peso"]].describe())
assert df["altura"].std() > 1e-3, "⚠️ Varianza de ALTURA ~ 0 (columna mal mapeada)."
assert df["peso"].std()   > 1e-3, "⚠️ Varianza de PESO ~ 0 (columna mal mapeada)."

# IMC y clase solo para estratificar (la cabeza de cls estará apagada)
df["imc"]   = df["peso"] / (df["altura"] ** 2)
df["clase"] = df["imc"].apply(lambda x: 0 if x < 18.5 else (1 if x < 25 else 2)).astype(int)
print("Distribución de clases (para split):", df["clase"].value_counts().to_dict())

# ========================
# IMÁGENES + TARGETS (SOLO 2: ALTURA, PESO)
# ========================
print("📷 Cargando imágenes...")
X = np.stack([load_image(fname) for fname in df["img_name"]], axis=0).astype("float32")
y_reg_full = df[["altura","peso"]].values.astype("float32")
y_class_full = tf.keras.utils.to_categorical(df["clase"].values, num_classes=3).astype("float32")

# ========================
# TRAIN / VAL (estratificado si hay ≥2 clases)
# ========================
unique_classes = df["clase"].nunique()
strat = df["clase"].values if unique_classes >= 2 else None
if strat is None:
    print("⚠️ Solo 1 clase detectada; split no estratificado.")

train_idx, val_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=strat
)
X_train, X_val = X[train_idx], X[val_idx]
y_reg_train, y_reg_val = y_reg_full[train_idx], y_reg_full[val_idx]
y_class_train, y_class_val = y_class_full[train_idx], y_class_full[val_idx]
val_filenames = df["img_name"].iloc[val_idx].tolist()

# ========================
# ESCALADO (SOLO 2 DIMENSIONES)
# ========================
scaler_y = StandardScaler()
y_reg_train = scaler_y.fit_transform(y_reg_train).astype("float32")
y_reg_val   = scaler_y.transform(y_reg_val).astype("float32")
print("y_reg_train mean/std (z):", y_reg_train.mean(axis=0), y_reg_train.std(axis=0))
joblib.dump(scaler_y, "scaler_medidas.pkl")
print("📏 Escalador guardado en scaler_medidas.pkl")

# ========================
# MODELO
# ========================
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="augment")

base_model = EfficientNetB0(include_top=False, weights="imagenet",
                            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # Fase 1

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_aug(inputs)
x = Lambda(preprocess_input, name="effnet_preproc")(x)
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)

# REG head (fase 1 sin dropout para maximizar señal)
x_reg = Dense(256, kernel_initializer="he_normal")(x)
x_reg = LeakyReLU(0.1)(x_reg)
salida_reg = Dense(2, activation="linear", name="regression")(x_reg)  # [altura, peso] z-space

# CLS head (apagada por loss_weights, pero presente)
x_cls = Dense(64, activation="relu")(x)
salida_cls = Dense(3, activation="softmax", name="classification")(x_cls)

model = Model(inputs=inputs, outputs=[salida_reg, salida_cls])

# ========================
# FASE 1 (head only)
# ========================
# ========================
# FASE 1 (head only)
# ========================
model.compile(
    optimizer=Adam(learning_rate=LR_FASE1, clipnorm=1.0),
    loss=[Huber(delta=1.5), CategoricalCrossentropy(label_smoothing=0.05)],
    metrics=[["mae"], ["accuracy"]],
    loss_weights=[1.0, 0.0],
)

cbs_f1 = [
    EarlyStopping(monitor="val_regression_mae", mode="min",
                  patience=12, min_delta=1e-4,   # << más paciencia + umbral
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("ckpts/best_model_f1.keras", monitor="val_regression_mae",
                    mode="min", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_regression_loss", mode="min",
                      factor=0.5, patience=6, cooldown=2, min_lr=1e-6, verbose=1),
    CSVLogger("logs/f1.csv", append=True),
    BackupAndRestore(backup_dir="logs/backup_f1"),
]

print("🚀 Fase 1 (warm-up, base congelada) ...")
history1 = model.fit(
    X_train, [y_reg_train, y_class_train],
    validation_data=(X_val, [y_reg_val, y_class_val]),
    epochs=EPOCHS_FASE1,
    batch_size=BATCH_SIZE,
    callbacks=cbs_f1,
    shuffle=True, verbose=1
)


# Chequeo de colapso tras Fase 1
pred_reg_z, _ = model.predict(X_val, batch_size=BATCH_SIZE)
print("STD pred z-space tras Fase 1 (altura_z, peso_z):", pred_reg_z.std(axis=0))

# ========================
# FASE 2 (fine-tuning con BN congeladas, solo últimas capas entrenables)
# ========================
base_model.trainable = True
for l in base_model.layers:
    if isinstance(l, BatchNormalization):
        l.trainable = False
# Deja solo las últimas ~100 capas entrenables (ajusta si hace falta)
for l in base_model.layers[:-100]:
    l.trainable = False



model.compile(
    optimizer=Adam(learning_rate=LR_FASE2, clipnorm=1.0),
    loss=[Huber(delta=1.5), CategoricalCrossentropy(label_smoothing=0.05)],
    metrics=[["mae"], ["accuracy"]],
    loss_weights=[1.0, 0.0],
)

cbs_f2 = [
    EarlyStopping(monitor="val_regression_mae", mode="min",
                  patience=30, min_delta=1e-4,   # << mucha paciencia para overnight
                  restore_best_weights=True, verbose=1),
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
    X_train, [y_reg_train, y_class_train],
    validation_data=(X_val, [y_reg_val, y_class_val]),
    epochs=EPOCHS_FASE2,
    batch_size=BATCH_SIZE,
    callbacks=cbs_f2,
    shuffle=True, verbose=1
)


# Chequeo de colapso tras Fase 2
pred_reg_z, _ = model.predict(X_val, batch_size=BATCH_SIZE)
print("STD pred z-space tras Fase 2 (altura_z, peso_z):", pred_reg_z.std(axis=0))

# ========================
# GUARDAR MODELO
# ========================
model.save("modelo_antropometrico.keras")
model.save("modelo_antropometrico.h5", include_optimizer=False)
print("✅ Guardado: .keras y .h5")

# ========================
# PREDICCIONES (VAL) + DESNORMALIZACIÓN
# ========================
scaler_y = joblib.load("scaler_medidas.pkl")
pred_reg, _ = model.predict(X_val, batch_size=BATCH_SIZE)
pred_reg = scaler_y.inverse_transform(pred_reg)  # <- ya en unidades reales

# y_reg_val está en z-space, así que SÍ hay que desescalarlo:
y_true = scaler_y.inverse_transform(y_reg_val)

# ❌ no vuelvas a hacer inverse_transform sobre pred_reg
y_pred = pred_reg  # ✅ usa tal cual

# tras obtener pred_reg (en reales) y y_true = inverse_transform(y_reg_val)
w_true = y_true[:,1]; w_pred = pred_reg[:,1]
A = np.vstack([w_pred, np.ones_like(w_pred)]).T
a,b = np.linalg.lstsq(A, w_true, rcond=None)[0]
pred_reg[:,1] = a*pred_reg[:,1] + b

mae_altura = np.mean(np.abs(y_true[:,0]-y_pred[:,0]))
mae_peso   = np.mean(np.abs(y_true[:,1]-y_pred[:,1]))

print(f"MAE altura: {mae_altura:.2f} m  |  MAE peso: {mae_peso:.1f} kg")

plt.figure(); plt.scatter(y_true[:,1], y_pred[:,1], s=8); plt.plot([40,160],[40,160])
plt.xlabel("Peso real"); plt.ylabel("Peso predicho"); plt.title("VAL peso")
plt.savefig("val_peso.png", dpi=140); plt.close()

plt.figure(); plt.scatter(y_true[:,0], y_pred[:,0], s=8); plt.plot([1.45,2.05],[1.45,2.05])
plt.xlabel("Altura real (m)"); plt.ylabel("Altura predicha (m)"); plt.title("VAL altura")
plt.savefig("val_altura.png", dpi=140); plt.close()



with open("predicciones_altura_peso.txt", "w", encoding="utf-8") as f:
    for fname, (alt, pes) in zip(val_filenames, pred_reg):
        f.write(f"Imagen: {fname} | Altura: {alt:.3f} m | Peso: {pes:.1f} kg\n")
print("📄 Predicciones guardadas en predicciones_altura_peso.txt")
