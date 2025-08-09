import os, sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib

print("exe:", sys.executable)
print("TF:", tf.__version__)
print("Dispositivos CPU/GPU visibles:", tf.config.list_physical_devices())

# ========================
# CONFIG
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
IMG_DIR = "data/images/datasets/datasets/ALLIMAGES"
CSV_PATH = "data/2DImage2BMI/ALL_feature/Image_train.csv"
LR = 1e-4

# ========================
# CARGA DE CSV Y ETIQUETAS
# ========================
df = pd.read_csv(CSV_PATH, header=None)
df.columns = ["img_name"] + [f"col_{i}" for i in range(1, df.shape[1])]

df["altura"] = df.iloc[:, -2].astype("float32")
df["pb"]     = df.iloc[:, 3].astype("float32")
df["pp"]     = df.iloc[:, 4].astype("float32")
df["peso"]   = df.iloc[:, -1].astype("float32")

df["imc"]   = df["peso"] / (df["altura"] ** 2)
df["clase"] = df["imc"].apply(lambda x: 0 if x < 18.5 else (1 if x < 25 else 2)).astype(int)

# ========================
# CARGA DE IMÁGENES
# ========================
def load_image(file):
    path = os.path.join(IMG_DIR, file)
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

print("📷 Cargando imágenes...")
X = np.stack([load_image(fname) for fname in df["img_name"]], axis=0).astype("float32")

y_reg   = df[["altura", "pb", "pp"]].values.astype("float32")
y_class = tf.keras.utils.to_categorical(df["clase"].values, num_classes=3).astype("float32")

# ========================
# TRAIN / VAL (estratificado)
# ========================
train_idx, val_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=42, stratify=df["clase"].values
)
X_train, X_val = X[train_idx], X[val_idx]
y_reg_train, y_reg_val = y_reg[train_idx], y_reg[val_idx]
y_class_train, y_class_val = y_class[train_idx], y_class[val_idx]
train_filenames = df["img_name"].iloc[train_idx].tolist()
val_filenames   = df["img_name"].iloc[val_idx].tolist()

# ========================
# Normalización de regresión
# ========================
scaler_y = StandardScaler()
y_reg_train = scaler_y.fit_transform(y_reg_train)
y_reg_val   = scaler_y.transform(y_reg_val)
joblib.dump(scaler_y, "scaler_medidas.pkl")
print("📏 Escalador guardado en scaler_medidas.pkl")

# ========================
# Balanceo por clases -> sample_weight
# ========================
clases_train = np.argmax(y_class_train, axis=1)
presentes = np.unique(clases_train)
pesos_presentes = compute_class_weight(
    class_weight='balanced',
    classes=presentes,
    y=clases_train
)
class_weight_dict = {int(c): float(w) for c, w in zip(presentes, pesos_presentes)}
for k in ({0,1,2} - set(presentes)):
    class_weight_dict[k] = 1.0

w_class_train = np.array([class_weight_dict[int(c)] for c in clases_train], dtype="float32")
w_reg_train = np.ones(len(X_train), dtype="float32")

print("📊 Distribución TRAIN:", {int(k): int(v) for k, v in zip(*np.unique(clases_train, return_counts=True))})
print("⚖️ Pesos de clase:", class_weight_dict)

# ========================
# MODELO
# ========================
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)

salida_reg = Dense(3, activation="linear",  name="regression")(x)
salida_cls = Dense(3, activation="softmax", name="classification")(x)

model = Model(inputs=base_model.input, outputs=[salida_reg, salida_cls])

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=["mse", "categorical_crossentropy"],
    metrics=[[], ["accuracy"]],
    loss_weights=[0.2, 1.0]
)

model.summary()

# ========================
# CALLBACKS
# ========================
cbs = [
    EarlyStopping(monitor="val_classification_accuracy", mode="max",
                  patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.keras", monitor="val_classification_accuracy",
                    mode="max", save_best_only=True, save_weights_only=False, verbose=1),
]

# ========================
# ENTRENAR
# ========================
print("🚀 Entrenando modelo (CPU)...")
history = model.fit(
    X_train,
    [y_reg_train, y_class_train],
    validation_data=(X_val, [y_reg_val, y_class_val]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sample_weight=[w_reg_train, w_class_train],
    callbacks=cbs,
    shuffle=True,
    verbose=1
)

# ========================
# GUARDAR MODELO
# ========================
model.save("modelo_antropometrico.keras")
model.save("modelo_antropometrico.h5", include_optimizer=False)
print("✅ Guardado: .keras y .h5")

# ========================
# PREDICCIONES (VAL) + Desnormalización
# ========================
scaler_y = joblib.load("scaler_medidas.pkl")
pred_reg, pred_class_prob = model.predict(X_val, batch_size=BATCH_SIZE)
pred_reg = scaler_y.inverse_transform(pred_reg)  # volver a valores reales
pred_class = np.argmax(pred_class_prob, axis=1)

etiquetas = {0: "Flaco", 1: "Normal", 2: "Sobrepeso"}
with open("predicciones.txt", "w", encoding="utf-8") as f:
    for fname, reg, cls in zip(val_filenames, pred_reg, pred_class):
        f.write(
            f"Imagen: {fname} | Altura: {reg[0]:.2f} m | PB: {reg[1]:.2f} cm | PP: {reg[2]:.2f} cm | Clase: {etiquetas[cls]}\n"
        )
print("📄 Predicciones guardadas en predicciones.txt")
