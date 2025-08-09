import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# ========================
# CONFIG
# ========================
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
IMG_DIR = "data/images/datasets/datasets/ALLIMAGES"  # tu carpeta única con todas las imágenes
CSV_PATH = "data/2DImage2BMI/ALL_feature/Image_train.csv"

# ========================
# CARGA DE CSV Y ETIQUETAS
# ========================
df = pd.read_csv(CSV_PATH, header=None)

# Columnas: [imagen, ..., PB(col 3), PP(col 4), ..., altura(-2), peso(-1)]
df.columns = ["img_name"] + [f"col_{i}" for i in range(1, df.shape[1])]

# Variables antropométricas
df["altura"] = df.iloc[:, -2]
df["pb"] = df.iloc[:, 3]
df["pp"] = df.iloc[:, 4]

# Calcular IMC y asignar clase
df["peso"] = df.iloc[:, -1]
df["imc"] = df["peso"] / (df["altura"] ** 2)
df["clase"] = df["imc"].apply(lambda x: 0 if x < 18.5 else (1 if x < 25 else 2))

# ========================
# CARGA DE IMÁGENES
# ========================
def load_image(file):
    path = os.path.join(IMG_DIR, file)
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

print("📷 Cargando imágenes...")
images = np.array([load_image(fname) for fname in df["img_name"]])
X = images

# Salidas
y_reg = df[["altura", "pb", "pp"]].values
y_class = tf.keras.utils.to_categorical(df["clase"].values, num_classes=3)

# ========================
# DIVISIÓN TRAIN / VAL
# ========================
X_train, X_val, y_reg_train, y_reg_val, y_class_train, y_class_val = train_test_split(
    X, y_reg, y_class, test_size=0.2, random_state=42
)

# ========================
# MODELO
# ========================
base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base_model.output)

salida_reg = Dense(3, activation="linear", name="regression")(x)
salida_class = Dense(3, activation="softmax", name="classification")(x)

model = Model(inputs=base_model.input, outputs=[salida_reg, salida_class])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss={"regression": "mse", "classification": "categorical_crossentropy"},
    metrics={"regression": "mae", "classification": "accuracy"}
)

model.summary()

# ========================
# ENTRENAR
# ========================
print("🚀 Entrenando modelo...")
history = model.fit(
    X_train,
    {"regression": y_reg_train, "classification": y_class_train},
    validation_data=(X_val, {"regression": y_reg_val, "classification": y_class_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ========================
# GUARDAR MODELO Y SALIDA
# ========================
model.save("modelo_antropometrico.h5")
print("✅ Modelo guardado como modelo_antropometrico.h5")

# Predicciones y salida en archivo
preds = model.predict(X_val)
pred_reg = preds[0]
pred_class = np.argmax(preds[1], axis=1)

with open("predicciones.txt", "w") as f:
    for i in range(len(pred_reg)):
        f.write(
            f"Imagen: {df['img_name'].iloc[i]} | Altura: {pred_reg[i][0]:.2f} m | PB: {pred_reg[i][1]:.2f} cm | PP: {pred_reg[i][2]:.2f} cm | Clase: {pred_class[i]}\n"
        )

print("📄 Predicciones guardadas en predicciones.txt")
