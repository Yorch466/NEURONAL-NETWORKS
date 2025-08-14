import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf # from tensorflow.keras.utils import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split # from tensorflow.keras.optimizers import Adam
from cnn_medidas import build_model

# Rutas
CSV_PATH = os.path.join("data", "annotations.csv")
IMAGE_DIR = os.path.join("data", "images")

# Cargar el CSV
df = pd.read_csv(CSV_PATH)

# Variables
X = []
y = []

# Cargar imágenes y etiquetas
for i, row in df.iterrows():
    filename = row["filename"]
    img_path = os.path.join(IMAGE_DIR, filename)

    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0  # Normalización
        X.append(img)
        y.append([row["altura_cm"], row["pb_cm"], row["pp_cm"]])
    else:
        print(f"❌ Imagen no encontrada: {filename}")

# Convertir a arrays
X = np.array(X)
y = np.array(y)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Construir modelo
model = build_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Entrenar
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=4)

# Guardar modelo
os.makedirs("models", exist_ok=True)
model.save(os.path.join("models", "cnn_model.h5"))
print("✅ Modelo guardado en models/cnn_model.h5")

