import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar modelo entrenado
model_path = os.path.join("models", "cnn_model.h5")
model = tf.keras.models.load_model(model_path)

# Ruta de imagen de prueba
img_path = os.path.join("data", "images", "persona1.jpg")  # cambiá esto si querés otra imagen

# Preprocesar imagen
img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predicción
pred = model.predict(img)
altura, pb, pp = pred[0]

print(f"📏 Altura estimada: {altura:.2f} cm")
print(f"💪 PB estimado: {pb:.2f} cm")
print(f"🦵 PP estimado: {pp:.2f} cm")

img = cv2.imread("dara/images/personal.jpg")
img_rgb= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.tittle("Imagen cargada")
plt.axis("off")
plt.show()