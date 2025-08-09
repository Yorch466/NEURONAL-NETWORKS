import os
import time
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

print("GPUs detectadas:", tf.config.list_physical_devices("GPU"))

# Pequeño modelo para prueba
model = models.Sequential([
    layers.Input((224,224,3)),
    layers.Conv2D(64, 3, activation="relu"),
    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPool2D(),
    layers.Conv2D(128, 3, activation="relu"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(3, activation="softmax")
])
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Datos de prueba
X = np.random.rand(256,224,224,3).astype("float32")
y = tf.one_hot(np.random.randint(0,3, size=(256,)), 3)

# Warmup
model.fit(X, y, batch_size=16, epochs=1, verbose=0)

# Medición
t0 = time.time()
model.fit(X, y, batch_size=16, epochs=3, verbose=0)
print(f"Tiempo total: {time.time()-t0:.2f} s")
