
print("DirectML plugin import: OK")

import tensorflow as tf
print("TF:", tf.__version__)
print("DML devices:", tf.config.list_physical_devices('DML'))

# Fuerza registro del backend DirectML
try:
    import tensorflow_directml_plugin  # <-- NO es tensorflow_directml
except Exception as e:
    print("Aviso: no se pudo cargar DirectML:", e)

import tensorflow as tf
print("TF:", tf.__version__)
print("DML devices:", tf.config.list_physical_devices('DML'))
