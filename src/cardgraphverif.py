import tensorflow as tf
print("GPUs detectadas:", tf.config.list_physical_devices('GPU'))
print("Usando:", "GPU" if tf.test.is_gpu_available() else "CPU")
