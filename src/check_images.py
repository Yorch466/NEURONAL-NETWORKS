import os

image_dir = "data/images"
images = os.listdir(image_dir)
print(f"📷 Total de imágenes en carpeta: {len(images)}")
print("Ejemplos:")
print(images[:10])

