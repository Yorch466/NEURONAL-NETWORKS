from scipy.io import loadmat
import os

# Ruta al archivo .mat
mat_path = os.path.join("data", "mpii_human_pose_v1_u12_2", "mpii_human_pose_v1_u12_2","mpii_human_pose_v1_u12_1.mat")

# Cargar estructura
data = loadmat(mat_path)

# Acceder al campo principal
release = data['RELEASE']
release = data['RELEASE']
print(type(release))
print(release.dtype)

annolist = release[0][0]['annolist']
print(f"Cantidad de imágenes: {len(annolist)}")

img_name = annolist[0][0]['image']['name'][0]
print(f"📷 Imagen 0: {img_name}")

print(f"🔍 Claves del archivo .mat: {list(data.keys())}")
