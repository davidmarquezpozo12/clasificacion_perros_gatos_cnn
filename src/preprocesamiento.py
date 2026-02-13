import tensorflow as tf
import os

IMG_SIZE = (128, 128)

def preprocesar_imagen(ruta):
    """
    - Lee una imagen desde disco
    - La redimensiona
    - Normaliza los píxeles (0-1)
    """
    if not os.path.exists(ruta):
        print("Archivo no encontrado")
        return None
    
    img = tf.keras.utils.load_img(ruta, target_size=IMG_SIZE)
    img = tf.keras.utils.img_to_array(img)
    img = img / 255.0
    return img
