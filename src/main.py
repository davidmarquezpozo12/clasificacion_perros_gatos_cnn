import tensorflow as tf
import os
from modelo import crear_modelo
from preprocesamiento import preprocesar_imagen
from evaluacion import evaluar_modelo
from keras.callbacks import EarlyStopping 

train_ds = tf.keras.utils.image_dataset_from_directory(
    "./datos",
    validation_split=0.3,
    subset="training",
    seed=42,
    image_size=(128,128),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "./datos",
    validation_split=0.3,
    subset="validation",
    seed=42,
    image_size=(128,128),
    batch_size=32
)

train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
modelo = crear_modelo()
history = modelo.fit(train_ds, validation_data=val_ds, epochs=15, callbacks=[early_stop])
evaluar_modelo(history)

# Predicción manual
while True:
    nombre = input("Introduce imagen o 'salir': ")
    if nombre == "salir": break
    img = preprocesar_imagen(f"./pruebas/{nombre}")
    if img is not None:
        img = tf.expand_dims(img, axis=0)
        pred = modelo.predict(img)
        if pred[0][0] > 0.5:
            print("Es un PERRO")
        else:
            print("Es un GATO")
