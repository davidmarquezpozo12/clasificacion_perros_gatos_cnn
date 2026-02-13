import matplotlib.pyplot as plt

def evaluar_modelo(history):
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.legend()
    plt.title("Precisión del modelo")
    plt.show()
