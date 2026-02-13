# Laboratorio: Clasificación de Perros y Gatos con CNN
## 1. Objetivo
El objetivo de este laboratorio es aprender a clasificar imágenes de perros y gatos utilizando **redes neuronales convolucionales
(CNN)** con TensorFlow.
Los alumnos aprenderán a:
- Preparar imágenes para la red.
- Construir y entrenar una CNN.
- Evaluar el rendimiento del modelo.
- Realizar predicciones con nuevas imágenes.
## 2. Librerías utilizadas
- **TensorFlow**: Para crear y entrenar la red neuronal.
- **NumPy**: Para manejar matrices y normalizar las imágenes.
- **Matplotlib**: Para visualizar la evolución del entrenamiento (precisión y pérdida).
## 3. Estructura del proyecto
clasificacion_perros_gatos_cnn
├── datos/
│ ├── perros/ # Imágenes de perros
│ └── gatos/ # Imágenes de gatos
├── pruebas/ # Nuevas imágenes para probar el modelo
├── src/
│ ├── preprocesamiento.py # Funciones para preparar imágenes
│ ├── modelo.py # Función para crear la CNN
│ ├── evaluacion.py # Funciones para mostrar resultados
│ └── main.py # Código principal de entrenamiento y prueba
├── requirements.txt # Librerías necesarias
└── README.md # Este archivo
## 4. Instalación
1. Clonar o descargar el proyecto.
2. Instalar las librerías necesarias:
```bash
pip install tensorflow numpy matplotlib
## 5. Uso
-Colocar imágenes de entrenamiento en las carpetas datos/perros y datos/gatos.
-Colocar imágenes de prueba en la carpeta pruebas.
-Ejecutar el laboratorio: python src/main.py
-Seguir las instrucciones para probar nuevas imágenes.
• Escriba el nombre del archivo de la imagen o salir para terminar.
## 6. Notas importantes
• Las imágenes se redimensionan automáticamente a 128x128 y se normalizan.
• La CNN aprende automáticamente las características importantes (bordes, texturas, formas), no necesita preprocesamiento
adicional como grises, blur o bordes.
• La evaluación del modelo se realiza con datos de validación y pruebas para comprobar su precisión