<img width="103" height="20" alt="image" src="https://github.com/user-attachments/assets/c0eddba6-3915-4730-81b9-ba7136df2889" />

<img width="115" height="20" alt="image" src="https://github.com/user-attachments/assets/23fed70b-ed6d-4824-836c-932bafa150f6" />

<img width="125" height="20" alt="image" src="https://github.com/user-attachments/assets/7188be8d-fe16-4dd4-b557-ccc8c041ab6a" />

<img width="82" height="20" alt="image" src="https://github.com/user-attachments/assets/f363ffb5-7948-4c3c-bb45-606104ca0561" />

📌 Descripción
Este proyecto implementa un dashboard interactivo para resolver un problema real de deserción escolar en instituciones educativas, utilizando técnicas de aprendizaje supervisado. El sistema permite cargar datos, preprocesarlos, entrenar múltiples modelos, evaluar su rendimiento y realizar predicciones en nuevos estudiantes — todo desde una interfaz web intuitiva.

El dashboard está construido con Streamlit y está diseñado para ser usado en contextos académicos, como parte de un curso de Machine Learning o Ciencia de Datos.

🎯 Objetivo
Predecir el estado académico final de un estudiante universitario a partir de sus características demográficas, académicas y socioeconómicas, con tres posibles resultados:

Graduate: El estudiante se graduó ✅
Dropout: El estudiante abandonó sus estudios ❌
Enrolled: El estudiante sigue inscrito ⏳
Este es un problema de clasificación multiclase, y el modelo más adecuado es la Regresión Logística, por su interpretabilidad y buen rendimiento en datos educativos.

📊 Dataset utilizado
Nombre: Student Dropout and Academic Success Prediction
Fuente: Kaggle
Tamaño: 4,424 estudiantes
Variables: 34 características + 1 variable objetivo (Target)
Variables clave:
Rendimiento académico (notas por semestre)
Asistencia y matrícula
Becas y situación financiera
Edad, género, nacionalidad
Curso y modo de admisión
✅ El dataset no contiene valores faltantes, lo que facilita el preprocesamiento. 

🧠 Modelos implementados
El dashboard soporta los siguientes algoritmos:

Para clasificación (como este caso):
Regresión Logística (recomendado)
Árbol de Decisión (CART)
Random Forest
K-Nearest Neighbors (KNN)
Red Neuronal (MLP)
SVM
Gradient Boosting
Para regresión (en otros datasets):
Regresión Lineal
Ridge
Lasso
Árbol de Regresión
SVR
Random Forest Regressor
MLP Regressor
🛠️ Funcionalidades principales
Carga de datos: Sube cualquier archivo CSV.
Preprocesamiento inteligente:
Selección de columnas
Manejo de valores faltantes (imputación o eliminación)
Codificación categórica (One-Hot o binaria)
Escalado numérico (estandarización o Min-Max)
Entrenamiento de modelos: Con ajuste de hiperparámetros.
Evaluación completa:
Métricas por clase (precisión, recall, F1)
Matriz de confusión
Importancia de características
Predicciones detalladas sobre el conjunto de prueba
Predicción en nuevos datos: Manual o por archivo CSV.
Descargas en CSV:
Datos preprocesados
Métricas del modelo
Predicciones del conjunto de prueba
Predicciones en nuevos datos
