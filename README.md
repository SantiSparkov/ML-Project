Clasificación de Géneros de Videojuegos con Machine Learning

Este proyecto de machine learning, desarrollado en Python, se centra en la predicción de géneros de videojuegos a partir de datos textuales y numéricos. Se implementó un pipeline completo que abarca desde el preprocesamiento de datos hasta la evaluación y optimización de modelos, aplicando diversas técnicas de aprendizaje automático y redes neuronales.

Tecnologías Principales

Lenguaje y Frameworks
	•	Python
	•	Scikit-learn para modelos tradicionales de Machine Learning
	•	Keras/TensorFlow para redes neuronales
	•	Weights & Biases (wandb) para el seguimiento de experimentos
	•	Matplotlib y Seaborn para visualización de datos

Procesamiento de Datos
	•	Pandas y NumPy para manipulación y análisis de datos
	•	TF-IDF Vectorizer y Tokenization para procesamiento de texto
	•	GridSearchCV para optimización de hiperparámetros
	•	Validación cruzada para mejorar la precisión de los modelos

Modelos de Machine Learning Implementados
	•	Random Forest
	•	Gradient Boosting
	•	Redes Neuronales (Deep Learning con Keras/TensorFlow)
	•	Regresión Logística

Estructura del Proyecto
	•	Obligatorio2024.ipynb: Notebook principal con el desarrollo del proyecto
	•	red_neuronal.ipynb: Implementación específica de redes neuronales
	•	game_genre_model.h5: Modelo entrenado de red neuronal
	•	best_gradient_boosting_model.joblib: Modelo optimizado de Gradient Boosting
	•	best_random_forest_model.joblib: Modelo optimizado de Random Forest
	•	dataset/: Datos de entrenamiento y prueba
	•	wandb/: Seguimiento de experimentos
	•	metrics/: Métricas de evaluación de los modelos
	•	predictions/: Predicciones generadas
	•	app.py: Aplicación para servir el modelo

Este proyecto combina técnicas clásicas y modernas de machine learning para realizar una clasificación precisa de géneros de videojuegos, aplicando buenas prácticas de preprocesamiento, optimización y validación de modelos.
