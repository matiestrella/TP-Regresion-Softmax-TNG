# TP 1: Clasificación de texto con regresión softmax (20 Newsgroups)

Trabajo práctico de Redes Neuronales / Deep Learning. Clasifica los posts del dataset **20 Newsgroups** en sus 20 categorías con una **regresión softmax** (una sola capa lineal) hecha en PyTorch.

## Qué hay en la notebook

`01_Softmax_Regression_20Newsgroups.ipynb`:

1. **Preprocesamiento configurable** (NLTK): stopwords, stemming o lematización.
2. **Vectorización TF-IDF** (scikit-learn): n-gramas, tamaño del vocabulario y `min_df`.
3. **Modelo**: `nn.Linear` + `CrossEntropyLoss`, entrenado con Adam.
4. **Búsqueda de hiperparámetros** con Optuna (TPE con semilla, 40 trials), registrada en MLflow.
5. **Entrenamiento final** con un split de validación real (80/20 del train) y evaluación única en test.
6. **Análisis**: baselines (clase más frecuente y Naive Bayes), sesgo/varianza, importancia de hiperparámetros, experimentos controlados de learning rate y batch size con 3 semillas, tiempos de convergencia y prueba de estandarización.
7. Preguntas de reflexión.

## Cómo correrlo

**En Google Colab (recomendado):** abrir la notebook desde GitHub (`File > Open notebook > GitHub`) y hacer `Runtime > Run all`. La primera celda instala lo que falta. Tarda unos 30 minutos en CPU (la parte más lenta es el preprocesamiento y la búsqueda de Optuna).

**En local:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter
jupyter notebook 01_Softmax_Regression_20Newsgroups.ipynb
```

El dataset se descarga solo con `fetch_20newsgroups` y los recursos de NLTK con `nltk.download`.

## Reproducibilidad

Todo usa la semilla `SEED = 42` (sampler de Optuna, inicialización de pesos, orden de los batches y split de validación). Los experimentos controlados usan además las semillas 0, 1 y 2 para medir la variación entre corridas.

## Ver los registros

- **TensorBoard**: la notebook lo abre con `%tensorboard --logdir runs`.
- **MLflow**: después de correr, `mlflow ui` en la misma carpeta muestra cada trial, cada experimento controlado y el modelo final.
