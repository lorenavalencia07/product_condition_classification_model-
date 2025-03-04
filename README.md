# Product condition classification model

Este proyecto tiene como objetivo entrenar y ajustar modelos de machine learning para predecir si un artículo es **nuevo** o **usado** (target: `condition`). Se utiliza un pipeline modular que abarca la carga y procesamiento de datos, el entrenamiento de modelos y, posteriormente, la generación de predicciones sobre nuevos datos.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
  - [Entrenamiento](#entrenamiento)
  - [Predicciones](#predicciones)
- [Notas Adicionales](#notas-adicionales)
- [Licencia](#licencia)

## Descripción

El flujo completo del proyecto incluye:

1. **Carga de Datos:** Se leen los datos de entrenamiento desde un archivo JSON Lines usando la clase `DataLoader`, siguiendo el esquema definido en `dataset_schema.json`.
2. **Procesamiento de Datos:** Los datos se preprocesan y aplanan mediante la clase `DataPreprocessor`.
3. **Conversión del Target:** La columna `condition` se convierte a valores numéricos (nuevo: 1, usado: 0).
4. **Selección de Features:** Las columnas de entrenamiento se definen en un archivo YAML (`training_columns.yaml`), lo que centraliza la selección de variables.
5. **Entrenamiento y Ajuste de Hiperparámetros:** Se entrenan distintos modelos (Logistic Regression, Random Forest, Gradient Boosting y XGBoost) mediante la clase `EntrenadorModelos`. Se selecciona el mejor modelo (según F1 score) y se realiza ajuste de hiperparámetros utilizando GridSearchCV.
6. **Guardado del Modelo:** El mejor modelo ajustado se guarda en un archivo `.pkl` (por ejemplo, `data/article_condition_predictor.pkl`) para su uso futuro.
7. **Predicciones:** La clase `ModelPredictor` permite cargar nuevos datos desde un archivo JSON Lines (con el mismo esquema de los datos de entrenamiento), procesarlos y generar predicciones utilizando el modelo guardado.

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/article_condition_classification_meli.git

2. Crea y activa entorno virtual
   ```bash
   python -m venv venv
   
- En Windows:
  ```bash
  venv\Scripts\activate
  
- En macOS/Linux:
  ```bash
  source venv/bin/activate

3. Instalar las dependencias
   ```bash
   pip install -r requirements.txt

## Uso
### Entrenamiento
    Para entrenar los modelos y ajustar el mejor modelo, ejecuta el script principal:

    El flujo realiza lo siguiente:

    - Carga y preprocesa los datos.
    - Convierte el target a valores numéricos.
    - Selecciona las columnas de entrenamiento definidas en data/training_columns.yaml.
    - Entrena varios modelos y selecciona el mejor según la métrica F1.
    - Ajusta hiperparámetros del mejor modelo utilizando GridSearchCV.
    - Guarda el modelo ajustado en data/article_condition_predictor.pkl.

### Predicciones
    Una vez guardado el modelo, puedes realizar predicciones sobre nuevos datos (en formato JSON Lines) utilizando la clase ModelPredictor del script model_predictor.py.
    Ejemplo de uso:

    python model_predictor.py

    El script:

    - Carga el modelo guardado (data/article_condition_predictor.pkl).
    - Carga nuevos datos desde un archivo JSON Lines (por ejemplo, data/new_data.jsonlines) utilizando el esquema definido en data/dataset_schema.json.
    - Procesa los nuevos datos.
    - Selecciona las columnas definidas en data/training_columns.yaml.
    - Retorna las predicciones.
    
## Notas Adicionales
    - Warnings: Se han configurado filtros para suprimir warnings no críticos (por ejemplo, de XGBoost).
    - Preprocesamiento: Es fundamental que los nuevos datos tengan el mismo formato y esquema que los datos de entrenamiento.
    - Personalización: Puedes ajustar las grillas de hiperparámetros o modificar el pipeline de preprocesamiento según las necesidades de tu proyecto.
