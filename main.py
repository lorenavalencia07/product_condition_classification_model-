"""
main.py

Este script ejecuta el flujo completo del proyecto:
  1. Carga de datos desde un archivo JSON Lines usando DataLoader.
  2. Preprocesamiento de los datos con DataPreprocessor.
  3. Conversión del target 'condition' (de 'new' y 'used' a 1 y 0).
  4. Selección de las columnas de entrenamiento (cargadas desde un archivo YAML).
  5. Entrenamiento simple de varios modelos de clasificación.
  6. Selección del modelo con mejor métrica F1.
  7. Ajuste de hiperparámetros (GridSearchCV) para el mejor modelo.
  8. Guardado del modelo ajustado en un archivo .pkl para predicciones futuras.

Requisitos:
  - Las clases DataLoader, DataPreprocessor y EntrenadorModelos deben estar definidas en src.
  - El archivo data/training_columns.yaml debe existir con la lista de columnas de entrenamiento.
  - Se utiliza joblib para persistir el modelo.
"""

# Importar clases definidas en src y utilidades
from src.data_loader import DataLoader
from src.data_processor import DataPreprocessor
from src.train_model import EntrenadorModelos
from utils.load_training_columns import load_training_columns

# Importar librerías externas
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
import warnings
import joblib

# Suprimir warnings innecesarios
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)
warnings.filterwarnings("ignore",
                        message='Parameters: { "use_label_encoder" } are not used.',
                        category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        module=r"xgboost\.core")


def main():
    """
    Flujo principal:
      - Carga y procesamiento de datos.
      - Conversión del target a numérico.
      - Selección de features a partir del archivo YAML.
      - Entrenamiento simple de varios modelos de clasificación.
      - Selección del modelo con mejor F1 y ajuste de hiperparámetros.
      - Guardado del modelo ajustado para predicción futura.
    """
    # Rutas de archivos de entrada
    input_file = "data/MLA_100k_checked_v3.jsonlines"
    schema_file = "data/dataset_schema.json"

    # 1. Cargar datos
    print("Loading data...")
    loader = DataLoader(input_file, schema_file)
    df = loader.load_data()
    print(f"Data loaded: {df.shape[0]} records, {df.shape[1]} columns")

    # 2. Procesar datos
    print("\nProcessing data...")
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.preprocess(df)
    print(
        f"Data processed: {df_processed.shape[0]} records, {df_processed.shape[1]} columns")

    # 3. Convertir el target 'condition' a numérico (new:1, used:0)
    print("\nConverting target column...")
    df_processed['condition'] = df_processed['condition'].map(
        {'new': 1, 'used': 0})
    df_processed = df_processed.dropna(subset=['condition'])

    # 4. Seleccionar las columnas de entrenamiento desde YAML
    features_columns = load_training_columns("data/training_columns.yaml")

    # 5. Inicializar la clase de entrenamiento y preparar datos
    print("\nPreparing model trainer...")
    trainer = EntrenadorModelos(df_processed, target_column='condition',
                                verbose=True)
    conjunto = trainer.preparar_datos(columnas_seleccionadas=features_columns,
                                      stratify=True)

    # 6. Entrenar modelos simples de clasificación
    modelos_a_entrenar = ['LogisticRegression', 'RandomForest',
                          'GradientBoosting', 'XGBoost']
    results = {}
    for model_name in modelos_a_entrenar:
        if model_name == 'LogisticRegression':
            model_instance = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'RandomForest':
            model_instance = RandomForestClassifier(random_state=42)
        elif model_name == 'GradientBoosting':
            model_instance = GradientBoostingClassifier(random_state=42)
        elif model_name == 'XGBoost':
            model_instance = xgb.XGBClassifier(random_state=42,
                                               use_label_encoder=False,
                                               eval_metric='logloss')
        else:
            continue

        print(f"\nTraining model: {model_name}")
        id_model, metrics = trainer.entrenar(model_instance, model_name)
        results[model_name] = metrics

    # 7. Seleccionar el modelo con mejor F1 score de los entrenados
    best_model_id = None
    best_f1 = 0
    for model_id, metrics in trainer.resultados.items():
        if metrics.get('f1', 0) > best_f1:
            best_f1 = metrics.get('f1', 0)
            best_model_id = model_id
    print(
        f"\nBest model from simple training: {best_model_id} with F1: {best_f1:.4f}")

    # 8. Ajuste de hiperparámetros para el mejor modelo usando GridSearchCV
    param_grids = {
        'LogisticRegression': {
            'classifier__C': [0.1, 1.0, 10.0],
            'classifier__penalty': ['l2']
        },
        'RandomForest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        },
        'GradientBoosting': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [3, 5, 7]
        }
    }

    if best_model_id is not None:
        base_model = best_model_id.split('_')[0]
        best_param_grid = param_grids.get(base_model, None)
        if best_param_grid:
            print(f"\nPerforming hyperparameter tuning for {base_model}...")
            if base_model == 'LogisticRegression':
                tuned_model = LogisticRegression(random_state=42,
                                                 max_iter=1000)
            elif base_model == 'RandomForest':
                tuned_model = RandomForestClassifier(random_state=42)
            elif base_model == 'GradientBoosting':
                tuned_model = GradientBoostingClassifier(random_state=42)
            elif base_model == 'XGBoost':
                tuned_model = xgb.XGBClassifier(random_state=42,
                                                use_label_encoder=False,
                                                eval_metric='logloss')
            else:
                tuned_model = None

            if tuned_model is not None:
                id_tuned, tuned_metrics = trainer.entrenar_con_grid(
                    tuned_model, base_model, param_grid=best_param_grid)
                print(f"\nTuned model: {id_tuned}")
                for k, v in tuned_metrics.items():
                    if k != 'confusion_matrix':
                        print(f"  {k}: {v:.4f}")

                # 9. Guardar el mejor modelo ajustado para predicción
                model_save_path = "data/article_condition_predictor.pkl"
                joblib.dump(trainer.modelos[id_tuned], model_save_path)
                print(f"\nBest tuned model saved at: {model_save_path}")
        else:
            print(f"No hyperparameter grid defined for model {base_model}.")


if __name__ == "__main__":
    main()
