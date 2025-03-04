import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score, log_loss, confusion_matrix


class EntrenadorModelos:
    """
    Clase para entrenar y evaluar un modelo de clasificación que predice si un producto es nuevo (1) o usado (0).

    La clase permite preparar los datos, entrenar el modelo de forma simple o con ajuste de hiperparámetros,
    evaluar las métricas y realizar predicciones.
    """

    def __init__(self, df: pd.DataFrame, target_column: str = 'condition',
                 test_size: float = 0.2, random_state: int = 42,
                 verbose: bool = True):
        """
        Inicializa la clase con el DataFrame y configuraciones básicas.

        Args:
            df (pd.DataFrame): DataFrame con los datos.
            target_column (str): Nombre de la columna objetivo.
            test_size (float): Proporción del conjunto de prueba.
            random_state (int): Semilla aleatoria para reproducibilidad.
            verbose (bool): Si se muestran mensajes de progreso.
        """
        self.df = df
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose

        self.modelos = {}  # Almacena los pipelines entrenados
        self.resultados = {}  # Almacena las métricas de evaluación
        self.conjuntos_variables = {}  # Almacena la información de la división de datos

        if self.verbose:
            print(
                f"Dataset con {len(df)} registros y {len(df.columns)} columnas.")

    def preparar_datos(self, columnas_seleccionadas: list = None,
                       stratify: bool = True,
                       nombre_conjunto: str = None) -> str:
        """
        Prepara los datos dividiendo en entrenamiento y prueba, e identifica columnas numéricas y categóricas.

        Args:
            columnas_seleccionadas (list): Lista de columnas a utilizar como features.
            stratify (bool): Si se debe estratificar según la variable objetivo.
            nombre_conjunto (str): Nombre para identificar este conjunto de variables.

        Returns:
            str: Nombre del conjunto de variables creado.
        """
        if columnas_seleccionadas is None:
            # Se usan todas las columnas excepto la objetivo
            columnas_seleccionadas = [col for col in self.df.columns if
                                      col != self.target_column]

        X = self.df[columnas_seleccionadas]
        y = self.df[self.target_column]

        stratify_param = y if stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state,
            stratify=stratify_param
        )

        self.columnas_numericas = X.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        self.columnas_categoricas = X.select_dtypes(
            include=['object', 'category', 'bool']).columns.tolist()

        if nombre_conjunto is None:
            nombre_conjunto = f"conjunto_{len(self.conjuntos_variables) + 1}"

        self.conjuntos_variables[nombre_conjunto] = {
            'columnas': columnas_seleccionadas,
            'columnas_numericas': self.columnas_numericas,
            'columnas_categoricas': self.columnas_categoricas,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }

        if self.verbose:
            print(
                f"Datos preparados para el conjunto '{nombre_conjunto}' usando {len(columnas_seleccionadas)} columnas.")
            print(f"  Numéricas: {self.columnas_numericas}")
            print(f"  Categóricas: {self.columnas_categoricas}")
            print(
                f"  X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")

        return nombre_conjunto

    def _crear_pipeline(self, columnas_numericas: list,
                        columnas_categoricas: list,
                        estrategia_imputacion_num: str = 'median',
                        estrategia_imputacion_cat: str = 'most_frequent',
                        usar_knn: bool = False, n_neighbors: int = 5,
                        encoding_method: str = 'onehot') -> ColumnTransformer:
        """
        Crea un pipeline de preprocesamiento para columnas numéricas y categóricas.

        Args:
            columnas_numericas (list): Lista de columnas numéricas.
            columnas_categoricas (list): Lista de columnas categóricas.
            estrategia_imputacion_num (str): Estrategia para imputación numérica.
            estrategia_imputacion_cat (str): Estrategia para imputación categórica.
            usar_knn (bool): Si se utiliza KNNImputer para numéricos.
            n_neighbors (int): Número de vecinos para KNNImputer.
            encoding_method (str): Método de encoding para categóricas ('onehot' o 'ordinal').

        Returns:
            ColumnTransformer: Transformador de columnas configurado.
        """
        # Pipeline para columnas numéricas
        if usar_knn:
            numeric_transformer = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=n_neighbors)),
                ('scaler', StandardScaler())
            ])
        else:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=estrategia_imputacion_num)),
                ('scaler', StandardScaler())
            ])

        # Pipeline para columnas categóricas
        if encoding_method == 'onehot':
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=estrategia_imputacion_cat)),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=estrategia_imputacion_cat)),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value',
                                           unknown_value=-1))
            ])

        transformers = []
        if columnas_numericas:
            transformers.append(
                ('num', numeric_transformer, columnas_numericas))
        if columnas_categoricas:
            transformers.append(
                ('cat', categorical_transformer, columnas_categoricas))

        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor

    def _calcular_metricas(self, y_test: pd.Series, y_pred: np.ndarray,
                           y_prob: np.ndarray = None) -> dict:
        """
        Calcula las métricas de evaluación a partir de las predicciones.

        Args:
            y_test (pd.Series): Valores reales.
            y_pred (np.ndarray): Predicciones del modelo.
            y_prob (np.ndarray, optional): Probabilidades predichas (si están disponibles).

        Returns:
            dict: Diccionario con las métricas calculadas.
        """
        metricas = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        if y_prob is not None:
            metricas['roc_auc'] = roc_auc_score(y_test, y_prob)
            metricas['logloss'] = log_loss(y_test, y_prob)
        metricas['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        return metricas

    def entrenar(self, modelo, nombre_modelo: str,
                 conjunto_variables: str = None,
                 estrategia_imputacion_num: str = 'median',
                 estrategia_imputacion_cat: str = 'most_frequent',
                 usar_knn: bool = False, n_neighbors: int = 5,
                 encoding_method: str = 'onehot') -> tuple:
        """
        Entrena un modelo sin ajuste de hiperparámetros.

        Args:
            modelo: Instancia del modelo a entrenar.
            nombre_modelo (str): Nombre identificativo del modelo.
            conjunto_variables (str): Nombre del conjunto de variables a usar.
            estrategia_imputacion_num (str): Estrategia para imputación numérica.
            estrategia_imputacion_cat (str): Estrategia para imputación categórica.
            usar_knn (bool): Si se usa KNNImputer para numéricos.
            n_neighbors (int): Número de vecinos para KNNImputer.
            encoding_method (str): Método de encoding para categóricas.

        Returns:
            tuple: (ID del modelo, métricas de evaluación).
        """
        if conjunto_variables is None:
            if not self.conjuntos_variables:
                raise ValueError(
                    "No hay conjuntos de variables preparados. Ejecute preparar_datos() primero.")
            conjunto_variables = list(self.conjuntos_variables.keys())[-1]

        conjunto = self.conjuntos_variables[conjunto_variables]
        X_train, X_test = conjunto['X_train'], conjunto['X_test']
        y_train, y_test = conjunto['y_train'], conjunto['y_test']

        preprocessor = self._crear_pipeline(
            conjunto['columnas_numericas'], conjunto['columnas_categoricas'],
            estrategia_imputacion_num, estrategia_imputacion_cat, usar_knn,
            n_neighbors, encoding_method
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', modelo)
        ])

        if self.verbose:
            print(
                f"Entrenando modelo {nombre_modelo} en conjunto '{conjunto_variables}'...")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
            if self.verbose:
                print("El modelo no soporta predict_proba.")

        metricas = self._calcular_metricas(y_test, y_pred, y_prob)
        id_modelo = f"{nombre_modelo}_{conjunto_variables}"
        self.modelos[id_modelo] = pipeline
        self.resultados[id_modelo] = metricas

        if self.verbose:
            print(f"Resultados para {id_modelo}:")
            for k, v in metricas.items():
                if k != 'confusion_matrix':
                    print(f"  {k}: {v:.4f}")
            print("Matriz de confusión:")
            print(metricas['confusion_matrix'])

        return id_modelo, metricas

    def entrenar_con_grid(self, modelo, nombre_modelo: str,
                          conjunto_variables: str = None,
                          param_grid: dict = None, cv: int = 5,
                          scoring: str = 'f1',
                          estrategia_imputacion_num: str = 'median',
                          estrategia_imputacion_cat: str = 'most_frequent',
                          usar_knn: bool = False, n_neighbors: int = 5,
                          encoding_method: str = 'onehot') -> tuple:
        """
        Entrena un modelo utilizando GridSearchCV para ajustar hiperparámetros.

        Args:
            modelo: Instancia del modelo a entrenar.
            nombre_modelo (str): Nombre identificativo del modelo.
            conjunto_variables (str): Nombre del conjunto de variables a usar.
            param_grid (dict): Diccionario de hiperparámetros para GridSearchCV.
            cv (int): Número de folds para validación cruzada.
            scoring (str): Métrica para optimizar.
            estrategia_imputacion_num (str): Estrategia para imputación numérica.
            estrategia_imputacion_cat (str): Estrategia para imputación categórica.
            usar_knn (bool): Si se usa KNNImputer para numéricos.
            n_neighbors (int): Número de vecinos para KNNImputer.
            encoding_method (str): Método de encoding para categóricas.

        Returns:
            tuple: (ID del modelo, métricas de evaluación).
        """
        if conjunto_variables is None:
            if not self.conjuntos_variables:
                raise ValueError(
                    "No hay conjuntos de variables preparados. Ejecute preparar_datos() primero.")
            conjunto_variables = list(self.conjuntos_variables.keys())[-1]

        conjunto = self.conjuntos_variables[conjunto_variables]
        X_train, X_test = conjunto['X_train'], conjunto['X_test']
        y_train, y_test = conjunto['y_train'], conjunto['y_test']

        preprocessor = self._crear_pipeline(
            conjunto['columnas_numericas'], conjunto['columnas_categoricas'],
            estrategia_imputacion_num, estrategia_imputacion_cat, usar_knn,
            n_neighbors, encoding_method
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', modelo)
        ])

        if self.verbose:
            print(
                f"Entrenando modelo {nombre_modelo} con GridSearchCV en conjunto '{conjunto_variables}'...")

        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring,
                            n_jobs=-1, verbose=1 if self.verbose else 0)
        grid.fit(X_train, y_train)

        best_pipeline = grid.best_estimator_
        y_pred = best_pipeline.predict(X_test)
        try:
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None
            if self.verbose:
                print("El modelo no soporta predict_proba.")

        metricas = self._calcular_metricas(y_test, y_pred, y_prob)
        id_modelo = f"{nombre_modelo}_{conjunto_variables}"
        self.modelos[id_modelo] = best_pipeline
        self.resultados[id_modelo] = metricas

        if self.verbose:
            print(
                f"Mejores parámetros para {nombre_modelo}: {grid.best_params_}")
            print(f"Resultados para {id_modelo}:")
            for k, v in metricas.items():
                if k != 'confusion_matrix':
                    print(f"  {k}: {v:.4f}")
            print("Matriz de confusión:")
            print(metricas['confusion_matrix'])

        return id_modelo, metricas

    def predecir(self, datos: pd.DataFrame, id_modelo: str) -> np.ndarray:
        """
        Realiza predicciones usando el modelo identificado por id_modelo.

        Args:
            datos (pd.DataFrame): DataFrame con los datos a predecir.
            id_modelo (str): Identificador del modelo a usar.

        Returns:
            np.ndarray: Array con las predicciones.
        """
        if id_modelo not in self.modelos:
            raise ValueError(f"El modelo {id_modelo} no existe.")
        pipeline = self.modelos[id_modelo]
        return pipeline.predict(datos)
