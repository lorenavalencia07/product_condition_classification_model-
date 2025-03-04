import pandas as pd
import yaml
import joblib
from src.data_loader import DataLoader
from src.data_processor import DataPreprocessor


class ModelPredictor:
    """
    Clase para realizar predicciones utilizando un modelo guardado.

    Esta clase:
     - Carga un modelo previamente guardado en un archivo .pkl.
     - Carga nuevos datos desde un archivo .jsonlines (con el mismo esquema de
       los datos de entrenamiento) usando la clase DataLoader.
     - Procesa los datos utilizando la clase DataPreprocessor.
     - Carga la lista de columnas de entrenamiento desde un archivo YAML.
     - Retorna las predicciones para los nuevos datos.
    """

    def __init__(self, model_path: str, columns_config_path: str):
        """
        Inicializa la clase ModelPredictor con el modelo guardado y la configuración
        de columnas de entrenamiento.

        Args:
            model_path (str): Ruta del archivo .pkl donde se encuentra el modelo guardado.
            columns_config_path (str): Ruta del archivo YAML que contiene la lista de columnas de entrenamiento.
        """
        self.model = joblib.load(model_path)

        # Cargar la configuración de columnas desde el YAML
        with open(columns_config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.training_columns = config.get('columns', [])

        # Inicializar el preprocesador de datos
        self.data_processor = DataPreprocessor()

    def load_new_data(self, jsonlines_path: str,
                      schema_path: str) -> pd.DataFrame:
        """
        Carga nuevos datos desde un archivo .jsonlines utilizando la clase DataLoader.

        Args:
            jsonlines_path (str): Ruta del archivo .jsonlines con los nuevos datos.
            schema_path (str): Ruta del archivo JSON que define el esquema de los datos.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.
        """
        loader = DataLoader(jsonlines_path, schema_path)
        df_new = loader.load_data()
        return df_new

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Procesa los datos nuevos utilizando la clase DataPreprocessor.

        Args:
            df (pd.DataFrame): DataFrame con los datos nuevos.

        Returns:
            pd.DataFrame: DataFrame procesado.
        """
        df_processed = self.data_processor.preprocess(df)
        return df_processed

    def predict(self, new_data_jsonlines: str, schema_path: str) -> pd.Series:
        """
        Realiza predicciones sobre los nuevos datos cargados desde un archivo .jsonlines.

        Args:
            new_data_jsonlines (str): Ruta del archivo .jsonlines que contiene los nuevos datos.
            schema_path (str): Ruta del archivo JSON con el esquema de los datos.

        Returns:
            pd.Series: Serie con las predicciones.
        """
        # Cargar nuevos datos utilizando DataLoader
        df_new = self.load_new_data(new_data_jsonlines, schema_path)
        # Procesar los datos nuevos
        df_processed = self.prepare_data(df_new)
        # Seleccionar solo las columnas utilizadas en el entrenamiento
        df_features = df_processed[self.training_columns]
        # Realizar las predicciones
        predictions = self.model.predict(df_features)
        return pd.Series(predictions, index=df_features.index)


# Ejemplo de uso:
if __name__ == "__main__":
    predictor = ModelPredictor(
        model_path="data/article_condition_predictor.pkl",
        columns_config_path="data/training_columns.yaml"
    )
    preds = predictor.predict(
        new_data_jsonlines="data/new_data.jsonlines",
        schema_path="data/dataset_schema.json"
    )
    print("Predicciones:")
    print(preds.head())
