import json
import pandas as pd
import numpy as np
from typing import Any, Dict


class DataLoader:
    """
    Clase para cargar y procesar datos desde un archivo JSON Lines
    a un DataFrame de pandas con la estructura aplanada.
    """

    def __init__(self, file_path: str, schema_path: str):
        """
        Inicializa la clase con la ruta al archivo de datos y al esquema.

        Args:
            file_path (str): Ruta al archivo JSON Lines con los datos.
            schema_path (str): Ruta al archivo JSON que define el esquema de los datos.
        """
        self.file_path = file_path
        self.schema_path = schema_path

    @staticmethod
    def flatten_dict(nested_dict: Dict[str, Any], parent_key: str = '',
                     separator: str = '_') -> Dict[str, Any]:
        """
        Aplana un diccionario anidado en un diccionario de una sola capa.

        Args:
            nested_dict (Dict[str, Any]): El diccionario que se desea aplanar.
            parent_key (str, optional): La clave padre para los valores anidados.
            separator (str, optional): El separador para concatenar las claves.

        Returns:
            Dict[str, Any]: Un diccionario aplanado.
        """
        items = []
        for key, value in nested_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, dict):
                items.extend(
                    DataLoader.flatten_dict(value, new_key, separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    @staticmethod
    def safe_convert_to_str(x):
        """
        Convierte de forma segura un valor a cadena, manejando None, NaN y listas.

        Args:
            x: El valor a convertir.

        Returns:
            str or np.nan: La representación en cadena o np.nan si no es posible.
        """
        if x is None:
            return np.nan
        if isinstance(x, (list, np.ndarray)):
            if len(x) == 0:
                return np.nan
            return str(x)
        if pd.isna(x):
            return np.nan
        return str(x)

    def load_data(self) -> pd.DataFrame:
        """
        Procesa el archivo JSON Lines usando el esquema definido y retorna
        un DataFrame de pandas con las columnas aplanadas y convertidas a los tipos apropiados.

        Returns:
            pd.DataFrame: DataFrame procesado.
        """
        # Cargar el esquema para entender la estructura de los datos
        with open(self.schema_path, 'r') as schema_file:
            schema = json.load(schema_file)

        # Leer el archivo JSON Lines
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():  # Saltar líneas vacías
                    data.append(json.loads(line))

        # Crear DataFrame inicial
        df = pd.DataFrame(data)

        # Identificar columnas que requieren ser aplanadas (diccionarios o listas)
        columns_to_flatten = []
        for col in df.columns:
            if col in schema and (
                    isinstance(schema[col], dict) or isinstance(schema[col],
                                                                list)):
                columns_to_flatten.append(col)

        # Procesar cada columna que necesita ser aplanada
        for col in columns_to_flatten:
            # Caso de diccionarios
            if isinstance(schema[col], dict):
                flattened_dicts = []
                for item in df[col]:
                    if pd.isna(item) or item is None:
                        flattened_dicts.append({})
                    else:
                        flattened_dicts.append(DataLoader.flatten_dict(item))
                flat_df = pd.DataFrame(flattened_dicts)
                if not flat_df.empty:
                    for new_col in flat_df.columns:
                        df[f"{col}_{new_col}"] = flat_df[new_col]

            # Caso de listas (pueden ser listas de diccionarios o listas simples)
            elif isinstance(schema[col], list):
                if schema[col] and isinstance(schema[col][0], dict):
                    sample_dict = schema[col][0]
                    keys = sample_dict.keys()
                    for key in keys:
                        df[f"{col}_{key}"] = df[col].apply(
                            lambda x: x[0][key] if isinstance(x,
                                                              list) and x and isinstance(
                                x[0], dict) and key in x[0] else np.nan
                        )
                else:
                    df[f"{col}_value"] = df[col].apply(
                        lambda x: x[0] if isinstance(x, list) and len(
                            x) == 1 else
                        np.nan if isinstance(x, list) and len(x) == 0 else
                        x if not isinstance(x, list) else
                        str(x)
                    )
            # Eliminar la columna original
            df = df.drop(columns=[col])

        # Convertir las columnas a tipos de datos apropiados
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
            if df[col].dtype == 'object':
                df[col] = df[col].apply(DataLoader.safe_convert_to_str)

        return df


# Ejemplo de uso de la clase
if __name__ == "__main__":
    input_file = "data/MLA_100k_checked_v3.jsonlines"
    schema_file = "data/dataset_schema.json"
    loader = DataLoader(input_file, schema_file)
    df = loader.load_data()
    print("Proceso completado.")
    print(f"Dimensiones del DataFrame: {df.shape}")
