import yaml

def load_training_columns(yaml_file_path: str) -> list:
    """
    Carga la lista de columnas de entrenamiento desde un archivo YAML.

    Args:
        yaml_file_path (str): Ruta del archivo YAML.

    Returns:
        list: Lista de columnas definidas en el YAML.
    """
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('columns', [])

# Ejemplo de uso:
if __name__ == "__main__":
    columns = load_training_columns("data/training_columns.yaml")
    print("Training columns:", columns)