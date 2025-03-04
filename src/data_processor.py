import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


class DataPreprocessor:
    """
    Clase para preprocesar un DataFrame aplicando diversas transformaciones:
      - Conversión de columnas a tipo datetime.
      - Manejo de valores nulos en la columna 'warranty'.
      - Conversión de precios de USD a ARS utilizando tasas de cambio.
      - Cálculo de la diferencia en días entre dos fechas.
      - Conteo de productos vendidos por cada vendedor.
      - Homologación de métodos de pago.
    """

    def __init__(self, exchange_rates: Optional[Dict[int, float]] = None):
        """
        Inicializa el preprocesador de datos.

        Args:
            exchange_rates: Diccionario con tasas de cambio por año.
                Si no se proporciona, se usa el siguiente mapeo por defecto:
                    {2013: 4.96, 2014: 7.87, 2015: 8.55}
        """
        if exchange_rates is None:
            self.exchange_rates = {
                2013: 4.96,
                2014: 7.87,
                2015: 8.55
            }
        else:
            self.exchange_rates = exchange_rates

    def convert_to_datetime(self, df: pd.DataFrame, col: str,
                            unit: Optional[str] = None) -> pd.DataFrame:
        """
        Convierte una columna a tipo datetime.
        Si 'unit' no es None, asume que los valores están en formato Unix/Epoch.

        Args:
            df: DataFrame a procesar.
            col: Nombre de la columna a convertir.
            unit: Unidad para la conversión, en caso de ser necesario.

        Returns:
            DataFrame con la columna convertida a datetime.
        """
        if unit:
            df[col] = pd.to_datetime(df[col], unit=unit, errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def handle_nulls_warranty(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea la variable 'has_warranty' a partir de la columna 'warranty'.

        Args:
            df: DataFrame a procesar.

        Returns:
            DataFrame con la nueva columna 'has_warranty'.
        """
        df["has_warranty"] = df["warranty"].notnull().astype(int)
        return df

    def convert_to_ars(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte los precios en USD a ARS según la tasa de cambio del año correspondiente.

        Args:
            df: DataFrame con las columnas 'price', 'currency_id' y 'date_created'.

        Returns:
            DataFrame con la columna 'price_ars' con los precios convertidos
            y una columna 'year_created' extraída de 'date_created'.
        """
        # Asegurarse de que 'date_created' es de tipo datetime
        if not pd.api.types.is_datetime64_any_dtype(df['date_created']):
            df = self.convert_to_datetime(df, 'date_created')

        df['year_created'] = df['date_created'].dt.year

        def convert_price(row):
            if row["currency_id"] == "USD":
                return row["price"] * self.exchange_rates.get(
                    row["year_created"], np.nan)
            return row["price"]

        df["price_ars"] = df.apply(convert_price, axis=1)
        return df

    def create_date_diff(self, df: pd.DataFrame, col1: str = "date_created",
                         col2: str = "last_updated",
                         new_col: str = "diff_created_updated") -> pd.DataFrame:
        """
        Crea una nueva columna con la diferencia en días entre dos columnas de tipo datetime.

        Args:
            df: DataFrame a procesar.
            col1: Columna de fecha inicial.
            col2: Columna de fecha final.
            new_col: Nombre de la nueva columna que contendrá la diferencia.

        Returns:
            DataFrame con la nueva columna de diferencia en días.
        """
        # Asegurarse de que ambas columnas sean datetime
        if not pd.api.types.is_datetime64_any_dtype(df[col1]):
            df = self.convert_to_datetime(df, col1)
        if not pd.api.types.is_datetime64_any_dtype(df[col2]):
            df = self.convert_to_datetime(df, col2)

        df[new_col] = (df[col2] - df[col1]).dt.total_seconds() / (3600 * 24)
        return df

    def create_sell_products(self, df: pd.DataFrame,
                             group_col: str = "seller_id",
                             new_col: str = "sell_products") -> pd.DataFrame:
        """
        Cuenta cuántos productos vende cada vendedor y asigna el valor a cada fila.

        Args:
            df: DataFrame a procesar.
            group_col: Columna por la cual agrupar, por ejemplo 'seller_id'.
            new_col: Nombre de la nueva columna que contendrá el conteo de productos.

        Returns:
            DataFrame con la nueva columna de conteo.
        """
        seller_counts = df.groupby(group_col).size().rename(new_col)
        df = df.merge(seller_counts, on=group_col, how="left")
        return df

    def homologar_metodos_pago(self, df: pd.DataFrame,
                               metodos_pago_map: Optional[
                                   Dict[str, str]] = None) -> pd.DataFrame:
        """
        Homologa los métodos de pago presentes en la columna
        'non_mercado_pago_payment_methods_description' usando un diccionario de mapeo.

        Args:
            df: DataFrame a procesar.
            metodos_pago_map: Diccionario para mapear cada categoría. Si no se proporciona,
                              se utiliza un mapeo por defecto.

        Returns:
            DataFrame con la nueva columna 'payment_method_grouped' que contiene la categoría homologada.
            Los valores no mapeados se asignan a 'otros'.
        """
        if metodos_pago_map is None:
            metodos_pago_map = {
                'Transferencia bancaria': 'transferencia',
                'Efectivo': 'efectivo',
                'Acordar con el comprador': 'acordar',
                'Tarjeta de crédito': 'tarjeta',
                'MasterCard': 'tarjeta',
                'Mastercard Maestro': 'tarjeta',
                'Visa Electron': 'tarjeta',
                'Visa': 'tarjeta',
                'Contra reembolso': 'efectivo',
                'Cheque certificado': 'cheque',
                'Giro postal': 'giro_postal'
            }
        df['payment_method_grouped'] = df[
            'non_mercado_pago_payment_methods_description'].map(
            metodos_pago_map)
        df['payment_method_grouped'] = df['payment_method_grouped'].fillna(
            'otros')
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica todas las transformaciones de preprocesamiento definidas.

        Args:
            df: DataFrame a preprocesar.

        Returns:
            DataFrame preprocesado.
        """
        df = self.convert_to_datetime(df, 'date_created')
        df = self.convert_to_datetime(df, 'last_updated')
        df = self.convert_to_datetime(df, "start_time", unit="ms")
        df = self.convert_to_datetime(df, "stop_time", unit="ms")
        df = self.handle_nulls_warranty(df)
        df = self.convert_to_ars(df)
        df = self.create_date_diff(df, col1="start_time", col2="stop_time",
                                   new_col="diff_start_stp_time")
        df = self.create_date_diff(df, col1="date_created", col2="last_updated",
                              new_col="diff_created_updated")
        df = self.create_sell_products(df, group_col="seller_id",
                                       new_col="sell_products")
        df = self.homologar_metodos_pago(df)
        return df