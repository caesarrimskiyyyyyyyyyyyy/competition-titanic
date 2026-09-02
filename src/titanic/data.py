import pandas as pd

from titanic.paths import RAW_DATA_DIR


def load():
    """
    Загружает исходные train и test датасеты.
    """

    df_train = pd.read_csv(RAW_DATA_DIR/'train.csv')
    df_test = pd.read_csv(RAW_DATA_DIR/'test.csv')
    return df_train, df_test