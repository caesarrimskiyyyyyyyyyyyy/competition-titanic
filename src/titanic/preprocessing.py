import numpy as np

from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(sparse_output=True):
    """
    Создает preprocessing для числовых
    и категориальных признаков.
    """

    # Обработка числовых признаков.
    numeric_preprocessing = make_pipeline(

        # Заполняем пропуски медианой.
        SimpleImputer(strategy='median'),

        # Приводим числовые признаки  к единому масштабу.
        StandardScaler()
    )

    # Обработка категориальных признаков.
    categorial_preprocessing = make_pipeline(

        # Заполняем пропуски самой частой категорией.
        SimpleImputer(strategy='most_frequent'),

        # Преобразует категории в бинарные столбцы.
        OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output)
    )

    return make_column_transformer(
        # Все числовые признаки направляем
        # в числовой препроцессинг.
        (
            numeric_preprocessing,
            make_column_selector(dtype_include='number')
        ),

        # Все остальные признаки направляем
        # в категориальный препроцессинг.
        (
            categorial_preprocessing,
            make_column_selector(dtype_exclude='number')
        )
    )


def prepare_catboost_data(data):
    """
    Подготавливает категориальные признаки для CatBoost.

    Числовые пропуски CatBoost обрабатывает самостоятельно.
    Пропуски в категориальных признаках должны быть
    представлены отдельным строковым значением.
    """

    # В Embarked есть два пропуска. Представляем
    # их отдельной категорией Unknown.
    return data.assign(
        Embarked=data.Embarked.fillna('Unknown')
    )


def prepare_neural_data(data):
    """
    Преобразует результат препроцессинга в формат PyTorch.

    Нейросеть получает плотную матрицу float32 вместо
    стандартного для sklearn массива float64.
    """

    return data.astype(
        np.float32,
        copy=False
    )