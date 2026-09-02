from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from titanic.features import make_features
from titanic.preprocessing import build_preprocessor


def build_pipeline(model):
    """
    Собирает полный sklearn pipeline.

    Порядок выполнения:
    1. создание новых признаков;
    2. preprocessing данных;
    3. переданная модель.
    """

    # Возвращаем sklearn пайплайн
    return make_pipeline(

        # Оборачиваем feature engineering 
        # функцию в sklearn transformer.
        FunctionTransformer(
            func=make_features,

            # Отключаем sklearn валидацию, чтобы сохранить
            # DF и обращаться к признакам по именам колонок.
            validate=False
        ),

        # Выполняем препроцессинг числовых
        # и категорильных признаков.
        build_preprocessor(),

        # Обучаем переданную модель
        # на подготовленных признаках.
        model
    )