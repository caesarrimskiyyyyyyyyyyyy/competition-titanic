from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from titanic.features import make_features
from titanic.preprocessing import build_preprocessor, prepare_catboost_data, prepare_neural_data


def build_pipeline(model):
    """
    Собирает полный sklearn pipeline.

    Порядок выполнения:
    1. создание новых признаков;
    2. preprocessing данных;
    3. переданная модель.
    """

    # Возвращаем sklearn пайплайн.
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


def build_catboost_pipeline(model):
    """
    Собирает полный pipeline для CatBoost.

    CatBoost получает категориальные признаки напрямую,
    поэтому OneHotEncoder и StandardScaler не используются.

    Порядок выполнения:
    1. создание новых признаков;
    2. подготовка категорий для CatBoost;
    3. обучение CatBoost.
    """

    return make_pipeline(
        # Используем то же feature engineering,
        # что и для остальных моделей.
        FunctionTransformer(
            func=make_features,

            # Сохраняем DataFrame и названия колонок.
            validate=False
        ),

        # Заполняем пропуски в категориальных признаках.
        FunctionTransformer(
            func=prepare_catboost_data,

            # CatBoost должен получить pandas DataFrame,
            # чтобы находить категории по именам колонок.
            validate=False
        ),

        # CatBoost самостоятельно обрабатывает
        # числовые и категориальные признаки.
        model
    )

def build_neural_pipeline(model):
    """
    Собирает полный pipeline для нейронной сети.
    """

    return make_pipeline(
        FunctionTransformer(make_features),

        # PyTorch не работает с разреженной матрицей,
        # поэтому OneHotEncoder возвращает dense-результат.
        build_preprocessor(sparse_output=False),

        # PyTorch ожидает float32.
        FunctionTransformer(prepare_neural_data),

        model
    )