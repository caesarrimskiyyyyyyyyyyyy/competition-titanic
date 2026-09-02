from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor():
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
        OneHotEncoder(handle_unknown='ignore')
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