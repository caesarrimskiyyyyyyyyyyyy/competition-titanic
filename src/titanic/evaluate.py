from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


# Единая метрика для сравнения экспериментов.
_SCORING = 'accuracy'


# Единая схема кросс-валидации для всех моделей
# и всех экспериментов с гиперпараметрами.
#
# StratifiedKFold сохраняет приблизительно одинаковое
# соотношение классов на каждом фолде.
_CROSS_VALIDATION = StratifiedKFold(
    n_splits=5,
    shuffle=True,

    # Фиксируем конкретное разбиение, чтобы
    # результаты разных экспериментов можно было сравнивать.
    random_state=42
)


def evaluate(pipeline, X, y, fit_params=None, n_jobs=-1):
    """
    Оценивает полный pipeline с помощью кросс-валидации.

    На каждом фолде отдельно выполняются преобразования
    данных и обучение модели.
    """

    # Запускаем кросс-валидацию и получаем
    # accuracy отдельно на каждом validation-фолде.
    scores = cross_val_score(

        # Полный pipeline: признаки, preprocessing и модель.
        estimator=pipeline,

        # Исходные признаки.
        X=X,

        # Целевая переменная.
        y=y,

        # Одинаковые фолды для всех экспериментов.
        cv=_CROSS_VALIDATION,

        # Единая метрика качества.
        scoring=_SCORING,

        # Для обычных моделей используем все ядра,
        # а для PyTorch передадим n_jobs=1.
        n_jobs=n_jobs,

        # Дополнительные параметры для pipeline.fit().
        # Для обычных моделей здесь остаётся None.
        params=fit_params
    )

    # Объединяем результаты фолдов в одну строку.
    formatted_scores = ', '.join(
        f'{score:.3f}'
        for score in scores
    )

    # Выводим качество и его разброс.
    print(f'Accuracy по фолдам: {formatted_scores}')
    print(f'Средняя accuracy: {scores.mean():.3f}')
    print(f'Std accuracy: {scores.std():.3f}')

    return scores


def run_grid_search(pipeline, param_grid, X, y, fit_params=None, n_jobs=-1):
    """
    Подбирает гиперпараметры полного pipeline.
    """

    # Создаём поиск по переданной сетке гиперпараметров.
    search = GridSearchCV(

        # Полный pipeline.
        estimator=pipeline,

        # Набор гиперпараметров для перебора.
        param_grid=param_grid,

        # Единая метрика качества.
        scoring=_SCORING,

        # Одинаковые фолды для всех экспериментов.
        cv=_CROSS_VALIDATION,

        # Параллелизм выбирается при запуске эксперимента.
        n_jobs=n_jobs,

        # Сохраняем train score для анализа переобучения.
        return_train_score=True,
    )

    # Передаём дополнительные параметры в pipeline.fit().
    search.fit(X, y, **(fit_params or {}))

    return search