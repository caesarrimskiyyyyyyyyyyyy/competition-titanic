from sklearn.model_selection import StratifiedKFold, cross_val_score


def evaluate(pipeline, X, y):
    """
    Оценивает полный piipeline с помощь кросс-валидации.

    На каждом этапе фолда отдельно обучаются: заполнение
    пропусков, масштабирование, OneHotEncoder, модель.
    """

    # StratifiedKFold сохраняет приблизительно одинаковое
    # соотношение классов на каждом фолде.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,

        # Фиксируем конкретное разбиение, чтобы
        # результаты экспериментов можно было сравнить.
        random_state=42
    )

    # Получаем accuracy отдельно на каждом validation фолде.
    scores = cross_val_score(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv,
        scoring='accuracy'
    )

    # Подготавливаем вывод результатов.
    formatted_scores = ', '.join(
        f'{score:.3f}'
        for score in scores
    )

    print(f'Accuracy по фолдам: {formatted_scores}')
    print(f'Средняя accuracy: {scores.mean():.3f}')
    print(f'Std accuracy: {scores.std():.3f}')

    return scores