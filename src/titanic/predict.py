from pathlib import Path

from joblib import load as load_model

from titanic.data import load
from titanic.paths import FINAL_MODEL_PATH, FINAL_SUBMISSION_PATH


def predict(model_path: Path, output_path: Path):
    """
    Выполняет инференс на test-датасете
    и сохраняет submission для Kaggle.
    """

    # Загружаем test через общий метод.
    _, df_test = load()

    # Загружаем обученный pipeline.
    pipeline = load_model(model_path)

    # Pipeline самостоятельно применяет feature engineering,
    # preprocessing и затем выполняет предсказание.
    predictions = pipeline.predict(df_test)

    # Оставляем PassengerId и добавляем предсказанный класс.
    submission = (
        df_test
        .PassengerId
        .to_frame()
        .assign(Survived=predictions.astype(int))
    )

    # Создаём директорию, если её ещё нет.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Индекс DataFrame в submission не записываем.
    submission.to_csv(output_path, index=False)

    print(f'Создано предсказаний: {len(submission)}')
    print(f'Submission сохранён: {output_path}')

    return submission


def main():
    predict(
        model_path=FINAL_MODEL_PATH,
        output_path=FINAL_SUBMISSION_PATH
    )


if __name__ == '__main__':
    main()