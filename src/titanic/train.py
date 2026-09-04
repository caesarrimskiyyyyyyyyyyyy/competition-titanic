from argparse import ArgumentParser
from pathlib import Path

from hydra.utils import instantiate
from joblib import dump
from omegaconf import OmegaConf

from titanic.data import load
from titanic.paths import FINAL_CONFIG_PATH, FINAL_MODEL_PATH
from titanic.pipelines import build_pipeline


def train(config_path: Path, output_path: Path):
    """
    Обучает финальный pipeline на всем train-датасете
    и сохраняет его для последующего инференса.
    """

    # Загружаем оба датасета через общий метод.
    df_train, _ = load()

    # Отделяем целевую переменную.
    X = df_train.drop(columns='Survived')
    y = df_train['Survived']

    # Загружаем стабильную конфигурацию победившей модели.
    config = OmegaConf.load(config_path)

    # Создаем RandomForestClassifier из YAML.
    model = instantiate(config.model)

    # Pipeline содержит feature engineering,
    # preprocessing и саму модель.
    pipeline = build_pipeline(model)

    # После выбора модели обучаем ее на всем train.
    pipeline.fit(X, y)

    # Создаем директорию, если ее нет
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Сохраняем целиком обученный pipeline.
    dump(pipeline, output_path)

    print(f'Обучено пассажиров: {len(df_train)}')
    print(f'Модель сохранена: {output_path}')


def main():
    train(
        config_path=FINAL_CONFIG_PATH,
        output_path=FINAL_MODEL_PATH
    )

if __name__ == '__main__':
    main()